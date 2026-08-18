"""One chat interface over several model providers.

The local model is fast, free and private, and it is also the reason
answers sometimes read as vague or invented — a 1.5B model asked to
summarize a desktop will make things up. Rather than replacing it, this
lets a better model be selected per *job*: a cheap local model for the
spoken back-and-forth where latency dominates, something stronger for
typed questions, and either for reading the screen.

Providers:

- `ollama`  — local, free, private. Always available, always the fallback.
- `groq`    — OpenAI-compatible API, very fast hosted inference.
- `gemini`  — Google's generativelanguage API.

Both cloud providers are plain HTTPS with urllib, deliberately: the
`groq` and `google-generativeai` SDKs are large dependency trees for what
is one POST each, and this project already talks to Ollama, Sarvam and
Gemini-vision the same way.

Cost rules are unchanged. `default_backend` stays local, the Claude Code
handoff still confirms, and a cloud model is only ever used for a role the
user has explicitly assigned it to in the dashboard — an assignment they
can see and change at any time, which is a clearer consent model than
asking per turn. Keys live in max/.env (gitignored), never in
config.json, which is a plain preferences file.
"""
import base64
import json
import urllib.error
import urllib.request

from max import env, ollama_client

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models"

# The vision model for looking at the screen. NOT gemini-2.0-flash, which
# this project defaulted to for months and which now answers "no longer
# available to new users" — a default that fails closed on a live key is
# worse than no default. Verified against the key in use: gemini-2.5-flash
# and 2.5-pro are also retired for new users; gemini-3-flash-preview reads
# an actual chess position correctly, and the 3.1-lite models do not (they
# answer "play e4" on a board where e4 has already been played).
DEFAULT_VISION_MODEL = "gemini-3-flash-preview"

# Which .env key each provider needs. A provider whose key is missing is
# still listed in the dashboard, marked unavailable, so the reason a model
# can't be picked is visible rather than mysterious.
PROVIDER_KEYS = {"groq": "GROQ_API_KEY", "gemini": "GEMINI_API_KEY"}

PROVIDER_LABELS = {"ollama": "Max", "groq": "Groq", "gemini": "Gemini"}

# Roles a model can be assigned to. Kept small and concrete — these are
# the three places a model actually answers.
ROLES = {
    "chat": "Typed conversation (terminal and web UI)",
    "voice": "Spoken replies, where latency matters most",
    "screen": "Questions about what's on screen",
    # Distils conversations into memory/profile.md. Runs in the
    # background, on nobody's critical path, so the cheapest capable model
    # is usually the right choice — and it stays local unless you say
    # otherwise, since it reads every conversation you have.
    "memory": "Learning your profile from conversations",
}


class LLMError(Exception):
    pass


def _api_key(provider: str) -> str:
    key_name = PROVIDER_KEYS.get(provider)
    if not key_name:
        return ""
    return env.load_env().get(key_name, "").strip()


def provider_available(provider: str) -> bool:
    return provider == "ollama" or bool(_api_key(provider))


def catalog(cfg: dict) -> list[dict]:
    """Every configured model, each marked with whether it can be used."""
    models = []
    for entry in cfg.get("models", []):
        provider = entry.get("provider", "ollama")
        models.append({
            "id": entry.get("id") or entry.get("model", ""),
            "provider": provider,
            "model": entry.get("model", ""),
            "label": entry.get("label") or entry.get("model", ""),
            "local": provider == "ollama",
            "available": provider_available(provider),
            "needs_key": "" if provider_available(provider) else PROVIDER_KEYS.get(provider, ""),
        })
    return models


def resolve(cfg: dict, role: str = "chat") -> dict:
    """The model assigned to `role`, or the local default.

    Falls back rather than failing when an assignment can't be honoured —
    a key that was removed, or a model id deleted from the catalog, should
    degrade to the local model that always works, not take the assistant
    offline.
    """
    wanted = (cfg.get("model_roles") or {}).get(role)
    entries = {m["id"]: m for m in catalog(cfg)}
    entry = entries.get(wanted)
    if entry and entry["available"] and entry["model"]:
        return entry
    return {
        "id": "local",
        "provider": "ollama",
        "model": cfg["ollama_model"],
        "label": f"Max · {cfg['ollama_model']}",
        "local": True,
        "available": True,
        "needs_key": "",
    }


def backend_name(entry: dict) -> str:
    """Backend string for logging and reply labels. Local stays plain
    "ollama" so nothing downstream that already knows that name changes."""
    if entry["provider"] == "ollama":
        return "ollama"
    return f"{entry['provider']}:{entry['model']}"


# --- Providers ----------------------------------------------------------
# Each takes the same OpenAI-shaped `messages` and returns the reply text,
# calling stream_callback with pieces as they arrive when it can.


# Cloudflare sits in front of Groq's API and rejects requests that arrive
# with no User-Agent — urllib doesn't send one, so every call came back
# "HTTP 403: error code: 1010" and looked exactly like a rejected API key.
# It cost a real debugging detour: the key was valid the whole time.
# Identifying ourselves is also just good manners for an API client.
_USER_AGENT = "Max/1.0 (macOS; +https://github.com/javedsaikia/JarvisOS)"


def _post(url: str, body: dict, headers: dict, timeout: int, stream: bool = False):
    request = urllib.request.Request(
        url,
        data=json.dumps(body).encode(),
        headers={
            "Content-Type": "application/json",
            "User-Agent": _USER_AGENT,
            **headers,
        },
    )
    try:
        return urllib.request.urlopen(request, timeout=timeout)
    except urllib.error.HTTPError as e:
        detail = ""
        try:
            detail = json.loads(e.read()).get("error", {}).get("message", "")
        except Exception:
            pass
        raise LLMError(f"HTTP {e.code}{': ' + detail if detail else ''}") from e
    except urllib.error.URLError as e:
        raise LLMError(f"could not reach the API ({e.reason})") from e
    # A timeout while *reading* the response is raised bare, not wrapped in
    # URLError — so without this a slow cloud model escaped as something
    # chat()'s callers don't catch, and handle_ollama's whole point (fall
    # back to the local model rather than lose the turn) never ran. The
    # 8s cloud timeout above makes this the likely failure, not a rare one.
    except TimeoutError as e:
        raise LLMError(f"the API timed out ({e})") from e


def _sse_lines(response):
    """Yield decoded JSON objects from a text/event-stream response."""
    for raw in response:
        line = raw.decode("utf-8", "replace").strip()
        if not line.startswith("data:"):
            continue
        payload = line[5:].strip()
        if not payload or payload == "[DONE]":
            continue
        try:
            yield json.loads(payload)
        except json.JSONDecodeError:
            continue


# Models that think before they answer, and bill the thinking to the same
# token budget as the reply. Left alone they spend the whole cap on hidden
# reasoning and return an EMPTY message — measured on gpt-oss-20b at our
# 80-token voice cap, which would have been a silent dead turn every time.
# reasoning_effort="low" brings a one-sentence answer in 33-44 tokens.
_REASONING_MODELS = ("gpt-oss", "qwen3", "o1", "o3")


def _is_reasoning_model(model: str) -> bool:
    lowered = (model or "").lower()
    return any(tag in lowered for tag in _REASONING_MODELS)


def _chat_groq(messages, model, options, stream_callback, timeout):
    key = _api_key("groq")
    if not key:
        raise LLMError("no GROQ_API_KEY in max/.env")
    body = {"model": model, "messages": messages}
    if options and options.get("num_predict"):
        body["max_tokens"] = options["num_predict"]
    if _is_reasoning_model(model):
        body["reasoning_effort"] = "low"
    headers = {"Authorization": f"Bearer {key}"}

    if stream_callback is None:
        with _post(GROQ_URL, body, headers, timeout) as response:
            data = json.loads(response.read())
        try:
            content = (data["choices"][0]["message"].get("content") or "").strip()
        except (KeyError, IndexError):
            raise LLMError("no reply in the response")
        if not content:
            # A reasoning model that spent its whole budget thinking.
            # Raising lets the caller fall back instead of speaking "".
            raise LLMError(
                f"{model} returned an empty reply — the token budget went on "
                "reasoning; raise num_predict for this call"
            )
        return content

    body["stream"] = True
    pieces = []
    with _post(GROQ_URL, body, headers, timeout, stream=True) as response:
        for event in _sse_lines(response):
            try:
                piece = event["choices"][0]["delta"].get("content") or ""
            except (KeyError, IndexError):
                continue
            if piece:
                pieces.append(piece)
                stream_callback(piece)
    return "".join(pieces).strip()


def _gemini_payload(messages):
    """OpenAI-shaped messages -> Gemini's contents/system_instruction."""
    system = "\n\n".join(m["content"] for m in messages if m["role"] == "system")
    contents = [
        {"role": "model" if m["role"] == "assistant" else "user",
         "parts": [{"text": m["content"]}]}
        for m in messages if m["role"] != "system"
    ]
    payload = {"contents": contents}
    if system:
        payload["system_instruction"] = {"parts": [{"text": system}]}
    return payload


def look(prompt: str, image_png: bytes, cfg: dict, timeout: int = 45) -> str:
    """Ask a vision model about an actual screenshot.

    Every other path here works from OCR text, which is the right tool for
    a menu bar and the wrong one for anything positional. Asked what to
    play next in Chess.app, a text model got 48 lines of OCR that say
    nothing about where the pieces are, and answered with the only thing
    it could: "control the centre and develop your pieces". True of every
    chess position ever played, and therefore useless.

    Gemini is the vision-capable model already configured here
    (screen_cloud_model), so this sends it the pixels instead.
    """
    if not image_png:
        raise LLMError("no image to look at")
    key = _api_key("gemini")
    if not key:
        return _look_local(prompt, image_png, cfg, timeout)
    model = cfg.get("screen_cloud_model") or DEFAULT_VISION_MODEL
    body = {
        "contents": [{
            "role": "user",
            "parts": [
                {"text": prompt},
                {"inline_data": {"mime_type": "image/png",
                                 "data": base64.b64encode(image_png).decode("ascii")}},
            ],
        }],
        # Gemini 3 thinks before answering and bills the wait to the user.
        # Measured on a real chess position, same screenshot, same
        # question: default 5.9s but 22-51s when the service is busy,
        # thinkingLevel=low 46.7s, thinkingBudget=0 4.0s — and all three
        # returned the SAME move with the same reasoning. The thinking was
        # buying nothing here and costing everything, so it is off.
        "generationConfig": {"thinkingConfig": {"thinkingBudget": 0}},
    }
    url = f"{GEMINI_URL}/{model}:generateContent?key={key}"
    try:
        with _post(url, body, {}, timeout) as response:
            data = json.loads(response.read())
    except LLMError as e:
        # A dead key should not mean Max cannot look at all — there is a
        # vision model on this machine.
        print(f"[llm] Gemini vision unavailable ({e}) — using the local one", flush=True)
        return _look_local(prompt, image_png, cfg, timeout)
    text = _gemini_text(data).strip()
    if not text:
        raise LLMError(f"{model} returned nothing for the screenshot")
    return text


def local_vision_model(cfg: dict) -> str:
    """A locally installed model that accepts images, or "" if none is."""
    configured = (cfg.get("screen_local_vision_model") or "").strip()
    host = (cfg.get("ollama_host") or "http://localhost:11434").rstrip("/")
    try:
        with urllib.request.urlopen(f"{host}/api/tags", timeout=5) as response:
            installed = {m.get("name", "") for m in json.loads(response.read()).get("models", [])}
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError):
        return configured
    if configured and configured in installed:
        return configured
    for name in installed:
        if any(tag in name.lower() for tag in ("moondream", "llava", "vl", "vision", "bakllava")):
            return name
    return ""


def _look_local(prompt: str, image_png: bytes, cfg: dict, timeout: int) -> str:
    """Vision without the network. Small, but it genuinely sees the pixels."""
    model = local_vision_model(cfg)
    if not model:
        raise LLMError(
            "no vision model available — set GEMINI_API_KEY in max/.env, or "
            "install one locally with `ollama pull moondream`"
        )
    body = {
        "model": model,
        "prompt": prompt,
        "images": [base64.b64encode(image_png).decode("ascii")],
        "stream": False,
    }
    host = (cfg.get("ollama_host") or "http://localhost:11434").rstrip("/")
    with _post(f"{host}/api/generate", body, {}, timeout) as response:
        data = json.loads(response.read())
    text = (data.get("response") or "").strip()
    if not text:
        raise LLMError(f"{model} returned nothing for the screenshot")
    return text


def _gemini_text(data) -> str:
    try:
        parts = data["candidates"][0]["content"]["parts"]
    except (KeyError, IndexError):
        return ""
    return "".join(p.get("text", "") for p in parts)


def _chat_gemini(messages, model, options, stream_callback, timeout):
    key = _api_key("gemini")
    if not key:
        raise LLMError("no GEMINI_API_KEY in max/.env")
    body = _gemini_payload(messages)
    if options and options.get("num_predict"):
        body["generationConfig"] = {"maxOutputTokens": options["num_predict"]}

    if stream_callback is None:
        url = f"{GEMINI_URL}/{model}:generateContent?key={key}"
        with _post(url, body, {}, timeout) as response:
            data = json.loads(response.read())
        text = _gemini_text(data).strip()
        if not text:
            raise LLMError("no reply in the response")
        return text

    url = f"{GEMINI_URL}/{model}:streamGenerateContent?alt=sse&key={key}"
    pieces = []
    with _post(url, body, {}, timeout, stream=True) as response:
        for event in _sse_lines(response):
            piece = _gemini_text(event)
            if piece:
                pieces.append(piece)
                stream_callback(piece)
    return "".join(pieces).strip()


def chat(
    messages: list[dict],
    entry: dict,
    cfg: dict,
    options: dict | None = None,
    stream_callback=None,
    timeout: int = 120,
) -> str:
    """Send `messages` to the model in `entry`. Raises LLMError on failure."""
    provider = entry["provider"]
    model = entry["model"]
    # Cloud with no network used to sit on the default 120s socket, so a
    # spoken turn looked dead. Fail fast and let handle_ollama fall local.
    if provider in ("groq", "gemini"):
        timeout = min(int(timeout or 120), 8)

    if provider == "ollama":
        try:
            if stream_callback is None:
                return ollama_client.chat(
                    messages, model, cfg["ollama_host"], options=options,
                    keep_alive=cfg.get("ollama_keep_alive"), timeout=timeout,
                )
            pieces = []
            for chunk in ollama_client.chat_stream(
                messages, model, cfg["ollama_host"], options=options,
                keep_alive=cfg.get("ollama_keep_alive"), timeout=timeout,
            ):
                message = chunk.get("message")
                piece = (message or {}).get("content") if isinstance(message, dict) else ""
                piece = piece or chunk.get("response") or ""
                if piece:
                    pieces.append(piece)
                    stream_callback(piece)
            return "".join(pieces).strip()
        except ollama_client.OllamaError as e:
            raise LLMError(str(e)) from e

    # Reads of the response body happen outside _post (streaming especially,
    # which stays inside the `with` for the whole generation), so the same
    # bare-TimeoutError hole exists here too. Caught once around the
    # dispatch rather than at four separate read sites.
    try:
        if provider == "groq":
            return _chat_groq(messages, model, options, stream_callback, timeout)
        if provider == "gemini":
            return _chat_gemini(messages, model, options, stream_callback, timeout)
    except TimeoutError as e:
        raise LLMError(f"the API timed out ({e})") from e
    raise LLMError(f"unknown provider {provider!r}")
