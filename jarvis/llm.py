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
asking per turn. Keys live in jarvis/.env (gitignored), never in
config.json, which is a plain preferences file.
"""
import json
import urllib.error
import urllib.request

from jarvis import env, ollama_client

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models"

# Which .env key each provider needs. A provider whose key is missing is
# still listed in the dashboard, marked unavailable, so the reason a model
# can't be picked is visible rather than mysterious.
PROVIDER_KEYS = {"groq": "GROQ_API_KEY", "gemini": "GEMINI_API_KEY"}

PROVIDER_LABELS = {"ollama": "Orin", "groq": "Groq", "gemini": "Gemini"}

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
        "label": f"Orin · {cfg['ollama_model']}",
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


def _post(url: str, body: dict, headers: dict, timeout: int, stream: bool = False):
    request = urllib.request.Request(
        url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json", **headers}
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


def _chat_groq(messages, model, options, stream_callback, timeout):
    key = _api_key("groq")
    if not key:
        raise LLMError("no GROQ_API_KEY in jarvis/.env")
    body = {"model": model, "messages": messages}
    if options and options.get("num_predict"):
        body["max_tokens"] = options["num_predict"]
    headers = {"Authorization": f"Bearer {key}"}

    if stream_callback is None:
        with _post(GROQ_URL, body, headers, timeout) as response:
            data = json.loads(response.read())
        try:
            return data["choices"][0]["message"]["content"].strip()
        except (KeyError, IndexError):
            raise LLMError("no reply in the response")

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


def _gemini_text(data) -> str:
    try:
        parts = data["candidates"][0]["content"]["parts"]
    except (KeyError, IndexError):
        return ""
    return "".join(p.get("text", "") for p in parts)


def _chat_gemini(messages, model, options, stream_callback, timeout):
    key = _api_key("gemini")
    if not key:
        raise LLMError("no GEMINI_API_KEY in jarvis/.env")
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

    if provider == "groq":
        return _chat_groq(messages, model, options, stream_callback, timeout)
    if provider == "gemini":
        return _chat_gemini(messages, model, options, stream_callback, timeout)
    raise LLMError(f"unknown provider {provider!r}")
