"""Minimal Ollama chat client over its local REST API. Stdlib only — no extra deps."""
import json
import urllib.error
import urllib.request


class OllamaError(Exception):
    pass


def chat_message(
    messages: list[dict],
    model: str,
    host: str,
    tools: list[dict] = None,
    options: dict = None,
    timeout: int = 120,
    keep_alive: str = None,
) -> dict:
    """Return the full response message dict (role, content, and — on models
    whose template supports it — tool_calls)."""
    url = f"{host.rstrip('/')}/api/chat"
    body = {"model": model, "messages": messages, "stream": False}
    if tools:
        body["tools"] = tools
    if options:
        body["options"] = options
    if keep_alive:
        body["keep_alive"] = keep_alive
    payload = json.dumps(body).encode()
    req = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read())
    except urllib.error.URLError as e:
        raise OllamaError(
            f"Could not reach Ollama at {host} ({e}). Is `ollama serve` running?"
        ) from e
    return data["message"]


def chat_stream(
    messages: list[dict],
    model: str,
    host: str,
    tools: list[dict] = None,
    options: dict = None,
    timeout: int = 120,
    keep_alive: str = None,
):
    """Yield streaming chat chunks from Ollama's chat endpoint."""
    url = f"{host.rstrip('/')}/api/chat"
    body = {"model": model, "messages": messages, "stream": True}
    if tools:
        body["tools"] = tools
    if options:
        body["options"] = options
    if keep_alive:
        body["keep_alive"] = keep_alive
    payload = json.dumps(body).encode()
    req = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            for raw in resp:
                line = raw.strip()
                if not line:
                    continue
                try:
                    chunk = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if "error" in chunk:
                    raise OllamaError(chunk["error"])
                yield chunk
    except urllib.error.URLError as e:
        raise OllamaError(
            f"Could not reach Ollama at {host} ({e}). Is `ollama serve` running?"
        ) from e


def chat(
    messages: list[dict],
    model: str,
    host: str,
    options: dict = None,
    timeout: int = 120,
    keep_alive: str = None,
) -> str:
    return chat_message(
        messages, model, host, options=options, timeout=timeout, keep_alive=keep_alive
    )["content"]
