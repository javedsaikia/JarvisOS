"""Composio session for Max.

Official SDK pattern:

    from composio import Composio
    composio = Composio()
    session = composio.create(user_id=existing_user_id)
    tools = session.tools()

The key is max/.env. The user id is the one Composio already stored
connections under — discovered from connected accounts, not invented.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

from max import env, llm
from max.config import load_config, update_config

SESSION_PATH = Path(__file__).parent / ".composio_session.json"
DEFAULT_USER_ID = "max"

# Toolkits we will attach when they have an ACTIVE connected account.
KNOWN_TOOLKITS = ("gmail", "github", "youtube", "googledrive", "cal")


class ComposioError(Exception):
    pass


def configured() -> bool:
    return bool(env.load_env().get("COMPOSIO_API_KEY", "").strip())


def _api_key() -> str:
    key = env.load_env().get("COMPOSIO_API_KEY", "").strip()
    if not key:
        raise ComposioError("COMPOSIO_API_KEY not set in max/.env")
    return key


def _client():
    os.environ["COMPOSIO_API_KEY"] = _api_key()
    from composio import Composio

    return Composio(api_key=_api_key())


def list_accounts() -> list[dict]:
    """Every connected account on this project, no user-id filter."""
    client = _client()
    resp = client.connected_accounts.list(limit=50)
    items = []
    for a in resp.items or []:
        items.append(
            {
                "id": a.id,
                "toolkit": a.toolkit.slug,
                "status": a.status,
                "user_id": a.user_id,
            }
        )
    return items


def active_accounts() -> dict[str, str]:
    """toolkit slug -> connected account id, ACTIVE only."""
    out: dict[str, str] = {}
    for a in list_accounts():
        if a["status"] == "ACTIVE" and a["toolkit"] not in out:
            out[a["toolkit"]] = a["id"]
    return out


def user_id(cfg: dict | None = None) -> str:
    """Prefer the user id that already owns connections."""
    cfg = cfg if cfg is not None else load_config()
    configured_id = (cfg.get("composio_user_id") or "").strip()
    accounts = list_accounts()
    active_uids = [a["user_id"] for a in accounts if a["status"] == "ACTIVE" and a.get("user_id")]
    if active_uids:
        # If config still has the placeholder, lock onto the real one.
        if not configured_id or configured_id == DEFAULT_USER_ID:
            discovered = active_uids[0]
            update_config(composio_user_id=discovered)
            return discovered
        if configured_id in active_uids:
            return configured_id
        return active_uids[0]
    return configured_id or DEFAULT_USER_ID


def _load_session_record() -> dict:
    try:
        return json.loads(SESSION_PATH.read_text())
    except (OSError, json.JSONDecodeError):
        return {}


def _save_session_record(record: dict) -> None:
    try:
        SESSION_PATH.write_text(json.dumps(record))
    except OSError:
        pass


def get_session(cfg: dict | None = None):
    """Create or reuse a session bound to the authorized accounts."""
    uid = user_id(cfg)
    accounts = active_accounts()
    toolkits = [t for t in KNOWN_TOOLKITS if t in accounts]
    fingerprint = {"user_id": uid, "toolkits": toolkits, "accounts": accounts}

    composio = _client()
    existing = _load_session_record()
    if (
        existing.get("session_id")
        and existing.get("user_id") == uid
        and existing.get("accounts") == accounts
    ):
        try:
            return composio.use(existing["session_id"])
        except Exception:
            pass

    kwargs: dict = {
        "user_id": uid,
        "manage_connections": False,
        "sandbox": {"enable": False},
    }
    if toolkits:
        kwargs["toolkits"] = toolkits
        kwargs["connected_accounts"] = {t: accounts[t] for t in toolkits}
    session = composio.create(**kwargs)
    _save_session_record({**fingerprint, "session_id": session.session_id})
    return session


def tools(cfg: dict | None = None):
    return get_session(cfg).tools()


def _tools_as_dicts(raw) -> list[dict]:
    out = []
    for item in raw or []:
        if isinstance(item, dict):
            d = item
        elif hasattr(item, "model_dump"):
            d = item.model_dump()
        else:
            try:
                d = dict(item)
            except Exception:
                continue
        # Groq rejects function.strict: null (OpenAI providers emit it).
        fn = d.get("function")
        if isinstance(fn, dict) and fn.get("strict") is None:
            fn = dict(fn)
            fn.pop("strict", None)
            d = {**d, "function": fn}
        out.append(d)
    return out


def _result_text(result) -> str:
    if result is None:
        return ""
    if isinstance(result, str):
        return result
    if hasattr(result, "model_dump"):
        data = result.model_dump()
    elif hasattr(result, "data") or hasattr(result, "error"):
        data = {
            "data": getattr(result, "data", None),
            "error": getattr(result, "error", None),
            "successful": getattr(result, "successful", None),
        }
    else:
        data = result
    try:
        return json.dumps(data, default=str)[:8000]
    except TypeError:
        return str(data)[:8000]


READ_TOOLS = {
    "email": ("GMAIL_FETCH_EMAILS", {"max_results": 5}),
    "github": ("GITHUB_LIST_REPOSITORIES_FOR_THE_AUTHENTICATED_USER", {"per_page": 8, "sort": "updated"}),
    "youtube": ("YOUTUBE_LIST_CHANNELS", {}),
    "googledrive": ("GOOGLEDRIVE_LIST_FILES", {"pageSize": 8}),
    "cal": ("CAL_FETCH_ALL_BOOKINGS", {}),
}


def execute_and_say(user_text: str, cfg: dict, tool_slug: str, arguments: dict | None = None) -> str:
    """Run one known tool, then phrase the result. No giant tool schemas."""
    session = get_session(cfg)
    try:
        result = session.execute(tool_slug, arguments=arguments or {})
    except Exception as e:
        raise ComposioError(f"{tool_slug} failed: {e}") from e
    blob = _result_text(result)
    fallback_speech = _plain_summary(tool_slug, result)
    entry = llm.resolve(cfg, "chat")
    messages = [
        {
            "role": "system",
            "content": (
                "You are Max. Turn this tool result into one or two short spoken "
                "sentences. No markdown, no raw JSON. If it failed, say so."
            ),
        },
        {"role": "user", "content": f"Request: {user_text}\n\nResult:\n{blob[:6000]}"},
    ]
    try:
        return llm.chat(messages, entry, cfg, options={"num_predict": 100})
    except llm.LLMError:
        return fallback_speech


def _plain_summary(tool_slug: str, result) -> str:
    data = getattr(result, "data", None) or {}
    if not isinstance(data, dict):
        return _result_text(result)[:500]
    inner = data.get("data") if isinstance(data.get("data"), dict) else data
    if tool_slug.startswith("GMAIL_") and inner.get("messages"):
        lines = []
        for m in inner["messages"][:5]:
            sender = m.get("sender") or "unknown sender"
            subject = m.get("subject") or "(no subject)"
            lines.append(f"{sender}: {subject}")
        if not lines:
            return "The inbox is empty."
        return "Latest emails. " + " ".join(f"{i}. {line}." for i, line in enumerate(lines, 1))
    if "messages" not in inner:
        return f"{tool_slug} returned. " + _result_text(result)[:400]
    return _result_text(result)[:500]


def run(user_text: str, cfg: dict, domain: str = "email", action: str = "read") -> str:
    if action != "write":
        if domain not in READ_TOOLS:
            raise ComposioError(f"No Composio read path for {domain}.")
        slug, args = READ_TOOLS[domain]
        return execute_and_say(user_text, cfg, slug, args)

    if domain != "email":
        raise ComposioError("That write goes through a confirm + specific tool — say it more plainly.")

    entry = llm.resolve(cfg, "chat")
    extract = llm.chat(
        [
            {
                "role": "system",
                "content": (
                    "Extract a Gmail send. Reply with ONLY JSON: "
                    '{"to":["email"],"subject":"...","body":"..."}. '
                    "If the user did not name a recipient, set to to []."
                ),
            },
            {"role": "user", "content": user_text},
        ],
        entry,
        cfg,
    )
    try:
        start = extract.find("{")
        end = extract.rfind("}") + 1
        payload = json.loads(extract[start:end])
    except Exception as e:
        raise ComposioError(f"Could not parse that email: {e}") from e
    to = payload.get("to") or payload.get("recipient_email") or []
    if isinstance(to, str):
        to = [to]
    if not to:
        raise ComposioError("I need an email address to send to.")
    return execute_and_say(
        user_text,
        cfg,
        "GMAIL_SEND_EMAIL",
        {"to": to, "subject": payload.get("subject") or "", "body": payload.get("body") or ""},
    )


def complete(user_text: str, cfg: dict, *, max_steps: int = 8) -> str:
    """Run one request through the session tools and the assigned chat model."""
    session = get_session(cfg)
    wrapped = _tools_as_dicts(session.tools())
    if not wrapped:
        raise ComposioError("Composio session returned no tools.")

    connected = ", ".join(sorted(active_accounts()) or ["none"])
    system = (
        "You are Max. Use the Composio tools to complete the user's request. "
        f"Already connected: {connected}. Do not ask them to authorize those. "
        "If a tool errors, say so plainly. Ask before creating, updating, or deleting. "
        "Answer in one or two spoken sentences, no markdown. No lectures."
    )
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user_text},
    ]
    entry = llm.resolve(cfg, "chat")

    for _ in range(max_steps):
        try:
            turn = llm.chat_tools(messages, entry, cfg, tools=wrapped)
        except llm.LLMError as e:
            # Tool schemas are large; the 70B chat role regularly hits Groq's
            # TPM cap. The 8B voice model is the same provider and fits.
            if "429" in str(e) and entry.get("id") != "groq-8b":
                fallback = next(
                    (m for m in llm.catalog(cfg) if m["id"] == "groq-8b" and m["available"]),
                    None,
                )
                if fallback:
                    entry = fallback
                    turn = llm.chat_tools(messages, entry, cfg, tools=wrapped)
                else:
                    raise
            else:
                raise
        calls = turn.get("tool_calls") or []
        content = (turn.get("content") or "").strip()
        if not calls:
            if content:
                return content
            raise ComposioError("The model returned no answer and called no tool.")

        messages.append(
            {
                "role": "assistant",
                "content": content or None,
                "tool_calls": [
                    {
                        "id": c["id"],
                        "type": "function",
                        "function": {
                            "name": c["name"],
                            "arguments": json.dumps(c["arguments"]),
                        },
                    }
                    for c in calls
                ],
            }
        )
        for call in calls:
            try:
                result = session.execute(call["name"], arguments=call["arguments"] or {})
                payload = _result_text(result)
            except Exception as e:
                payload = json.dumps({"error": str(e), "successful": False})
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call["id"],
                    "content": payload,
                }
            )

    raise ComposioError("Composio tool loop ran too long without a final answer.")
