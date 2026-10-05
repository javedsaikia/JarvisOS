"""Confirmed native-app control via the macos-harness CLI.

The harness is a separate uv tool on Python 3.12. Bella's venv is 3.14, so
this module never imports macos_harness. It runs the `macos-harness`
binary with a fixed program; arguments travel in MAX_DESKTOP_ARGS as
JSON so model or user text never becomes Python source.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from pathlib import Path

TIMEOUT_SECONDS = 30
_HOME_BIN = Path.home() / ".local" / "bin" / "macos-harness"
_APP_RE = re.compile(r"^[A-Za-z0-9 .+\-]{1,64}$")
_KEY_RE = re.compile(
    r"^(?:(?:cmd|command|ctrl|control|alt|option|shift)\+)*"
    r"(?:[a-z0-9]|enter|return|tab|space|escape|esc|delete|backspace|"
    r"up|down|left|right|home|end|pageup|pagedown|f(?:[1-9]|1[0-2]))$",
    re.IGNORECASE,
)
_MAX_TARGET = 120
_MAX_TEXT = 2000

_PROG_FRONTMOST = """
import json
name = mac.script(
    'tell application "System Events" to get name of first '
    'application process whose frontmost is true'
)
print(json.dumps({"name": name}))
"""

_PROG_SEE = """
import json, os
args = json.loads(os.environ["MAX_DESKTOP_ARGS"])
frame = mac.see(args["app"])
app = frame.get("app") or {}
print(json.dumps({
    "app": app.get("name") or args["app"],
    "pid": frame.get("pid"),
    "window_id": frame.get("window_id"),
    "bounds": frame.get("bounds"),
    "focus": frame.get("focus"),
}))
"""

_PROG_FIND = """
import json, os
args = json.loads(os.environ["MAX_DESKTOP_ARGS"])
hits = mac.ax.query(
    app=args["app"],
    text=args["target"],
    limit=8,
    max_nodes=400,
    include_actions=True,
)
out = []
for hit in hits:
    out.append({
        "element_index": hit.get("element_index"),
        "role": hit.get("role"),
        "title": hit.get("title"),
        "description": hit.get("description"),
        "value": hit.get("value"),
        "actions": hit.get("actions") or [],
    })
print(json.dumps({"app": args["app"], "target": args["target"], "hits": out}))
"""

_PROG_CLICK = """
import json, os
args = json.loads(os.environ["MAX_DESKTOP_ARGS"])
app = args["app"]
target = args["target"]
needle = target.casefold()
mac.see(app)
hits = mac.ax.query(
    app=app,
    text=target,
    limit=12,
    max_nodes=400,
    include_actions=True,
)
roles = {
    "AXButton", "AXLink", "AXCheckBox", "AXMenuItem",
    "AXPopUpButton", "AXRadioButton", "AXTab", "AXImage",
}
pressable = []
for hit in hits:
    title = " ".join(
        str(hit.get(k) or "")
        for k in ("title", "description", "value")
    ).casefold()
    if needle not in title:
        continue
    actions = hit.get("actions") or []
    if hit.get("role") in roles or "AXPress" in actions:
        pressable.append(hit)
if not pressable:
    pressable = [
        hit for hit in hits
        if needle in " ".join(
            str(hit.get(k) or "")
            for k in ("title", "description", "value")
        ).casefold()
    ]
exact = [
    hit for hit in pressable
    if needle == str(hit.get("title") or "").casefold()
]
if len(exact) == 1:
    pressable = exact
if len(pressable) != 1:
    print(json.dumps({
        "ok": False,
        "error": "not_found" if not pressable else "ambiguous",
        "app": app,
        "target": target,
        "candidates": [
            {
                "role": hit.get("role"),
                "title": hit.get("title"),
                "description": hit.get("description"),
            }
            for hit in pressable[:6]
        ],
    }))
else:
    hit = pressable[0]
    mac.ax.perform(hit["element_index"], "AXPress")
    after = mac.see(app)
    print(json.dumps({
        "ok": True,
        "app": app,
        "target": target,
        "role": hit.get("role"),
        "title": hit.get("title"),
        "focus": after.get("focus"),
    }))
"""

_PROG_TYPE = """
import json, os
args = json.loads(os.environ["MAX_DESKTOP_ARGS"])
app = args["app"]
text = args["text"]
target = args.get("target") or ""
mac.see(app)
if target:
    needle = target.casefold()
    hits = mac.ax.query(
        app=app,
        text=target,
        limit=8,
        max_nodes=400,
        include_actions=True,
    )
    fields = []
    for hit in hits:
        blob = " ".join(
            str(hit.get(k) or "")
            for k in ("title", "description", "value", "placeholder")
        ).casefold()
        if needle in blob:
            fields.append(hit)
    if len(fields) != 1:
        print(json.dumps({
            "ok": False,
            "error": "not_found" if not fields else "ambiguous",
            "app": app,
            "target": target,
        }))
    else:
        hit = fields[0]
        if "AXPress" in (hit.get("actions") or []):
            mac.ax.perform(hit["element_index"], "AXPress")
        mac.type(text, app=app)
        print(json.dumps({"ok": True, "app": app, "target": target}))
else:
    mac.type(text, app=app)
    print(json.dumps({"ok": True, "app": app, "target": ""}))
"""

_PROG_KEY = """
import json, os
args = json.loads(os.environ["MAX_DESKTOP_ARGS"])
app = args["app"]
key = args["key"]
mac.see(app)
mac.key(key, app=app)
print(json.dumps({"ok": True, "app": app, "key": key}))
"""

_PROG_DOCTOR = """
import json
print(json.dumps(mac.doctor()))
"""


class DesktopError(Exception):
    pass


_binary_cache: str | None = None
_doctor_cache: dict | None = None


def harness_binary() -> str:
    """Path to the installed CLI. Not the Bella venv."""
    global _binary_cache
    if _binary_cache:
        return _binary_cache
    found = shutil.which("macos-harness")
    if found:
        _binary_cache = found
        return found
    if _HOME_BIN.is_file() and os.access(_HOME_BIN, os.X_OK):
        _binary_cache = str(_HOME_BIN)
        return _binary_cache
    raise DesktopError(
        "macos-harness is not on PATH. It stays a separate uv tool, "
        "not something installed into Bella's venv."
    )


def _run(program: str, args: dict) -> dict:
    env = os.environ.copy()
    env["MAX_DESKTOP_ARGS"] = json.dumps(args)
    try:
        result = subprocess.run(
            [harness_binary()],
            input=program,
            capture_output=True,
            text=True,
            timeout=TIMEOUT_SECONDS,
            env=env,
        )
    except subprocess.TimeoutExpired as e:
        raise DesktopError("That desktop action timed out.") from e
    except FileNotFoundError as e:
        raise DesktopError("macos-harness is not installed.") from e
    stdout = (result.stdout or "").strip()
    stderr = (result.stderr or "").strip()
    combined = f"{stdout}\n{stderr}"
    if "FocusChangedError" in combined:
        raise DesktopError(
            "The app came to the front, so I stopped rather than "
            "fighting you for focus."
        )
    if "No capturable windows" in combined:
        raise DesktopError("That app is running but has no window I can see.")
    if result.returncode != 0 and not stdout:
        raise DesktopError(stderr or "macos-harness failed.")
    payload = _last_json(stdout)
    if payload is None:
        raise DesktopError(stderr or "macos-harness returned nothing I could read.")
    return payload


def _last_json(text: str) -> dict | None:
    for line in reversed(text.splitlines()):
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            data = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(data, dict):
            return data
    return None


def _check_app(name: str) -> str:
    cleaned = (name or "").strip()
    if not cleaned or not _APP_RE.fullmatch(cleaned):
        raise DesktopError("That is not an app name I will send to the harness.")
    return cleaned


def _check_target(target: str) -> str:
    cleaned = " ".join((target or "").split())
    if not cleaned or len(cleaned) > _MAX_TARGET:
        raise DesktopError("I need a short name for the control.")
    return cleaned


def _check_text(text: str) -> str:
    if not isinstance(text, str) or not text:
        raise DesktopError("There is nothing to type.")
    if len(text) > _MAX_TEXT:
        raise DesktopError("That is too long to type in one go.")
    return text


def _check_key(key: str) -> str:
    cleaned = (key or "").strip().replace(" ", "+")
    if not _KEY_RE.fullmatch(cleaned):
        raise DesktopError(f'I will not press "{key}".')
    return cleaned.casefold()


def resolve_app(app: str | None) -> str:
    """Use the named app, or the one already frontmost. Never launch."""
    cleaned = (app or "").strip()
    if cleaned:
        return _check_app(cleaned)
    data = _run(_PROG_FRONTMOST, {})
    name = str(data.get("name") or "").strip()
    return _check_app(name)


def doctor() -> dict:
    global _doctor_cache
    if _doctor_cache is None:
        _doctor_cache = _run(_PROG_DOCTOR, {})
    return _doctor_cache


def see_app(app: str | None = None) -> str:
    name = resolve_app(app)
    data = _run(_PROG_SEE, {"app": name})
    shown = data.get("app") or name
    focus = (data.get("focus") or {}).get("frontmost") or {}
    front = focus.get("name") or "unknown"
    target_front = (data.get("focus") or {}).get("target_is_frontmost")
    where = "already in front" if target_front else f"{front} stayed in front"
    return f'{shown} is open ({where}).'


def find_control(app: str | None, target: str) -> str:
    name = resolve_app(app)
    needle = _check_target(target)
    data = _run(_PROG_FIND, {"app": name, "target": needle})
    hits = data.get("hits") or []
    if not hits:
        return f'Nothing labeled "{needle}" in {name}.'
    labels = []
    for hit in hits[:5]:
        label = hit.get("title") or hit.get("description") or hit.get("role") or "?"
        labels.append(str(label))
    return f'In {name} I can see: {", ".join(labels)}.'


def click(app: str | None, target: str) -> str:
    name = resolve_app(app)
    needle = _check_target(target)
    data = _run(_PROG_CLICK, {"app": name, "target": needle})
    if not data.get("ok"):
        return _miss(data, needle, name)
    label = data.get("title") or needle
    return f'Clicked "{label}" in {name}.'


def type_text(app: str | None, text: str, target: str = "") -> str:
    name = resolve_app(app)
    body = _check_text(text)
    field = _check_target(target) if target else ""
    data = _run(_PROG_TYPE, {"app": name, "text": body, "target": field})
    if not data.get("ok"):
        return _miss(data, field or "the field", name)
    if field:
        return f'Typed into "{field}" in {name}.'
    return f"Typed in {name}."


def press_key(app: str | None, key: str) -> str:
    name = resolve_app(app)
    combo = _check_key(key)
    _run(_PROG_KEY, {"app": name, "key": combo})
    return f"Pressed {combo} in {name}."


def _miss(data: dict, target: str, app: str) -> str:
    kind = data.get("error")
    if kind == "ambiguous":
        names = []
        for hit in data.get("candidates") or []:
            label = hit.get("title") or hit.get("description")
            if label:
                names.append(str(label))
        extra = f" ({', '.join(names)})" if names else ""
        return f'I can see more than one "{target}" in {app}{extra}. Which one?'
    return f'I could not find "{target}" in {app}.'
