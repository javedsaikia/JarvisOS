"""File/shell tool schemas + the manual JSON tool-call protocol.

Ollama's native `tools` field is passed through in case a future model's
template populates `message.tool_calls` properly, but qwen2.5-coder does not
(verified empirically) — it emits the call as a bare JSON object in
`content` instead. The system prompt below asks for exactly that format, and
`parse_tool_call` reads it back. This only ever runs for turns the router
has already flagged as file/shell-shaped (see router.detect_tool), so a
normal chat reply never gets treated as a stray tool call.
"""
import json

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a text file within the JarvisOS project directory.",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string", "description": "Path relative to the project root"}},
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Write (or overwrite) a text file within the JarvisOS project directory.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path relative to the project root"},
                    "content": {"type": "string", "description": "Full text content to write"},
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_dir",
            "description": "List files in a directory within the JarvisOS project directory.",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string", "description": "Directory path relative to project root, default '.'"}},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_shell",
            "description": "Run a shell command in the JarvisOS project directory. Always confirmed with the user before running.",
            "parameters": {
                "type": "object",
                "properties": {"command": {"type": "string", "description": "The shell command to run"}},
                "required": ["command"],
            },
        },
    },
]

TOOL_NAMES = {t["function"]["name"] for t in TOOL_SCHEMAS}


def build_tool_prompt() -> str:
    lines = ["You have access to these tools for this request only:\n"]
    for t in TOOL_SCHEMAS:
        fn = t["function"]
        props = fn["parameters"].get("properties", {})
        args = ", ".join(f'{name}: {spec.get("type", "any")}' for name, spec in props.items())
        lines.append(f"- {fn['name']}({args}) — {fn['description']}")
    lines.append(
        "\nIf a tool is needed, respond with ONLY a single JSON object, nothing else, "
        'exactly like: {"name": "<tool_name>", "arguments": {...}}\n'
        "If no tool is needed, just answer normally in plain text."
    )
    return "\n".join(lines)


def parse_tool_call(content: str) -> dict | None:
    """Find a tool-call JSON object anywhere in the model's reply.

    Some models don't follow "respond with ONLY JSON" — observed live:
    qwen2.5-coder prefacing the JSON with conversational text ("Sure thing,
    I'll run..."), sometimes also wrapped in a markdown code fence. A strict
    "must start with {" check misses these and silently falls through to
    treating the whole chatty reply as a plain answer — which means the
    tool never runs and, worse, never asks for confirmation either. Scanning
    for the first valid JSON object anywhere in the text (via raw_decode,
    which stops at the end of the JSON value and ignores anything after)
    handles leading/trailing text and fences in one pass.
    """
    idx = content.find("{")
    if idx == -1:
        return None
    try:
        data, _ = json.JSONDecoder().raw_decode(content[idx:])
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict):
        return None
    name = data.get("name")
    if name not in TOOL_NAMES:
        return None
    return {"name": name, "arguments": data.get("arguments", {}) or {}}
