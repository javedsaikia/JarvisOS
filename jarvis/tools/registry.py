"""File/shell and browser tool schemas + the manual JSON tool-call
protocol.

Ollama's native `tools` field is passed through in case a future model's
template populates `message.tool_calls` properly, but qwen2.5-coder does not
(verified empirically) — it emits the call as a bare JSON object in
`content` instead. The system prompt below asks for exactly that format, and
`parse_tool_call` reads it back. This only ever runs for turns the router
has already flagged as file/shell-shaped or browser-shaped (see
router.detect_tool), so a normal chat reply never gets treated as a stray
tool call.

Two separate schema lists (FILE_TOOL_SCHEMAS, BROWSER_TOOL_SCHEMAS) rather
than one combined list — a file-domain turn should only ever see file
tools, not browser tools and vice versa, so the model can't call the wrong
domain's tool from the wrong handler (cli.execute_file_tool and
cli.execute_browser_tool each only know how to dispatch their own names).
build_tool_prompt()/parse_tool_call() both take the relevant schema list
as a parameter for exactly this reason.

Tool descriptions deliberately avoid mentioning "always confirmed before
running" (confirmation is enforced in code by cli.execute_*_tool's
confirm_fn regardless of what's in this text) — measured live, that phrase
in a description made the model pre-empt the app's own confirmation by
asking a natural-language "are you sure?" in `content` instead of ever
emitting the JSON call (click's tool-call rate: 1/6 with the phrase, 6/6
without it, same model/temperature/prompt otherwise).
"""
import json

FILE_TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": (
                "Read a text file's contents. Allowed anywhere under the JarvisOS project, "
                "Desktop, Documents, or Downloads — anywhere else is refused."
            ),
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string", "description": "Absolute path, or relative to the project root"}},
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Write (or overwrite) a text file within the JarvisOS project directory only.",
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
            "description": (
                "List files in a directory. Allowed anywhere under the JarvisOS project, "
                "Desktop, Documents, or Downloads — anywhere else is refused."
            ),
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string", "description": "Absolute path, or relative to the project root, default '.'"}},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_files",
            "description": (
                "Search by filename or content across the Desktop, Documents, Downloads, and "
                "the JarvisOS project directory. Use this to find a file before reading it "
                "when the exact path isn't already known."
            ),
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string", "description": "Filename or content to search for"}},
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_shell",
            "description": "Run a shell command in the JarvisOS project directory.",
            "parameters": {
                "type": "object",
                "properties": {"command": {"type": "string", "description": "The shell command to run"}},
                "required": ["command"],
            },
        },
    },
]

# Kept for compatibility with anything still expecting the pre-browser-tools
# name. New code should use FILE_TOOL_SCHEMAS directly.
TOOL_SCHEMAS = FILE_TOOL_SCHEMAS

BROWSER_TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "open_url",
            "description": "Navigate the browser to a URL.",
            "parameters": {
                "type": "object",
                "properties": {"url": {"type": "string", "description": "The URL to open"}},
                "required": ["url"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "click",
            "description": (
                "Click an element on the current page, described in plain language (e.g. "
                "\"the Submit button\", \"Sign In\")."
            ),
            "parameters": {
                "type": "object",
                "properties": {"target": {"type": "string", "description": "Plain-language description of what to click"}},
                "required": ["target"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "type_text",
            "description": (
                "Type text into a field on the current page, described in plain language (e.g. "
                "\"the email field\")."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "target": {"type": "string", "description": "Plain-language description of the field"},
                    "text": {"type": "string", "description": "The text to type into it"},
                },
                "required": ["target", "text"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_page_text",
            "description": "Read the visible text content of the current page. Read-only, never confirmed.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_current_url",
            "description": "Get the URL of the current page. Read-only, never confirmed.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
]


def tool_names(schemas: list[dict]) -> set[str]:
    return {t["function"]["name"] for t in schemas}


def build_tool_prompt(schemas: list[dict] = FILE_TOOL_SCHEMAS) -> str:
    lines = ["You have access to these tools for this request only:\n"]
    for t in schemas:
        fn = t["function"]
        props = fn["parameters"].get("properties", {})
        args = ", ".join(f'{name}: {spec.get("type", "any")}' for name, spec in props.items())
        lines.append(f"- {fn['name']}({args}) — {fn['description']}")
    lines.append(
        "\nThese tools are REAL and available to you in this turn, and they work. The "
        "general rule that you cannot act on this computer does NOT apply to them — it "
        "exists for requests that reach you with no tool attached. Do not refuse a "
        "request these tools can satisfy, and do not claim a path or folder is off "
        "limits: each tool enforces its own limits and will tell you if something is "
        "actually disallowed."
    )
    lines.append(
        "\nIMPORTANT: if the user's message matches one of these tools, you MUST call it — "
        "do not ask a clarifying follow-up question first, do not just chat about it. Respond "
        'with ONLY a single JSON object, nothing else, exactly like: {"name": "<tool_name>", '
        '"arguments": {...}}\n'
        "Only skip the tool call and answer normally in plain text if none of the tools above "
        "apply to the request at all."
    )
    return "\n".join(lines)


def parse_tool_call(content: str, valid_names: set[str] = None) -> dict | None:
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
    if valid_names is None:
        valid_names = tool_names(FILE_TOOL_SCHEMAS)
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
    if name not in valid_names:
        return None
    return {"name": name, "arguments": data.get("arguments", {}) or {}}
