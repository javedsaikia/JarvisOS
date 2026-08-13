"""Notes access via AppleScript (macOS Notes.app). Read + write, no deps."""
import html

from jarvis.applescript import as_string, run

# Marker appended by the script when it stopped because it hit the scan cap,
# not because it ran out of notes — lets the caller tell "genuinely no
# matches" apart from "stopped looking early" without a second round-trip.
_SCAN_CAPPED_MARKER = "[SCAN_LIMIT_REACHED]"

# search_notes' match counter only increments on a hit, so a query with few
# or no matches used to walk every note in every folder/account with no
# bound at all — fine for a small library, very slow for a large one (each
# check fetches a note's full plaintext body through the AppleScript
# bridge). `visited` counts every note examined, independent of whether it
# matched, and caps total scripting-bridge work regardless of hit rate.
_FOLDER_LOOP = """
set output to ""
set matches to 0
set visited to 0
set capped to false
tell application "Notes"
    repeat with acc in accounts
        repeat with fld in folders of acc
            if name of fld is not "Recently Deleted" then
                repeat with n in notes of fld
                    if visited ≥ {scan_limit} then
                        set capped to true
                        exit repeat
                    end if
                    if matches ≥ {result_limit} then exit repeat
                    set visited to visited + 1
                    {body}
                end repeat
            end if
        end repeat
    end repeat
end tell
if capped then
    set output to output & "{marker}"
end if
return output
"""


def _run_folder_loop(result_limit: int, scan_limit: int, body: str, timeout: int) -> tuple[str, bool]:
    script = _FOLDER_LOOP.format(
        result_limit=result_limit, scan_limit=scan_limit, body=body, marker=_SCAN_CAPPED_MARKER
    )
    result = run(script, timeout=timeout, app_name="Notes")
    capped = result.endswith(_SCAN_CAPPED_MARKER)
    if capped:
        result = result[: -len(_SCAN_CAPPED_MARKER)].rstrip("\n")
    return result, capped


def list_notes(limit: int = 10) -> str:
    result, _ = _run_folder_loop(
        result_limit=limit,
        scan_limit=limit,
        body='set output to output & (name of n) & return\n                    set matches to matches + 1',
        timeout=20,
    )
    return result if result else "No notes found."


def search_notes(query: str, limit: int = 10, scan_limit: int = 500) -> str:
    q = as_string(query)
    result, capped = _run_folder_loop(
        result_limit=limit,
        scan_limit=scan_limit,
        body=(
            f'if (name of n contains {q}) or (plaintext of n contains {q}) then\n'
            f'                        set output to output & (name of n) & return\n'
            f"                        set matches to matches + 1\n"
            f"                    end if"
        ),
        # Worst case here checks scan_limit notes' full plaintext bodies
        # through the scripting bridge — give it real headroom, same
        # reasoning as calendar.py's list_events timeout.
        timeout=45,
    )

    if not result:
        if capped:
            return (
                f'No notes found matching "{query}" in the first {scan_limit} notes checked '
                "(stopped early to avoid a long scan — try a more specific term)."
            )
        return f'No notes found matching "{query}".'

    if capped:
        return result + f"\n(stopped after checking {scan_limit} notes — there may be more matches; try a more specific term.)"
    return result


def create_note(title: str, body: str = "") -> str:
    # Notes.app derives the note's title from the first HTML paragraph of its
    # body — a literal newline in a plain string does NOT create a paragraph
    # break, so title and body would merge into one line without this.
    paragraphs = [title] + (body.split("\n") if body else [])
    html_body = "".join(f"<div>{html.escape(p)}</div>" for p in paragraphs)
    script = f"""
tell application "Notes"
    tell default account
        make new note with properties {{body:{as_string(html_body)}}}
    end tell
end tell
return "ok"
"""
    run(script, app_name="Notes")
    return f'Created note "{title}".'
