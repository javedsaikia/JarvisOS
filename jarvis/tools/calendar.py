"""Calendar access via AppleScript (macOS Calendar.app). Read + write, no deps."""
from datetime import datetime, timedelta

from jarvis.applescript import as_string, run

# Subscribed/synced calendars can be dramatically slower than local ones for
# Calendar.app's AppleScript bridge to resolve actual event property values
# from — measured live: "India Holidays" (217 entries) alone accounted for
# ~9-10s of a ~13-16s query, even after eliminating the "whose" predicate
# below and combining property fetches into one call per calendar. Neither
# touched it — the slowness is inherent to how Calendar.app's scripting
# bridge handles that particular calendar, not the query strategy. Without
# it, the remaining calendars resolve in ~2-3s total. It's excluded from
# quick lookups by name rather than fixed some other way because there's no
# known AppleScript-level fix for it.
_EXCLUDED_CALENDAR_NAMES = ("India Holidays",)


def _as_string_list(names) -> str:
    return "{" + ", ".join(as_string(n) for n in names) + "}"


def _date_setters(var: str, dt: datetime) -> str:
    return "\n".join(
        [
            f"set {var} to current date",
            f"set year of {var} to {dt.year}",
            f"set month of {var} to {dt.month}",
            f"set day of {var} to {dt.day}",
            f"set hours of {var} to {dt.hour}",
            f"set minutes of {var} to {dt.minute}",
            f"set seconds of {var} to {dt.second}",
        ]
    )


def list_events(days: int = 1) -> str:
    """List events from the start of today through `days` day(s) out."""
    now = datetime.now()
    range_start = datetime(now.year, now.month, now.day)
    range_end = range_start + timedelta(days=days)

    script = f"""
{_date_setters("rangeStart", range_start)}
{_date_setters("rangeEnd", range_end)}
set output to ""
set excludedCalendars to {_as_string_list(_EXCLUDED_CALENDAR_NAMES)}
tell application "Calendar"
    repeat with cal in calendars
        set calName to name of cal
        if calName is not in excludedCalendars then
            -- "whose"/"where" predicate filters are catastrophically slow
            -- in Calendar.app's AppleScript implementation — it evaluates
            -- the predicate via a separate Apple Event round-trip per
            -- candidate event. Bulk-fetching each property as one parallel
            -- list per calendar is a small, fixed number of Apple Events
            -- regardless of how many events exist, and the actual date
            -- filtering then happens as local list comparisons with no
            -- further round-trips.
            set theseSummaries to summary of every event of cal
            set theseStarts to start date of every event of cal
            set theseEnds to end date of every event of cal
            repeat with i from 1 to (count of theseStarts)
                set evStart to item i of theseStarts
                if evStart ≥ rangeStart and evStart < rangeEnd then
                    set output to output & (item i of theseSummaries) & " — " & (evStart as string) & " to " & (item i of theseEnds as string) & " [" & calName & "]" & return
                end if
            end repeat
        end if
    end repeat
end tell
return output
"""
    result = run(script, timeout=60, app_name="Calendar")
    return result if result else "No events found in that range."


def add_event(title: str, start: datetime, end: datetime, calendar_name: str = None) -> str:
    cal_ref = f"calendar {as_string(calendar_name)}" if calendar_name else "calendar 1"
    script = f"""
{_date_setters("evStart", start)}
{_date_setters("evEnd", end)}
tell application "Calendar"
    tell {cal_ref}
        make new event with properties {{summary:{as_string(title)}, start date:evStart, end date:evEnd}}
    end tell
    reload calendars
end tell
return "ok"
"""
    run(script, app_name="Calendar")
    return f'Added "{title}" to your calendar for {start.strftime("%a %b %d, %I:%M %p")}.'
