"""Cheapest-capable-first routing. Pure rule-based (zero token cost) — no model call needed to decide.

Escalates to Claude Code only when the request clearly needs multi-file coding,
complex tool use, or system changes. Everything else stays on Ollama.
"""
import re

CLAUDE_CODE_PATTERNS = [
    r"\bbuild me\b",
    r"\bwrite (a |some )?(script|program|code|function|app|cli|tool)\b",
    r"\brefactor\b",
    r"\bdebug\b",
    r"\bfix (this|the) bug\b",
    r"\bimplement\b",
    r"\bcreate a (script|project|repo|app|file)\b",
    r"\bmulti-?file\b",
    r"\bgit (commit|push|pr|branch|merge)\b",
    r"\brun (the )?tests?\b",
    r"\binstall\b.*\bpackage\b",
    r"\bdeploy\b",
    r"\bclaude code\b",
]

_COMPILED = [re.compile(p, re.IGNORECASE) for p in CLAUDE_CODE_PATTERNS]

CALENDAR_WRITE_PATTERNS = [
    r"\badd\b.*\b(event|to (my )?calendar)\b",
    # "add a meeting"/"add an appointment" without the word "event" or
    # "calendar" — seen live: "Can you add a meeting at 1pm today with the
    # insurance agents?" fell through every pattern here, landed on plain
    # Ollama chat, and the model *hallucinated a confirmation* ("Sure,
    # I'll create a meeting...") without anything actually being added to
    # the calendar. Silent data loss dressed up as success — worse than a
    # miss the user would notice.
    r"\badd\b.*\b(meeting|appointment|call)\b",
    r"\bschedule\b.*\b(meeting|event|call|appointment)\b",
    r"\bcreate (an? )?event\b",
    r"\bbook\b.*\b(meeting|appointment)\b",
    r"\bput\b.*\bon (my )?calendar\b",
]

# Determiners are deliberately loose ("my"/"the"/none). Spoken phrasing
# varies far more than typed — observed live: "what's going on THE calendar"
# missed a "my"-only pattern, fell through to Ollama, and got answered by a
# model with no calendar access at all. A router miss here isn't a cheap
# failure: it silently hands a tool question to an LLM that can only guess.
CALENDAR_READ_PATTERNS = [
    r"\bwhat.*\bon (my |the )?(calendar|schedule)\b",
    r"\b(my|the) (calendar|schedule|agenda)\b",
    r"\bupcoming events?\b",
    r"\bevents? (today|tomorrow|this week|next week)\b",
    r"\bam i free\b",
    r"\bdo i have\b.*\b(meeting|event|call)s?\b",
    r"\bwhat('s| is)\b.*\b(scheduled|going on)\b",
]

NOTES_WRITE_PATTERNS = [
    r"\b(add|create|make|write|jot down|take) (a )?note\b",
]

NOTES_READ_PATTERNS = [
    r"\b(my|the) notes?\b",
    r"\bsearch (my |the )?notes?\b",
    r"\bfind\b.*\bnote\b",
    r"\bread\b.*\bnote\b",
    r"\blist\b.*\bnotes?\b",
]

# Escalates to Claude Code (see cli.handle_email), which already has a
# real, first-party Gmail connection — JARVIS's own backend isn't an MCP
# host and has no mailbox access of its own. Deliberately read-only for
# now, same scoping as calendar/notes reads; "send an email" is not
# handled here and would need its own write-confirm path if ever added.
# One vocabulary for "the mailbox", reused by every pattern below. "gmail"
# was missing from all of them, so the most natural way to ask — "check my
# Gmail" — matched nothing, fell through to the plain LLM, and got answered
# with "I can't access your email", which is what "JARVIS can't check my
# inbox" actually was. The Gmail connection itself was working the whole
# time (verified end to end against the real mailbox).
_MAILBOX = r"(?:e-?mails?|inbox|mail|gmail|g-mail)"

EMAIL_READ_PATTERNS = [
    rf"\bcheck (my |the )?{_MAILBOX}\b",
    rf"\bread (out |through )?(my |the )?{_MAILBOX}\b",
    rf"\bgo through (my |the )?{_MAILBOX}\b",
    rf"\bsummar(?:ize|ise) (my |the )?{_MAILBOX}\b",
    rf"\bopen (my |the )?{_MAILBOX}\b",
    rf"\bdo i have (any )?(new |unread )?{_MAILBOX}\b",
    rf"\bwhat('s| is| are)? ?(in )?(my |the )?{_MAILBOX}\b",
    rf"\bany (new |unread )?{_MAILBOX}\b",
    rf"\b(unread|new) {_MAILBOX}\b",
    rf"\b(any|new)\b.*\b(unread )?{_MAILBOX}\b",
]

# Read-only by design — see cli.handle_spotify. The connected Spotify MCP
# tools have no playback-control tool at all (no play/pause/skip); search,
# add_to_library, etc. are explicitly widget-only per their own tool
# descriptions and won't function from a headless Claude Code handoff. Only
# "what's playing" (get_currently_playing) is actually usable this way, so
# that's the only thing routed here.
SPOTIFY_READ_PATTERNS = [
    r"\bwhat('s| is) playing\b",
    r"\bwhat (song|track) is (this|playing)\b",
    r"\bwhat('s| is) (this )?(song|track)\b",
    r"\bcurrent(ly playing)? (song|track)\b",
    r"\bnow playing\b",
]

# Local AppleScript playback control (jarvis/tools/spotify.py) — distinct
# from SPOTIFY_READ_PATTERNS above, which asks Claude Code for account-wide
# "what's playing" status. These control THIS Mac's Spotify.app directly,
# no cloud round-trip, no Claude Code handoff.
SPOTIFY_PLAY_PATTERNS = [
    # "the song"/"the track" deliberately excluded here — that definite
    # article almost always precedes a specific title ("play the song
    # Bohemian Rhapsody"), which needs SPOTIFY_PLAY_TRACK_PATTERNS below,
    # not this generic/no-name path. Seen live: without this split, "play
    # the song Bohemian Rhapsody" matched here first and started the
    # default playlist instead of the actual requested song.
    r"\bplay (some |a )?(something|music|songs?|anything)\b",
    r"\bplay the (something|music|anything|playback)\b",
    r"\bplay spotify\b",
    r"\bresume (the )?(music|song|playback|spotify)\b",
    # "start the music" is ordinary phrasing that previously matched
    # nothing and fell through to the LLM. Safe alongside
    # SPOTIFY_OPEN_PATTERNS because that group is checked first, so
    # "start spotify" still means launch-without-playing.
    r"\bstart (the )?(music|song|playback)\b",
    r"\bunpause\b",
]

# Just launch the app — never start playback. Checked before the play
# group so "open spotify" can't be read as a request to resume. Requires
# the word "spotify" specifically, so "start the music" still plays.
SPOTIFY_OPEN_PATTERNS = [
    r"\b(open|launch|start|fire up)\s+(up\s+)?spotify\b",
]

SPOTIFY_PAUSE_PATTERNS = [
    r"\bpause (the )?(music|song|spotify|playback)\b",
    r"\bstop (the )?(music|song|spotify|playback)\b",
    # Explicit negative instruction ("do not play"/"don't play the song") —
    # requires a music-related noun so idioms ("don't play games with me",
    # "don't play it safe") don't misfire. Mapped to pause rather than a
    # no-op: if something's already playing this stops it; if nothing is,
    # pause is a harmless no-op — either way nothing starts playing.
    r"\b(do not|don't) play (the |any |that |this )?(music|song|spotify|track)\b",
    # "stop playing", "stop playing on/in Spotify", "stop playing the
    # music" — the patterns above need a music word directly after "stop",
    # so the very common "stop PLAYING ..." phrasing matched nothing.
    # Anchored to the end so it stays about playback and an unrelated
    # "stop playing games with me" doesn't pause the music.
    r"\bstop playing\b(\s+(the\s+)?(music|song|track|playback|spotify)|\s+(on|in)\s+spotify)?\s*[.!?]?\s*$",
]

# Mute (volume to 0, remembers the level for unmute) is distinct from
# pause above — the point is to let the mic hear a spoken command clearly
# without music underneath, not to stop playback. `.*` gaps (matching the
# same loose style as SPOTIFY_PLAY_TRACK_PATTERNS below) instead of a
# fixed word sequence, since natural phrasing varies a lot here — e.g.
# "the volume of Spotify should get muted" has words between "volume" and
# "muted" that a tight sequence wouldn't allow for.
SPOTIFY_MUTE_PATTERNS = [
    r"\bmute (the )?(spotify|music|song|volume)\b",
    r"\b(spotify|volume|music)\b.*\bmuted\b",
]

SPOTIFY_UNMUTE_PATTERNS = [
    r"\bunmute (the )?(spotify|music|song|volume)\b",
    r"\b(spotify|volume|music)\b.*\bunmuted\b",
]

SPOTIFY_VOLUME_DOWN_PATTERNS = [
    r"\bdecrease (the )?volume\b",
    r"\bturn down (the )?(volume|music|spotify|song)\b",
    r"\bvolume down\b",
    r"\blower (the )?volume\b",
    r"\b(spotify|music)\b.*\b(volume|it)\b.*\bdecreased\b",
]

SPOTIFY_VOLUME_UP_PATTERNS = [
    r"\bincrease (the )?volume\b",
    r"\bturn up (the )?(volume|music|spotify|song)\b",
    r"\bvolume up\b",
    r"\bmake it louder\b",
]

SPOTIFY_NEXT_PATTERNS = [
    r"\bnext (song|track)\b",
    r"\bskip (this |the )?(song|track)\b",
]

SPOTIFY_PREVIOUS_PATTERNS = [
    r"\bprevious (song|track)\b",
    r"\b(go back|play) (the )?(previous|last) (song|track)\b",
    r"\bgo back (a|one) (song|track)\b",
]

# Named-song/artist requests — checked AFTER SPOTIFY_PLAY_PATTERNS in
# _TOOL_GROUPS below, so generic phrasings ("play something on spotify")
# still resolve to the cheap generic action first. Requires an explicit
# "on/in spotify" or "the song/track" cue rather than a bare "play X" —
# a bare catch-all would misfire on unrelated things like "play devil's
# advocate". See jarvis.tools.parsing.extract_song_query for the text
# actually sent to Spotify's search API.
SPOTIFY_PLAY_TRACK_PATTERNS = [
    r"\bplay\b.+\bon spotify\b",
    r"\bplay\b.+\bin spotify\b",
    r"\bplay the (song|track)\b.+",
    r"\bput on\b.+\bspotify\b",
]

# Deterministic, not LLM-mediated — read-only informational, same tier as
# calendar/notes/email reads.
WEATHER_PATTERNS = [
    r"\bwhat('s| is) the weather\b",
    r"\bhow('s| is) the weather\b",
    r"\bweather (today|now|outside|like)\b",
    r"\bis it (going to |gonna )?rain(ing)?\b",
    r"\btemperature outside\b",
]

# Category (restaurant/hotel/gas station/...) is extracted by
# jarvis.tools.location.detect_category — a plain keyword lookup over a
# small fixed vocabulary, not LLM extraction. Deliberately requires both
# a place-type word AND a proximity word ("nearby"/"near me"/...) so a
# plain "restaurant" mention in unrelated conversation doesn't misfire.
NEARBY_PATTERNS = [
    r"\b(nearby|near me|around here|close by)\b.*\b(restaurant|hotel|gas station|petrol|fuel|cafe|coffee|pharmacy|food)s?\b",
    r"\b(restaurant|hotel|gas station|petrol|fuel|cafe|coffee|pharmacy|food)s?\b.*\b(nearby|near me|around here|close by)\b",
    r"\bfind (a |the )?(nearby |closest |nearest )?(restaurant|hotel|gas station|cafe|pharmacy)\b",
    r"\bwhere('s| is) the (nearest|closest) (restaurant|hotel|gas station|cafe|pharmacy)\b",
]

# Confirmed — opens a real app with a real URL. See MAPS_PATTERNS' domain
# entry in _TOOL_GROUPS: ("location", "open_maps"), gated in
# cli.handle_open_maps.
MAPS_PATTERNS = [
    r"\bopen (that |this |it )?(in|on) (apple )?maps\b",
    r"\bshow (that |this |it )?on (the |a )?map\b",
    r"\bopen maps\b",
]

# Confirmed, deterministic — NOT LLM-mediated, same reasoning as
# SHELL_PATTERNS above: calling a real phone number is a real-world
# action, so whether it happens can't depend on a model reliably emitting
# a tool-call. jarvis.tools.calling.parse_phone_number does the actual
# number extraction; this just detects the calling intent.
CALL_PATTERNS = [
    r"\bcall (this |that )?number\b",
    r"\bcall\b.*\d{3}",
    r"\bdial\b.*\d{3}",
    r"\bdial (this |that )?number\b",
    # Calling a place by name ("call Hotel Heritage") or referring back to
    # one ("call them"). Without these, only spoken digits reached
    # cli.handle_call and everything else fell through to the plain LLM —
    # which answered "I'll place a call" and then "I'm en route", having
    # done neither. The number is looked up and shown before anything is
    # dialled; see handle_call.
    #
    # The negative lookahead is what keeps this safe. English is full of
    # "call" idioms that have nothing to do with telephones, and dialling
    # a stranger because someone said "call it a day" is not a recoverable
    # mistake.
    # Bare it/this/that are excluded here rather than allowed: they are far
    # more often idiom ("what do you call this", "call it a day") than a
    # real request, and the genuine forms are already covered — "call this
    # number" by the first pattern above and "call that place" by the
    # referential one below.
    r"\b(?:call|ring|phone)\s+(?:up\s+)?(?:the\s+)?"
    r"(?!it\b|this\b|that\b|me\b|you\b|us\b|back\b|off\b|out\b|in\b|for\b|on\b|upon\b|"
    r"an? (?:meeting|ambulance|taxi|cab|uber)\b)"
    r"[A-Za-z][\w'&.\-]*(?:\s+[\w'&.\-]+){0,4}\s*$",
    r"\bcall (them|him|her|that place|that one|that restaurant|that hotel)\b",
]

# Deliberately narrow — "write a script to rename files" must still escalate
# to Claude Code (matches CLAUDE_CODE_PATTERNS' "write a script"), not get
# swallowed here. These only match direct, simple file operations that still
# need LLM extraction (a path/content isn't literally present in the text
# the way a shell command is) — see SHELL_PATTERNS below for the
# deterministic, non-LLM-mediated shell path.
# Deterministic, not LLM-mediated — there's nothing to extract from the
# phrasing (unlike browser targets/URLs below), so this is the same
# "cheap and direct" treatment as calendar/notes/email reads.
SCREEN_PATTERNS = [
    r"\bwhat('s| is) on (my |the )?screen\b",
    r"\bwhat('s| is) on (this|the) page\b",
    r"\bdescribe what i('m| am) (looking at|seeing)\b",
    r"\bwhat do you see\b",
    r"\btake a screenshot\b",
]

# Browser actions are LLM-mediated (like FILES_PATTERNS below) — click
# targets, typed text, and URLs are too free-form for deterministic
# extraction. Every action these route to that changes state
# (open_url/click/type_text) is confirmed in cli.execute_browser_tool
# before it runs; only get_page_text/get_current_url are automatic.
BROWSER_PATTERNS = [
    r"\bclick (the |on )?.+\b(button|link)\b",
    r"\bfill (in |out )?(the )?form\b",
    r"\bnavigate to\b",
    r"\bopen (this |the )?(url|website|webpage|link)\b",
    r"\btype\b.+\b(into|in) the\b",
    r"\bgo to\b.+\b(and|then)\b",
    # Bare "go to <url>" (no compound "and do X" clause) — seen live:
    # without this, "go to example.com" fell through to plain Ollama,
    # which HALLUCINATED a fake "you're now on the page" confirmation
    # without ever actually navigating. Requires something URL-shaped
    # after "go to" (a dot-TLD or a scheme) so plain "go to sleep"/"go to
    # the store" don't misfire.
    r"\bgo to\b.+(\.(com|org|net|io|co|dev|app|in)\b|https?://)",
    # Same gap, different verb — seen live: "open wikipedia.org in the
    # browser" fell through to plain Ollama (which correctly declined
    # rather than hallucinating, but still never navigated) because the
    # existing "open ... url/website/webpage/link" pattern only matches
    # those literal nouns, not a bare domain. Same URL-shape requirement
    # as the "go to" pattern above so "open the calendar"/"open spotify"
    # don't misfire.
    r"\bopen\b.+(\.(com|org|net|io|co|dev|app|in)\b|https?://)",
]

FILES_PATTERNS = [
    r"\bread (the |a )?file\b",
    r"\bopen (the |a )?file\b",
    r"\bshow me (the )?(contents of )?(the |a )?file\b",
    r"\blist (the )?files? in\b",
    r"\bwhat('s| is) in\b.*\b(folder|directory)\b",
    r"\bwrite\b.*\b(to|into) (a |the )?file\b",
    r"\bcreate (a |an )?file\b",
    r"\bsave\b.*\b(to|into|as) (a |the )?file\b",
    r"\bproject (files|directory|folder)\b",
    r"\bsearch (for |my )?(a |the )?files?\b",
    r"\bfind (a |the |my )?files?\b",
    r"\bsearch (my |the )?(desktop|documents|downloads)\b",
]

# Shell commands get their own domain, deliberately NOT LLM-mediated: "run
# X" is a literal instruction, so the command is regex-extracted
# (parsing.extract_shell_command) and confirmed/executed directly. Routing
# this through Ollama's tool-calling (like the "files" domain) was tried
# first and found unreliable — the model doesn't always emit the tool-call
# JSON even when told to, and when it doesn't, the fabricated text response
# was returned as the answer with no confirmation ever asked and no command
# ever actually run. Since "always confirm before executing" is a hard
# requirement, the decision to run can't depend on model compliance.
#
# "run"/"execute" is deliberately treated as an explicit, literal shell
# invocation that always wins over CLAUDE_CODE_PATTERNS (e.g. "run git
# status" goes to the confirmed local shell tool, not Claude Code, even
# though "git" also appears in CLAUDE_CODE_PATTERNS) — this is checked
# first in detect_tool. The distinction is: "run <command>" names a literal
# command to execute; "commit this" / "install the package" describe an
# outcome and still need Claude Code to figure out specifics.
_SHELL_VERBS = (
    r"ls|cd|cat|echo|pwd|mkdir|rmdir|rm|cp|mv|touch|grep|find|head|tail|wc|df|du|ps|"
    r"whoami|date|open|say|which|chmod|chown|kill|top|env|uname|git|npm|npx|pip3?|"
    r"python3?|node|brew|make|yarn|curl|wget|tar|zip|unzip|sed|awk|osascript|bash|sh|zsh"
)

SHELL_PATTERNS = [
    r"\brun\b.*\b(shell )?command\b",
    r"\bexecute\b.*\bcommand\b",
    r"\bshell command\b",
    # bare "run/execute <literal command>", e.g. "Run ls -la", "execute git status"
    rf"\b(run|execute)\b\s+(?:this\s+|the\s+|following\s+|exact\s+)*({_SHELL_VERBS})\b",
    # generic: "run/execute" plus shell syntax (a flag, pipe, redirect, or chain)
    r"\b(run|execute)\b.*(\s-{1,2}\w|\||&&|;|>)",
]

_TOOL_GROUPS = [
    ("calendar", "write", [re.compile(p, re.IGNORECASE) for p in CALENDAR_WRITE_PATTERNS]),
    ("calendar", "read", [re.compile(p, re.IGNORECASE) for p in CALENDAR_READ_PATTERNS]),
    ("notes", "write", [re.compile(p, re.IGNORECASE) for p in NOTES_WRITE_PATTERNS]),
    ("notes", "read", [re.compile(p, re.IGNORECASE) for p in NOTES_READ_PATTERNS]),
    ("email", "read", [re.compile(p, re.IGNORECASE) for p in EMAIL_READ_PATTERNS]),
    ("spotify", "read", [re.compile(p, re.IGNORECASE) for p in SPOTIFY_READ_PATTERNS]),
    # pause checked BEFORE play — seen live: "don't play the music" contains
    # the substring "play the music", which SPOTIFY_PLAY_PATTERNS matches
    # regardless of the "don't" in front (patterns aren't negation-aware).
    # Checking the narrower, negation-including pause group first means a
    # negated play phrase resolves correctly; no genuine "play X" request
    # matches any pause pattern, so this reordering doesn't change any
    # other outcome.
    ("spotify", "open", [re.compile(p, re.IGNORECASE) for p in SPOTIFY_OPEN_PATTERNS]),
    ("spotify", "pause", [re.compile(p, re.IGNORECASE) for p in SPOTIFY_PAUSE_PATTERNS]),
    ("spotify", "play", [re.compile(p, re.IGNORECASE) for p in SPOTIFY_PLAY_PATTERNS]),
    ("spotify", "mute", [re.compile(p, re.IGNORECASE) for p in SPOTIFY_MUTE_PATTERNS]),
    ("spotify", "unmute", [re.compile(p, re.IGNORECASE) for p in SPOTIFY_UNMUTE_PATTERNS]),
    ("spotify", "volume_down", [re.compile(p, re.IGNORECASE) for p in SPOTIFY_VOLUME_DOWN_PATTERNS]),
    ("spotify", "volume_up", [re.compile(p, re.IGNORECASE) for p in SPOTIFY_VOLUME_UP_PATTERNS]),
    ("spotify", "next", [re.compile(p, re.IGNORECASE) for p in SPOTIFY_NEXT_PATTERNS]),
    ("spotify", "previous", [re.compile(p, re.IGNORECASE) for p in SPOTIFY_PREVIOUS_PATTERNS]),
    ("spotify", "play_track", [re.compile(p, re.IGNORECASE) for p in SPOTIFY_PLAY_TRACK_PATTERNS]),
    ("location", "weather", [re.compile(p, re.IGNORECASE) for p in WEATHER_PATTERNS]),
    ("location", "nearby", [re.compile(p, re.IGNORECASE) for p in NEARBY_PATTERNS]),
    ("location", "open_maps", [re.compile(p, re.IGNORECASE) for p in MAPS_PATTERNS]),
    ("call", "auto", [re.compile(p, re.IGNORECASE) for p in CALL_PATTERNS]),
    # Checked before "files" — a request naming a literal command (e.g. "run
    # ls -la") must never fall through to the LLM-mediated files path.
    ("shell", "auto", [re.compile(p, re.IGNORECASE) for p in SHELL_PATTERNS]),
    ("screen", "read", [re.compile(p, re.IGNORECASE) for p in SCREEN_PATTERNS]),
    # Checked before "files" — e.g. "open this website" must not fall
    # through to the files domain's "open (the|a) file" pattern.
    ("browser", "auto", [re.compile(p, re.IGNORECASE) for p in BROWSER_PATTERNS]),
    ("files", "auto", [re.compile(p, re.IGNORECASE) for p in FILES_PATTERNS]),
]


def detect_tool(text: str) -> tuple[str, str] | None:
    """Return (domain, action) e.g. ('calendar', 'read'), or None if this
    isn't a Calendar/Notes/Files/Shell-shaped request. Checked before
    decide_backend — tool calls are cheaper than both Ollama and Claude Code.
    'files' action is always 'auto': the LLM decides which specific file
    tool to call, since paths/content are too free-form for regex
    extraction. 'shell' is also 'auto' but is NOT LLM-mediated — see
    SHELL_PATTERNS above for why.
    """
    stripped = text.strip()
    for domain, action, patterns in _TOOL_GROUPS:
        for pattern in patterns:
            if pattern.search(stripped):
                return domain, action
    return None


def decide_backend(text: str, default_backend: str = "ollama") -> str:
    """Return 'claude_code' or 'ollama'."""
    stripped = text.strip()

    if stripped.startswith("/claude"):
        return "claude_code"
    if stripped.startswith("/ollama"):
        return "ollama"

    for pattern in _COMPILED:
        if pattern.search(stripped):
            return "claude_code"

    return default_backend
