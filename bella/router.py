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

# Gmail via Composio (cli.handle_email). Writes confirm first.
_MAILBOX = r"(?:e-?mails?|inbox|mail|gmail|g-mail)"

EMAIL_WRITE_PATTERNS = [
    r"\bsend (an? |this )?(e-?mail|gmail)\b",
    r"\bcompose (an? )?(e-?mail|gmail)\b",
    r"\bdraft (an? |this )?(e-?mail|gmail|mail)\b",
    r"\bwrite (an? |this )?(e-?mail|gmail)\b",
    r"\breply to (that |this |the )?(e-?mail|gmail|message)\b",
    # "reply to Zapier and say thank you" — name, not "the email".
    r"\breply to \S+",
    r"\b(e-?mail|gmail)\b.{0,40}\b(say|saying)\b",
    rf"\b{_MAILBOX}\b.+\b(to|saying|about)\b",
]

EMAIL_READ_PATTERNS = [
    rf"\bcheck (my |the )?{_MAILBOX}\b",
    # "my/the" required. Seen live: "D-Read Gmail.com" matched
    # "read … gmail" and fetched the inbox instead of taking a recipient.
    rf"\bread (out |through )?(my |the |your ){_MAILBOX}\b",
    rf"\bgo through (my |the )?{_MAILBOX}\b",
    rf"\bsummar(?:ize|ise) (my |the )?{_MAILBOX}\b",
    rf"\bopen (my |the )?{_MAILBOX}\b",
    rf"\bdo i have (any )?(new |unread )?{_MAILBOX}\b",
    rf"\bwhat('s| is| are)? ?(in )?(my |the )?{_MAILBOX}\b",
    rf"\bany (new |unread )?{_MAILBOX}\b",
    rf"\b(unread|new) {_MAILBOX}\b",
    rf"\b(any|new)\b.*\b(unread )?{_MAILBOX}\b",
    r"ই(?:-)?মেইল",
    r"ইনবক্স",
]

# Newly connected Composio toolkits. Each requires the product name, for
# the same reason GitHub/YouTube/Drive do: "my calendar" is the local
# Apple Calendar (instant, offline, no round trip), and only "google
# calendar" should pay for the network.
GOOGLE_CALENDAR_PATTERNS = [
    r"\bgoogle cal(?:endar)?\b",
    r"\bg[- ]cal\b",
]

GOOGLE_TASKS_PATTERNS = [
    r"\bgoogle tasks?\b",
    r"\b(?:my )?task list\b",
]

LINKEDIN_PATTERNS = [
    r"\blinked ?in\b",
]

REDDIT_PATTERNS = [
    r"\breddit\b",
    r"\br/\w+",
    r"\bsubreddit\b",
]

# Connected Composio toolkits. Require the product name so they cannot
# steal local calendar, local files, or a literal `git` shell command.
GITHUB_PATTERNS = [
    r"\bgithub\b",
]

YOUTUBE_PATTERNS = [
    r"\byoutube\b",
]

DRIVE_PATTERNS = [
    r"\bgoogle drive\b",
    r"\bgoogledrive\b",
    r"\b(my |the )?drive (file|folder|doc|document)s?\b",
]

# Cal.com — not Apple Calendar. Those stay on the calendar patterns above.
CAL_COM_PATTERNS = [
    r"\bcal\.com\b",
    r"\bcal com\b",
    r"\bcal booking\b",
    r"\bcal\.com (booking|event|link|schedule)\b",
]

# ChatFAI — only when they name the product, so movie/OMDb questions stay put.
CHATFAI_PATTERNS = [
    r"\bchatfai\b",
    r"\bchat fai\b",
]

# Local AppleScript status on this Mac — not a cloud account lookup.
SPOTIFY_READ_PATTERNS = [
    r"\bwhat('s| is) playing\b",
    r"\bwhat (song|track) is (this|playing)\b",
    r"\bwhat('s| is) (this )?(song|track)\b",
    r"\bcurrent(ly playing)? (song|track)\b",
    r"\bnow playing\b",
]

# Local AppleScript playback control (bella/tools/spotify.py).
SPOTIFY_PLAY_PATTERNS = [
    # "the song"/"the track" deliberately excluded here — that definite
    # article almost always precedes a specific title ("play the song
    # Bohemian Rhapsody"), which needs SPOTIFY_PLAY_TRACK_PATTERNS below,
    # not this generic/no-name path. Seen live: without this split, "play
    # the song Bohemian Rhapsody" matched here first and started the
    # default playlist instead of the actual requested song.
    # "play a song" / "play some music" with NOTHING named after it.
    # "play a song Nothing Else Matters by Metallica" used to match here
    # because of `songs?`, then play() resumed whatever was paused
    # (High Hopes). A title after "song" must go to play_track.
    r"\bplay (some |a )?(something|music|anything)\b",
    r"\bplay (some |a )?songs?\s*$",
    r"\bplay (some |a )?songs?\s+(on|in)\s+spotify\s*$",
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

# Bella's spoken voice — not Spotify. Generic "volume down" used to hit
# Spotify and leave the reply still piercing on earphones. Music-specific
# phrasing stays on the Spotify groups below.
VOICE_VOLUME_DOWN_PATTERNS = [
    r"\b(too loud|so loud|way too loud|piercing)\b",
    r"\bhurts my (ear|ears)\b",
    r"\b(speak|talk)\b.*\b(quieter|softer|lower)\b",
    r"\b(quieter|softer)\b",
    r"\bturn (your )?(voice|volume) down\b",
    r"\b(your|bella['’]?s?)\b.{0,20}\bvolume\b.{0,20}\b(down|lower|decrease)\b",
    r"\b(decrease|lower)\b.{0,20}\b(your |the )?(voice |speech )?volume\b",
    r"\bdecrease (the )?volume\b",
    r"\bturn down (the )?volume\b",
    r"\bvolume down\b",
    r"\blower (the )?volume\b",
]

VOICE_VOLUME_UP_PATTERNS = [
    r"\b(speak|talk)\b.*\b(louder|up)\b",
    r"\b(louder)\b",
    r"\bturn (your )?(voice|volume) up\b",
    r"\b(your|bella['’]?s?)\b.{0,20}\bvolume\b.{0,20}\b(up|higher|increase)\b",
    r"\bincrease (the )?volume\b",
    r"\bturn up (the )?volume\b",
    r"\bvolume up\b",
    r"\bmake it louder\b",
]

SPOTIFY_VOLUME_DOWN_PATTERNS = [
    r"\bturn down (the )?(music|spotify|song)\b",
    r"\b(spotify|music)\b.*\b(volume|it)\b.*\b(decreased|down|lower)\b",
    r"\b(decrease|lower) (the )?(spotify|music) volume\b",
]

SPOTIFY_VOLUME_UP_PATTERNS = [
    r"\bturn up (the )?(music|spotify|song)\b",
    r"\b(spotify|music)\b.*\b(volume|it)\b.*\b(increased|up|higher|louder)\b",
    r"\b(increase|raise) (the )?(spotify|music) volume\b",
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
# advocate". See bella.tools.parsing.extract_song_query for the text
# actually sent to Spotify's search API.
SPOTIFY_PLAY_TRACK_PATTERNS = [
    r"\bplay\b.+\bon spotify\b",
    r"\bplay\b.+\bin spotify\b",
    r"\bplay the (song|track)\b.+",
    r"\bput on\b.+\bspotify\b",
    # Follow-ups that name no title — "play it", "go ahead and play that".
    # Must route here (not to chat) so play_track can reuse the last query.
    # The pronouns are excluded from the bare-title pattern below so they
    # are never searched as a song name. Seen live: "Go ahead and play it"
    # searched "it" and started a Hindi film song.
    r"\bplay (it|that|this|the song|the track)\b",
    r"\bgo ahead and play\b",
    r"\bjust play (it|that|this)?\b",
    # Bare "play <something>" with no "on Spotify" — said live as "Play
    # Pink Floyd High Hopes." It matched nothing, fell through to the
    # model, and the model replied "Okay, I'm playing High Hopes by Pink
    # Floyd on Spotify now" while nothing played. A hallucinated action is
    # the worst possible outcome here, so the bare form has to route.
    # SPOTIFY_PLAY_PATTERNS (generic "play some music") is checked first,
    # so this only sees requests that name something.
    # it/that/this are excluded — those are follow-ups, handled above.
    r"\bplay\b\s+(?!some\b|the\b|any\b|music\b|something\b|anything\b|it\b|that\b|this\b)\S+",
    # STT of "play" — seen live: "Bile, nothing else matters by Metallica".
    r"\b(bile|ply)\b[,\s]+\S+",
    # Last resort: the word "Spotify" appearing at all. Every more
    # specific Spotify action — what's playing, open, pause, mute, volume,
    # next, previous — is checked BEFORE this group, so this only catches
    # what none of them wanted.
    #
    # It exists because speech-to-text mangles song titles far more than
    # it mangles the word "Spotify". Measured live: "<title> by <band> on
    # Spotify" came through as "by being deployed on Spotify" — no "play"
    # anywhere in it, so nothing matched, the turn fell through to the
    # chat model, and the model announced it was playing the song while
    # nothing happened. Saying Spotify means Spotify.
    r"\bspotify\b",
]

# Deterministic, not LLM-mediated — read-only informational, same tier as
# calendar/notes/email reads.
# Real headlines via RSS (cli.handle_news). Must match before chat so the
# model cannot invent "today's news" from training data — seen live on
# weather and politics questions. Requires a news word, not a bare
# "what's happening at work".
# Movie/series facts via OMDb. After nearby so "movie theater" stays maps.
OMDB_PATTERNS = [
    r"\b(movie|film|imdb)\b",
    r"\b(tv series|tv show|web series)\b",
    r"\bwho directed\b",
    r"\bwho (stars|starred|acts|acted) in\b",
    r"\bplot of\b",
    r"\bimdb (rating|score)\b",
    r"\b(rating|ratings) (for|of)\b.+\b(movie|film|show|series)\b",
    r"\bwhen (was|did)\b.+\b(released|come out)\b",
]

NEWS_PATTERNS = [
    r"\b(latest |today'?s |the |daily )?(ai |tech(?:nology)? |politic(?:al|s)? |finance |financial |business |world |global )?news\b",
    # Bare "headlines" stole an inbox summary — "summarize it, only the
    # headlines" after email. Need a news word, not just the noun.
    r"\b(news|today'?s|daily) headlines\b",
    r"\bwhat'?s happening (today|in the world|globally|across the globe|in (ai|tech|politics|finance))\b",
    r"\bnews (today|briefing|update|roundup)\b",
]

# Hotel / flight listing search. Read-only: names, prices, times, links.
# Nearby "hotels near me" stays on Maps (NEARBY_PATTERNS). These need a
# city, a route, or an "options/prices" word so GPS nearby is not stolen.
DISCOVERY_HOTEL_PATTERNS = [
    r"\bhotel (options?|listings?|deals?|prices?|rates?)\b",
    r"\b(hotels?|rooms?|places to stay|stays?)\b.{0,40}\bin\b",
    r"\b(book|find|search|show) (a |an |me )?(hotel|stay|room)\b.{0,32}\bin\b",
    r"\bwhere (can i|to) stay\b",
    r"\bstay in\b",
]
DISCOVERY_FLIGHT_PATTERNS = [
    r"\bflights?\b.{0,48}\b(from|to|between)\b",
    r"\b(plane |air )?tickets?\b.{0,48}\b(from|to)\b",
    r"\bairfare\b",
    r"\bfly from\b",
    r"\b(search|find|show) (me )?(a |the )?flights?\b",
]

# Live board at a cinema — not OMDb (that's plot/cast) and not Spotify
# ("what's playing"). Maps names the hall; BookMyShow has the shows.
SHOWTIME_PATTERNS = [
    r"\b(now showing|now-showing|showtimes?|show times?)\b",
    # "screen" deliberately NOT in this list. A cinema screen and the one
    # on this Mac are the same word, and showtimes is checked before the
    # screen group — so this pattern was claiming "what's on my screen",
    # the single most documented phrase in the whole tool, and answering it
    # with cinema listings. "the big screen" keeps the film sense.
    r"\bwhat('s| is| movies? are) (on|playing|running|showing)\b.*\b(theater|theatre|cinema|big screen)\b",
    r"\b(movies?|films?|shows?)\b.*\b(playing|running|showing)\b.*\b(theater|theatre|cinema|now|today)\b",
    r"\b(playing|running|showing)\b.*\b(theater|theatre|cinema)\b",
    r"\bwhich (movies?|films?|shows?)\b.*\b(theater|theatre|cinema|screen)\b",
    r"\b(movies?|films?)\b.*\b(in (the |this )?(theater|theatre|cinema)|on (at )?(the )?screen)\b",
    r"ছবি.{0,20}(?:চলিছে|চলি আছে|শ্ব)",
    r"(?:চিনেমা|থিয়েটাৰ|theater|theatre).{0,24}(?:ছবি|শ্ব|চলিছে)",
    r"শ্ব['’]?\s*টাইম",
]

WEATHER_PATTERNS = [
    r"\bwhat('s| is) the weather\b",
    r"\bhow('s| is) the weather\b",
    r"\bhow'?s weather\b",
    r"\bweather (today|now|outside|like|here|forecast|report|in|for|at)\b",
    r"\b(current |today'?s |local |my )?weather\b",
    r"\b(check|tell me|give me|get) (the |today'?s )?weather\b",
    r"\b(weather )?forecast\b",
    r"\bforecast (for|in|at)\b",
    r"\bis it (going to |gonna )?rain(ing)?\b",
    r"\bwill it rain\b",
    r"\btemperature (outside|here|today|in|for)\b",
    r"\bhow (hot|cold|warm) is it\b",
    r"\bhow'?s it (look )?(outside|out there)\b",
    # Assamese / Hindi — otherwise these skip the tool and the model
    # invents a forecast or asks which city.
    r"বতৰ",
    r"আবহাওয়া",
    r"मौसम",
    r"\b(botor|mausam)\b",
]

# Category (restaurant/hotel/gas station/...) is extracted by
# bella.tools.location.detect_category — a plain keyword lookup over a
# small fixed vocabulary, not LLM extraction. Deliberately requires both
# a place-type word AND a proximity word ("nearby"/"near me"/...) so a
# plain "restaurant" mention in unrelated conversation doesn't misfire.
# One vocabulary for "a place", reused by every pattern below. Seen live:
# "what is the nearest restaurant right now which is open at the moment?"
# matched nothing, fell through to the plain model, and got "I don't have
# real-time access to current locations" — an apology for a capability
# this actually has. A miss here is not a cheap failure: it hands a
# location question to something with no location.
_PLACE = (r"(?:restaurants?|hotels?|gas stations?|petrol(?: pump)?s?|fuel|"
          r"caf[eé]s?|coffee|pharmac(?:y|ies)|food|places? to eat|somewhere to eat|"
          r"bars?|atms?|hospitals?|clinics?|supermarkets?|groceries|grocery stores?"
          r"|movie theaters?|movie theatres?|cinemas?|theat(?:er|re)s?"
          r"|malls?|shopping malls?|banks?|parks?|parking|houses?|homes?"
          r"|kfc|mcdonald'?s|domino'?s|pizza hut|starbucks|subway|burger king)")
# "close to my location" is how this is actually asked. Without it,
# NEARBY missed and HERE's "my location" stole the turn — seen live:
# "restaurants close to my location" / "any KFC close to my location"
# both answered only "You're in Jorhat. That's the GPS talking."
_NEAR = (
    r"(?:nearby|near me|near here|near my location|around here|around me|"
    r"around my location|close by|close to(?: me| here| my location)?|"
    r"closest|nearest|in the area)"
)

# "Where am I?" is not weather. Seen live: "Bella মোৰ location-টো কি?" and
# "মোৰ ঠাইটুকুৰাৰ নাম কি?" missed every location group, fell through to
# Sarvam, and the model either invented a place name or told the user to
# ask for the weather — because the persona's weather fallback was the
# only location-shaped hint it had. Town name comes from GPS, never the
# model. Keep this narrower than a bare "location" so "location services"
# and STT junk like "watch my location" stay out.
HERE_PATTERNS = [
    r"\bwhere am i\b",
    r"\bwhere are we\b",
    r"\bwhat('s| is) my (current )?location\b",
    r"\bwhat('s| is) (the )?(name of )?(this |my |the )?(current )?(city|town|place)\b",
    r"\b(my |our )(current )?location\b",
    r"\bthe current location\b",
    r"\bcurrent location\b",
    r"\bwhat (city|town|place) am i (in|at)\b",
    r"\bwhich (city|town) (am i|are we) (in|at)\b",
    r"\bname of (this |the |my )?(town|city|place)\b",
    r"\b(tell me|say) (my |the )?(current )?(location|city|town)\b",
    # Assamese — ঠাই/জেগা/ক'ত আছোঁ, plus the English loan "location"
    # with the Assamese classifier (location-টো).
    r"ঠাইটুকুৰা",
    r"মোৰ ঠাই",
    r"ঠাই.{0,12}নাম",
    r"জেগা",
    r"ক['\u2019]?ত আছ",
    r"মোৰ location",
    r"location[- ]?টো",
    # Hindi
    r"मैं कहाँ हूँ",
    r"कहाँ हूँ",
    r"मेरी (लोकेशन|location)",
]

NEARBY_PATTERNS = [
    rf"\b{_NEAR}\b.*\b{_PLACE}\b",
    rf"\b{_PLACE}\b.*\b{_NEAR}\b",
    rf"\bfind (?:a |the |me )?(?:nearby |closest |nearest |good )?{_PLACE}\b",
    rf"\bwhere(?:'s| is| can i find)? (?:the )?(?:nearest|closest) ?{_PLACE}?\b",
    # "where can I eat", "somewhere to grab a coffee" — the intent is a
    # place near the user even with no proximity word at all.
    r"\bwhere can i (?:eat|get (?:some )?(?:food|coffee|lunch|dinner|breakfast))\b",
    # Seen live: "I wanted to have some crispy fried chicken" reached
    # chat, which invented "Chicken Curry & Wings" at 850 m.
    r"\b(?:want(?:ed)? (?:to have |some )?|craving |get (?:some )?)"
    r".{0,24}\b(?:chicken|pizza|burger|biryani|noodles|momos|thali|food)\b",
    r"\b(?:what'?s|whats) (?:near me|around me|around here|close by)\b",
    r"\bplaces? (?:near me|around me|around here|nearby|close by)\b",
    rf"\b(?:any|some) (?:good |decent |nice )?{_PLACE}\b.*\b(?:open|around|here|near)\b",
    # Asking for the names is the whole request — no proximity word.
    # Seen live: "মোৰ ওচৰৰ restaurant-ৰ নাম" skipped the tool and Sarvam
    # invented Spice Garden / Burger Junction.
    rf"\b(?:list|show|give me|tell me|what(?:'s| is| are)?) (?:some |the |any |a )?(?:nearby |nearest |closest |local |good )?{_PLACE}\b",
    rf"\bnames? of (?:the |some |any )?(?:nearby |nearest |local )?{_PLACE}\b",
    rf"\b{_PLACE}\b[- ]?(?:names?|list)\b",
    # "identify restaurants close to my location" / "any KFC close to…"
    rf"\b(?:identify|find|list|show|any|some)\b.+\b{_PLACE}\b",
    rf"\b{_PLACE}\b.+\b(?:location|nearby|near|close)\b",
    r"\b(?:is there (?:any |a )?|any |find )\w[\w'&.-]{1,24}.+\b(?:close to|near|nearby|my location)\b",
    # Assamese / Hindi — restaurant as an English loan is how it is
    # actually said. ৰেষ্টুৰেণ্ট / रेस्तरां for the native spellings.
    r"ওচৰ.{0,40}(?:restaurant|ৰেষ্ট[োু]ৰেণ্ট|হোটেল|cafe|ক্যাফে|চিনেমা|পেট্ৰল)",
    r"(?:restaurant|ৰেষ্ট[োু]ৰেণ্ট|হোটেল).{0,24}(?:ওচৰ|নাম|পাঁজৰে)",
    r"restaurant[- ]?ৰ নাম",
    r"ৰেষ্ট[োু]ৰেণ্ট",
    r"ওচৰে.?পাঁজৰে",
    r"লাঞ্চ.{0,48}(?:বাহিৰ|ৰেষ্ট|restaurant)",
    r"খাবলৈ.{0,32}(?:বাহিৰ|ৰেষ্ট|restaurant)",
    r"চিনেমা",
    r"মুভি.{0,24}(?:চাব|হল|ওচৰ)",
    r"ছবি.{0,16}হল",
    r"आसपास.{0,16}(?:रेस्तरां|होटल|restaurant)",
    r"रेस्तरां",
    # Catch-all so an unlisted place type still hits Google Maps, not HERE
    # or the chat model. Seen live: "nearest movie theater to my location"
    # answered only "You're in Jorhat."
    r"\b(?:nearest|closest|nearby)\b.{0,48}\b(?:to my location|near me|close to me|close to mine|around here)\b",
    r"\b(?:find|show|any|list)\b.{0,24}\b(?:close to mine|close to my location|near me|nearby)\b",
]

DISTANCE_PATTERNS = [
    r"\bhow far\b",
    r"\bhow long (?:does it take|to (?:get|drive|go|reach))\b",
    r"\bdistance (?:to|from|between)\b",
    r"\b(?:drive|travel) time\b",
    r"\bdirections to\b",
    r"\broute to\b",
]

# "What can you do?" answered from the actual tool list rather than by the
# model. Deterministic for the same reason calendar reads are: the model
# does not know what tools exist, and — measured live — it anchors on its
# own earlier denial and repeats it. Asked "do you have access to my local
# folders?" it replied "I don't have direct access to your personal
# files", which is false, and teaches the user not to bother asking.
# Checked AFTER the action groups, so "read my screen" still reads the
# screen instead of describing the ability to.
# Which model is answering is a setting, so it has an exact answer.
# Asked live, the model replied "I'm based on OpenAI's GPT-4" and, on the
# local model, "I am using an Apple MacBook Pro with M1 chip" — both
# invented, and both about the one subject the code knows for certain.
MODEL_QUESTION_PATTERNS = [
    # Both word orders: "which model ARE YOU using" and "which model YOU
    # ARE using" — the second is how it was actually asked out loud.
    r"\b(which|what) (model|llm|ai)\b[^.?!]{0,30}\b(are you|you are|you.re|is this|do you|you.?ve got)\b",
    r"\bwhat('s| is) (your|the) (model|llm|backend)\b",
    r"\bare you (running|using) (on |a )?(local|groq|gemini|ollama|cloud)\b",
    r"\bwho (made|built|trained) you\b",
]

# Jokes must not fall through to Sarvam's polite-mode. Seen live:
# "adult joke" / "dirty joke" / "sexy joke" in Assamese got a children's
# punchline. Route here so Groq tells the actual joke.
JOKE_PATTERNS = [
    r"\b(adult|dirty|sexy|filthy|rude|nsfw)\s+joke\b",
    r"\b(dirty|sexy|filthy)\s+jokes?\b",
    r"\btell (me )?(a |an )?(adult |dirty |sexy |filthy )?(joke|jokes)\b",
    r"\b(a |an )?(adult |dirty |sexy )?joke\b",
    r"\bjoke\s+(এটা|eta|eka)\b",
    r"\b(এটা|eta)\s+(adult |dirty |sexy )?joke\b",
    r"জোক",
    r"ধেমালি\s*(কথা|কৈ|কও|কোৱা)",
]

# "Can you hear me?" must not go to Groq — seen live it answered
# "You said you're in Jorhat" from leftover context.
PING_PATTERNS = [
    r"\bcan you hear me\b",
    r"\bare you (there|listening|awake)\b",
    r"\byou (there|listening)\b",
]

CAPABILITY_PATTERNS = [
    r"\bwhat (can|do) you do\b",
    r"\bwhat are (your|the) (capabilities|abilities|features|tools)\b",
    r"\bwhat can you help (me )?with\b",
    r"\bdo you have access to\b",
    r"\bcan you access\b",
    r"\bdo you have (any )?(tools|abilities|access)\b",
    r"\bwhat (tools|abilities) do you have\b",
    r"\bare you able to (access|use|control)\b",
]

# Confirmed — opens a real app with a real URL. See MAPS_PATTERNS' domain
# entry in _TOOL_GROUPS: ("location", "open_maps"), gated in
# cli.handle_open_maps.
MAPS_PATTERNS = [
    r"\bopen (that |this |it )?(in|on) (apple )?maps\b",
    r"\bshow (that |this |it )?on (the |a )?map\b",
    r"\bopen maps\b",
]

# Reminders land in Google Calendar (timed) or Google Tasks (dated) —
# see bella/tools/reminder.py. Checked BEFORE the call group, because
# "remind me to call mum at 5" is a reminder, not a request to phone her:
# measured, that exact sentence routed to tools:call.
REMINDER_PATTERNS = [
    r"\bremind me\b",
    r"\bset (?:up )?(?:an? )?reminder\b",
    r"\b(?:add|create|make|put) (?:an? )?reminder\b",
    r"\breminder (?:to|about|for)\b",
    r"\bdon'?t let me forget\b",
]

# "What's the phone number for X" — a LOOKUP, never a dial. Checked before
# CALL_PATTERNS so asking for a number does not ring anybody.
#
# It exists because there was no route at all: measured live, "do you have
# the contact number at the KFC?" fell through to the chat model, which
# answered "+91-361-2504870" with total confidence. That number is
# invented — 0361 is the Guwahati code, not Jorhat — and the persona
# explicitly forbids inventing phone numbers. The only defence that works
# is never letting the question reach the model.
PHONE_LOOKUP_PATTERNS = [
    r"\b(?:contact|phone|mobile|tele(?:phone)?)\s*(?:number|no\.?)s?\b",
    r"\bnumbers? (?:of|for|at)\b",
    r"\bwhat(?:'s| is| are)\b[^.?!]{0,24}\bnumbers?\b",
    # Seen live: "check their numbers", "please share the number if you
    # have". Singular-only "number of" and "give me/find" missed both,
    # the chat model said "let me pull that up" and never looked.
    r"\b(?:do you have|give me|find|get me|check|share|tell me|pull up|look up)\b[^.?!]{0,40}\bnumbers?\b",
    r"\b(?:need|want) (?:the |their )?(?:contact |phone )?numbers?\b",
    r"\btheir numbers?\b",
]

# Confirmed, deterministic — NOT LLM-mediated, same reasoning as
# SHELL_PATTERNS above: calling a real phone number is a real-world
# action, so whether it happens can't depend on a model reliably emitting
# a tool-call. bella.tools.calling.parse_phone_number does the actual
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
    # Name no longer has to be the last thing in the utterance. Seen live:
    # "Can you call Nova, call and reserve a booking?" missed the $-anchored
    # form, fell through to chat, and the model asked for a number — even
    # though the restaurant had just been listed and handle_call already
    # looks names up. "and"/"or" stay excluded so "call and reserve" is
    # not treated as a place called "and".
    r"\b(?:call|ring|phone)\s+(?:up\s+)?(?:the\s+)?"
    r"(?!it\b|this\b|that\b|me\b|you\b|us\b|back\b|off\b|out\b|in\b|for\b|on\b|upon\b|"
    r"and\b|or\b|logs?\b|"
    r"an? (?:meeting|ambulance|taxi|cab|uber)\b)"
    r"[A-Za-z][\w'&.\-]*",
    r"\bcall (them|him|her|that place|that one|that restaurant|that hotel)\b",
]

# Deliberately narrow — "write a script to rename files" must still escalate
# to Claude Code (matches CLAUDE_CODE_PATTERNS' "write a script"), not get
# swallowed here. These only match direct, simple file operations that still
# need LLM extraction (a path/content isn't literally present in the text
# the way a shell command is) — see SHELL_PATTERNS below for the
# deterministic, non-LLM-mediated shell path.
# Things that only exist on a screen. The guide-me patterns below all
# require one, so "show me"/"where is" can't claim a sentence that was
# never about the display.
_CONTROL = (
    r"(?:button|tab|link|icon|menu|option|setting|toggle|checkbox|"
    r"field|box|bar|panel|window|dialog|screen|page|cursor|pointer|mouse)"
)

# Circle part of the screen and ask about just that (bella/region_select.py).
# Checked BEFORE the whole-screen group, so "what is this area" opens the
# crayon rather than describing the entire desktop — "what's on my screen"
# has no area word and still gets the whole thing.
REGION_PATTERNS = [
    r"\b(circle|highlight|select|mark)\b[^.?!]{0,20}\b(this|that|an? )?(area|region|part|section|bit|portion)\b",
    r"\blet me (circle|highlight|select|mark|show you)\b",
    r"\bi'?ll (circle|highlight|select|mark) (it|this|that)\b",
    r"\bwhat('s| is)\b[^.?!]{0,12}\b(this|that)\b[^.?!]{0,12}\b(area|region|part|section|bit|picture|image|diagram|photo|chart|graph)\b",
    r"\b(tell me|explain|describe)\b[^.?!]{0,20}\b(this|that)\b[^.?!]{0,12}\b(area|region|part|section|picture|image|diagram|photo|chart|graph)\b",
    r"\bwhat (is|are) (this|these|that) (picture|image|diagram|photo|chart|graph|thing)s?\b",
    r"\b(circle|highlight) (it|this|that)\b",
    r"\bonly (this|that) (part|area|bit|section)\b",
    r"\bjust (this|that) (part|area|bit|section)\b",
]

# Deterministic, not LLM-mediated — there's nothing to extract from the
# phrasing (unlike browser targets/URLs below), so this is the same
# "cheap and direct" treatment as calendar/notes/email reads.
SCREEN_PATTERNS = [
    # The apostrophe was required, so a plain "whats on my screen" missed.
    # Whisper also mangles this exact phrase more than any other line said
    # to Bella, because it is said more than any other line — seen live:
    # "I watch on my screen", which reached the chat model and got "What
    # would you like to look at?" instead of a screenshot. The manglings
    # all keep "on my screen"; it is the front of the sentence that goes.
    r"\b(?:what'?s|what is|wat'?s|watch|what son|was)\s+on\s+(?:my |the )?screen\b",
    r"\bwhat'?s the question\b",
    r"\bwhat is the questions?\b",
    r"\bquestions? (?:in|on) (?:this |the |that )?page\b",
    r"\b(?:there is a |read the )?question on (?:the |this )?page\b",
    r"\bthere is a question\b",
    r"\bread the question\b",
    r"\bexplain (?:me )?(?:the |this )?question\b",
    # Assamese / Saaras spellings of the same ask. Seen live: "পেলা ৱাটছ
    # অট মাই স্ক্ৰীণ?" reached chat, which said it could not see the
    # screen and asked the student to describe it.
    r"স্ক্ৰীণ|স্ক্ৰেইন|স্ক্ৰিন|স্ক্ৰীন|স্ক্রিন|স্ক্রীন",
    r"ৱাটছ.{0,16}স্ক্ৰ",
    # Live: "মোৰ স্ক্ৰিনত কি দেখিছা তুমি?" — Assamese ৰ + short ই + ন,
    # not the ণ spelling already listed. Went to Sarvam, which said it
    # could not see the screen.
    r"স্ক্ৰিনত|মোৰ স্ক্ৰ|কি দেখিছা",
    r"পৰ্দাত কি|পৰ্দাখনত",
    r"\bwhats?\s+ot\s+(?:my |the )?screen\b",
    r"^\s*(?:i\s+)?watch\s+on\s+(?:my |the )?screen\b",
    # "what's on" heard as "what son" — the s migrates onto the next word.
    r"\bwhat\s+son\s+(?:my |the )?screen\b",
    r"\bwhat (can |do )?you (see|tell).{0,40}(screen|desktop)\b",
    r"\bwhat('s| is) on (this|the) page\b",
    # "Summarize this page" — read what is actually on screen rather than
    # letting the chat model invent an article it has never seen.
    r"\b(summari[sz]e|sum up|tldr|give me the gist)\b[^.?!]{0,24}\b(this|the|that|it)?\s*"
    r"(page|article|document|doc|screen|tab|site|website|post|email|thread)?\b",
    r"\bdescribe what i('m| am) (looking at|seeing)\b",
    r"\bwhat do you see\b",
    r"\btake a screenshot\b",
    # The obvious ways to ask, none of which matched: "what am I looking
    # at", "describe my screen", "read my screen", "can you see my screen".
    r"\bwhat am i (looking at|seeing)\b",
    r"\b(describe|read|check|look at|analyse|analyze) (my |the |this )?screen\b",
    r"\bcan you see (my |the )?screen\b",
    r"\bwhat('s| is) (this|that) on (my |the )?screen\b",
    # Questions *about* something on the screen, rather than a request for
    # a general description. These used to fall through to plain chat,
    # where the model has no screen and cheerfully invented an answer —
    # "what does the screen say about the installer stages?" produced a
    # confident description of an installer that was never on screen.
    r"\bwhat does (my |the )?screen say\b",
    r"\b(read|check|find|describe|explain|summari[sz]e)\b[^.?!]{0,40}\bon (my|the) screen\b",
    # Teacher / whole-desktop phrasing. Without these, "teach me this" and
    # "read my desktop" fell through to chat, which has no screen.
    # Anchored to the screen/file/desktop so "explain photosynthesis"
    # stays ordinary chat.
    r"\b(teach|tutor) (me )?(this|that|what.?s on|the screen|my (screen|desktop|code|file))\b",
    r"\bexplain (this|that|the screen|what.?s on|my (screen|desktop|code|file))\b",
    r"\bwalk me through (this|that|the screen|my (code|file|screen|desktop))\b",
    r"\bhelp me understand (this|that|the screen|what.?s on)\b",
    r"\bwhat('s| is) (this|that) (file|code|screen|page)\b",
    r"\bwhat file is (this|that|open)\b",
    r"\b(read|check|look at|explain) (this |that )(file|code)\b",
    r"\b(read|check|look at|explain) (the |my )?(code|desktop)\b",
    r"\bread (the |my )?file on (my |the )?screen\b",
    r"\bwhat('s| is) on (my |the )?desktop\b",
    r"\bread (my |the )?(whole )?desktop\b",
    # Guide-me: point the cursor at a control they cannot find.
    #
    # Every pattern in this group has to name something that lives on a
    # screen. Measured: without that requirement the trailing groups here
    # were all optional, so these two collapsed to a bare "show "/"where
    # is" and swallowed ordinary conversation — "where is Paris", "show me
    # a good recipe for pasta" and "how do I show gratitude" all took a
    # screenshot and answered from the desktop. A router miss costs one
    # turn; this cost a full capture + OCR + model pass on text that was
    # never about the screen at all, and it fires far more often than the
    # misses these were widened to catch.
    rf"\b(?:show|point|guide) (?:me )?(?:where |to )?(?:the |this |that )?[^.?!]{{0,40}}\b{_CONTROL}s?\b",
    r"\b(?:go to|take me to|navigate to|open)\b.{0,40}\btabs?\b",
    r"\bguide me (?:to|towards|onto)\b",
    # Tabs only. "click the Save button" is a desktop do-it (below).
    # Putting every button/link here stole that path and the harness
    # never ran. A named tab is a "show me where" request.
    r"\b(?:click|press|tap)\b.{0,40}\btabs?\b",
    r"\bwhere do i (click|press|tap|go)\b",
    r"\bhelp me (?:find|click|go to|open|get to)\b",
    # Bare "Help me" / "can you help me?" after a screen look. Without
    # this, those went to chat ("what do you need help with?") and the
    # pointer never moved. Anchored so "help me write an email" is still
    # email and "help me find an ATM" is still nearby.
    r"^(?:hey[, ]+)?(?:bella[, ]+)?(?:please )?(?:can you |could you )?(?:just )?"
    r"(?:help me|guide me|show me(?: it| that| this| where)?)\s*[.?!]*$",
    rf"\bwhere (?:is|are) (?:the |this |that )?[^.?!]{{0,40}}\b{_CONTROL}s?\b",
    r"\bi (can'?t|cannot|am not able to) see\b",
    r"\b(show|find) (me )?(the |this |that )?.{0,40}(button|tab|link|on (my |the )?screen)\b",
    r"\bwhat('s| is) this .{0,24}(button|link|tab|icon|menu)\b",
    r"\b(the |this |that )?(pull requests?|settings) (button|tab|link)\b",
    r"\bpoint (a |the )?(cursor|pointer|mouse)\b",
    # Pronoun required: bare "point out" is ordinary English ("point out
    # the obvious") and was routing it to a screenshot.
    r"\b(show|point) (it|that|this) (to me|out)\b",
    r"\bwhere exactly\b",
    # Bare "show me" as the whole line — the follow-up after Bella offers to
    # point something out. Anchored to the end of the utterance: with a
    # \b there instead, this matched the opening words of any sentence
    # ("show me a good recipe"), which is the same over-capture as above.
    r"^(?:hey[, ]+)?(?:bella[, ]+)?(?:yes[, ]+)?(?:can you )?show me\s*(?:it|that|this|where)?\s*[.?!]*$",
    # Teach wins over desktop: "how do I click Save" is a lesson, not a click.
    r"\bhow (do i|do you|can i|should i|to)\b.{0,48}\b(click|press|type)\b",
]

# Do-it phrasing for already-open native apps (bella/tools/desktop.py).
# Teach/show/where stay on SCREEN_PATTERNS above. Spotify play/pause is
# earlier in _TOOL_GROUPS, so "press play on Spotify" never reaches here.
# Ambiguous wording defaults to teaching, not clicking.
DESKTOP_PATTERNS = [
    r"\bclick (the |on )?.+\b(button|link|icon|menu|tab|checkbox|toggle)\b",
    r"\bclick (the |on )?.+\bin\b",
    r"\bpress (cmd|command|ctrl|control|alt|option|shift)\b",
    r"\bpress\b.{0,24}\b(key|shortcut)\b",
    r"\btype\b.+\b(into|in) (the |this |that )?",
    r"\b(just |please )?(click|press|type) (it|that|this)\b",
    r"\bdo (that|it|this) for me\b",
    r"\b(click|press) (it|that|this) for me\b",
]

# Browser actions are LLM-mediated (like FILES_PATTERNS below) — click
# targets, typed text, and URLs are too free-form for deterministic
# extraction. Every action these route to that changes state
# (open_url/click/type_text) is confirmed in cli.execute_browser_tool
# before it runs; only get_page_text/get_current_url are automatic.
# Native "click the Save button" now belongs to DESKTOP_PATTERNS.
BROWSER_PATTERNS = [
    r"\bfill (in |out )?(the )?form\b",
    r"\bnavigate to\b",
    r"\bopen (this |the )?(url|website|webpage|link)\b",
    # "can you open a browser?" — asked live, matched nothing, and the
    # model denied having the ability at all.
    r"\bopen (a |the )?browser\b",
    r"\b(open|launch|start) (chrome|chromium|safari|firefox)\b",
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

# Bella writes the text and, only after a yes, types it in (bella/compose.py).
# Checked AFTER the email group, so "draft an email to bob@x.com" still
# goes to Gmail, which can resolve the address and actually send it;
# compose.detect() stands aside on its own when a recipient is named.
COMPOSE_PATTERNS = [
    r"\b(write|draft|compose|create|prepare|make)\b[^.?!]{0,40}?\b"
    r"(prompt|letter|cover letter|covering letter|resume|résumé|cv|"
    r"instructions?|essay|paragraph|bio|caption|post)\b",
    # "write an email about the launch" — content-first, no recipient.
    r"\b(write|draft|compose)\b\s+(an?\s+|the\s+)?(e-?mail|message|reply|notes?)\b",
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
    # "can you check my local files" — asked live, matched nothing, and
    # the model answered "I don't actually have access to your personal
    # files", which is false. Any phrasing that names files/folders plus
    # a looking-at verb belongs here.
    r"\b(check|look at|look through|browse|access|go through|see)\b[^.?!]{0,20}\b(my |the |local )*(files?|folders?|documents|directory|directories)\b",
    r"\b(my|the|local) (files?|folders?)\b",
    # Ordinary phrasings that matched nothing: "what files do I have on my
    # Desktop", "do I have a file about taxes", "show me the contents of
    # README.md" (the existing rule needs the literal word "file" after
    # "contents of", which nobody says when naming an actual filename).
    r"\bwhat files\b",
    r"\bdo i have (a |any )?(file|document)s?\b",
    r"\bshow me (the )?contents of\b",
    r"\b(any|which) (files?|documents?)\b.*\b(desktop|documents|downloads|folder|directory)\b",
    r"\blook (in|at) (my |the )?(desktop|documents|downloads)\b",
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

# Undoing the last add. Checked BEFORE the write group, because "remove
# that meeting" contains "meeting" and would otherwise be read as a
# request to create one. This is the escape hatch for adding without a
# yes/no confirmation.
CALENDAR_UNDO_PATTERNS = [
    r"\b(remove|delete|cancel|undo|scrap|take (that|it) off)\b.*\b(that|it|event|meeting|appointment|calendar)\b",
    r"\b(remove|delete|undo) that\b",
    r"\bthat'?s wrong,? (remove|delete) it\b",
]

_TOOL_GROUPS = [
    ("calendar_undo", "auto", [re.compile(p, re.IGNORECASE) for p in CALENDAR_UNDO_PATTERNS]),
    ("calendar", "write", [re.compile(p, re.IGNORECASE) for p in CALENDAR_WRITE_PATTERNS]),
    ("calendar", "read", [re.compile(p, re.IGNORECASE) for p in CALENDAR_READ_PATTERNS]),
    ("notes", "write", [re.compile(p, re.IGNORECASE) for p in NOTES_WRITE_PATTERNS]),
    ("notes", "read", [re.compile(p, re.IGNORECASE) for p in NOTES_READ_PATTERNS]),
    ("email", "write", [re.compile(p, re.IGNORECASE) for p in EMAIL_WRITE_PATTERNS]),
    ("email", "read", [re.compile(p, re.IGNORECASE) for p in EMAIL_READ_PATTERNS]),
    ("github", "auto", [re.compile(p, re.IGNORECASE) for p in GITHUB_PATTERNS]),
    ("youtube", "auto", [re.compile(p, re.IGNORECASE) for p in YOUTUBE_PATTERNS]),
    ("googledrive", "auto", [re.compile(p, re.IGNORECASE) for p in DRIVE_PATTERNS]),
    ("cal", "auto", [re.compile(p, re.IGNORECASE) for p in CAL_COM_PATTERNS]),
    ("chatfai", "auto", [re.compile(p, re.IGNORECASE) for p in CHATFAI_PATTERNS]),
    ("googlecalendar", "auto", [re.compile(p, re.IGNORECASE) for p in GOOGLE_CALENDAR_PATTERNS]),
    ("googletasks", "auto", [re.compile(p, re.IGNORECASE) for p in GOOGLE_TASKS_PATTERNS]),
    ("linkedin", "auto", [re.compile(p, re.IGNORECASE) for p in LINKEDIN_PATTERNS]),
    ("reddit", "auto", [re.compile(p, re.IGNORECASE) for p in REDDIT_PATTERNS]),
    ("location", "showtimes", [re.compile(p, re.IGNORECASE) for p in SHOWTIME_PATTERNS]),
    # After showtimes so cinema stays cinema. Before nearby so
    # "hotels in Goa" is a listing search, while "hotels near me" still
    # hits Maps. Before distance so "flights from X to Y" is not a drive.
    ("discovery", "hotels", [re.compile(p, re.IGNORECASE) for p in DISCOVERY_HOTEL_PATTERNS]),
    ("discovery", "flights", [re.compile(p, re.IGNORECASE) for p in DISCOVERY_FLIGHT_PATTERNS]),
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
    ("voice", "volume_down", [re.compile(p, re.IGNORECASE) for p in VOICE_VOLUME_DOWN_PATTERNS]),
    ("voice", "volume_up", [re.compile(p, re.IGNORECASE) for p in VOICE_VOLUME_UP_PATTERNS]),
    ("spotify", "volume_down", [re.compile(p, re.IGNORECASE) for p in SPOTIFY_VOLUME_DOWN_PATTERNS]),
    ("spotify", "volume_up", [re.compile(p, re.IGNORECASE) for p in SPOTIFY_VOLUME_UP_PATTERNS]),
    ("spotify", "next", [re.compile(p, re.IGNORECASE) for p in SPOTIFY_NEXT_PATTERNS]),
    ("spotify", "previous", [re.compile(p, re.IGNORECASE) for p in SPOTIFY_PREVIOUS_PATTERNS]),
    ("spotify", "play_track", [re.compile(p, re.IGNORECASE) for p in SPOTIFY_PLAY_TRACK_PATTERNS]),
    ("news", "auto", [re.compile(p, re.IGNORECASE) for p in NEWS_PATTERNS]),
    ("location", "weather", [re.compile(p, re.IGNORECASE) for p in WEATHER_PATTERNS]),
    ("location", "distance", [re.compile(p, re.IGNORECASE) for p in DISTANCE_PATTERNS]),
    ("location", "nearby", [re.compile(p, re.IGNORECASE) for p in NEARBY_PATTERNS]),
    ("omdb", "auto", [re.compile(p, re.IGNORECASE) for p in OMDB_PATTERNS]),
    # After weather/nearby so "weather at my location" and "restaurants
    # near my location" stay weather/nearby. Bare "where am I" / "my
    # location" only reach here.
    ("location", "here", [re.compile(p, re.IGNORECASE) for p in HERE_PATTERNS]),
    ("location", "open_maps", [re.compile(p, re.IGNORECASE) for p in MAPS_PATTERNS]),
    ("joke", "auto", [re.compile(p, re.IGNORECASE) for p in JOKE_PATTERNS]),
    ("voice", "ping", [re.compile(p, re.IGNORECASE) for p in PING_PATTERNS]),
    ("reminder", "auto", [re.compile(p, re.IGNORECASE) for p in REMINDER_PATTERNS]),
    ("call", "number", [re.compile(p, re.IGNORECASE) for p in PHONE_LOOKUP_PATTERNS]),
    ("call", "auto", [re.compile(p, re.IGNORECASE) for p in CALL_PATTERNS]),
    # After every action group above, so a request to DO something is
    # never answered with a description of being able to do it.
    ("capabilities", "model", [re.compile(p, re.IGNORECASE) for p in MODEL_QUESTION_PATTERNS]),
    ("capabilities", "read", [re.compile(p, re.IGNORECASE) for p in CAPABILITY_PATTERNS]),
    # Checked before "files" — a request naming a literal command (e.g. "run
    # ls -la") must never fall through to the LLM-mediated files path.
    ("shell", "auto", [re.compile(p, re.IGNORECASE) for p in SHELL_PATTERNS]),
    ("compose", "auto", [re.compile(p, re.IGNORECASE) for p in COMPOSE_PATTERNS]),
    ("screen", "region", [re.compile(p, re.IGNORECASE) for p in REGION_PATTERNS]),
    ("screen", "read", [re.compile(p, re.IGNORECASE) for p in SCREEN_PATTERNS]),
    # After screen so "how do I click" still teaches. Before browser so a
    # native click is not answered with "Browser control is disabled."
    ("desktop", "auto", [re.compile(p, re.IGNORECASE) for p in DESKTOP_PATTERNS]),
    # Checked before "files" — e.g. "open this website" must not fall
    # through to the files domain's "open (the|a) file" pattern.
    ("browser", "auto", [re.compile(p, re.IGNORECASE) for p in BROWSER_PATTERNS]),
    ("files", "auto", [re.compile(p, re.IGNORECASE) for p in FILES_PATTERNS]),
]


# Saaras as-IN writes spoken ENGLISH in Assamese letters.
# Live: "what's the question?" -> "ৱাটছ দা কোৱাশ্চন?"
#        "there is a question on the page" -> "দেৰ ইজ এ কোশ্চেন অন দা পেজ।"
# Those never match English screen patterns, so Sarvam answers with
# "I cannot see your screen". Fold before routing. Idempotent.
_SCREEN_PHONETIC = (
    (re.compile(r"দেৰ ইজ এ কোশ্চেন অন দা পেজ[।.!?]*", re.I),
     "there is a question on the page"),
    (re.compile(
        r"ৱাট ইজ দা কোশ্চেনছ ইন দিছ পেজ(?: ৰাইট নাউ)?[।.!?]?",
        re.I,
    ), "what's the question on this page"),
    (re.compile(r"কোশ্চেন অন দিছ[।,!?]?", re.I),
     "question on this page"),
    (re.compile(r"ৱাটছ দা (?:কোৱাশ্চন|কোশ্চেন|কোৱেচন|কুৱেশ্চন)[।.!?]?", re.I),
     "what's the question"),
    (re.compile(r"ৱাটছ অন মাই (?:স্ক্ৰিন|স্ক্ৰীণ|স্ক্রিন)[।.!?]?", re.I),
     "what's on my screen"),
    (re.compile(r"(?:কান ইউ )?ৰিড মাই (?:স্ক্ৰিন|স্ক্ৰীণ|স্ক্রিন)[।.!?]?", re.I),
     "read my screen"),
    (re.compile(r"ষ্টপ দ(?:া|্য)? মিউজিক[।.!?]?", re.I), "stop the music"),
    (re.compile(r"কেন ইউ", re.I), "can you"),
    (re.compile(r"এক্সপ্লেইন", re.I), "explain"),
    (re.compile(r"ৰাইট নাউ", re.I), "right now"),
    (re.compile(r"কোশ্চেনছ", re.I), "questions"),
    (re.compile(r"কোৱাশ্চন|কোশ্চেন|কোৱেচন|কুৱেশ্চন", re.I), "question"),
    (re.compile(r"ৱাট ইজ", re.I), "what is"),
    (re.compile(r"দিছ পেজ", re.I), "this page"),
    (re.compile(r"পেজ", re.I), "page"),
)


def fold_screen_phonetic(text: str) -> str:
    """Assamese-letter English (Saaras) -> English screen phrasing."""
    out = text or ""
    for pattern, repl in _SCREEN_PHONETIC:
        out = pattern.sub(repl, out)
    return out


def detect_tool(text: str) -> tuple[str, str] | None:
    """Return (domain, action) e.g. ('calendar', 'read'), or None if this
    isn't a Calendar/Notes/Files/Shell-shaped request. Checked before
    decide_backend — tool calls are cheaper than both Ollama and Claude Code.
    'files' action is always 'auto': the LLM decides which specific file
    tool to call, since paths/content are too free-form for regex
    extraction. 'shell' is also 'auto' but is NOT LLM-mediated — see
    SHELL_PATTERNS above for why.
    """
    stripped = fold_screen_phonetic(text.strip())
    # A whole utterance that is just a phone number — the follow-up after
    # "call Sabina" when they dictate digits. Seen live: "1 2 3 2 4 9 8 0 0"
    # fell through to chat and Groq talked about the number instead of dialling.
    digits_only = re.sub(r"[\d\s+\-().]", "", stripped)
    digit_count = sum(c.isdigit() for c in stripped)
    if digit_count >= 7 and len(digits_only) <= 2:
        return "call", "auto"
    # "what do you call studio mode" is English idiom, not a phone call.
    # Seen live: that phrase hit tools:call and then Apple Maps error 4.
    _not_a_call = bool(re.search(
        r"\bwhat(?:'s| is| do you| you)?\s+call\b", stripped, re.IGNORECASE
    ))
    _music_volume = bool(re.search(
        r"\b(spotify|music|song|track)\b", stripped, re.IGNORECASE
    ))
    # "Write an email about the launch" names no recipient — it is a
    # document to compose into whatever is on screen, not a Gmail draft to
    # address and send. "Draft an email TO Sabina" still belongs to Gmail,
    # which can resolve the name to an address and actually send it, so
    # the presence of a recipient is what decides.
    _compose_email = (
        bool(re.search(r"\b(write|draft|compose)\b[^.?!]{0,24}\be-?mail\b",
                       stripped, re.IGNORECASE))
        and not re.search(r"\bto\s+\S", stripped, re.IGNORECASE)
        and bool(re.search(r"\b(about|regarding|explaining|announcing)\b",
                           stripped, re.IGNORECASE))
    )
    for domain, action, patterns in _TOOL_GROUPS:
        if domain == "call" and _not_a_call:
            continue
        if domain == "email" and action == "write" and _compose_email:
            continue
        # "volume down" is Bella's voice unless they named the music.
        if domain == "voice" and action.startswith("volume") and _music_volume:
            continue
        # Cinema board is showtimes / Maps, not a track and not OMDb.
        if domain in ("spotify", "omdb") and re.search(
            r"\b(theater|theatre|cinema|showtimes?|now showing|screen)\b|চিনেমা|থিয়েটাৰ|ছবি.{0,12}চলি",
            stripped,
            re.IGNORECASE,
        ):
            continue
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
