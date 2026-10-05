"""Spoken-friendly formatting over bella/location_client.py — same
separation as bella/tools/spotify.py wrapping bella/spotify_client.py.
Weather and nearby-place lookups are read-only; the confirmed "open in
Apple Maps" action itself lives in cli.py next to the other confirm-gated
tools (execute_browser_tool is the direct precedent), not here.
"""
import difflib
import re
import urllib.parse

from bella import location_client as lc

_HERE_RE = re.compile(
    r"\b(here|my location|my city|current location|this city|this town|"
    r"outside|around here|near me|locally)\b",
    re.IGNORECASE,
)
# Trailing chatter after a city. "in Bangalore today?" used to fail the
# old $-anchored regex because of the question mark, then weather silently
# used GPS (Jorhat) — seen live.
_WEATHER_TRAIL_RE = re.compile(
    r"(?:\s+(?:today|tonight|tomorrow|now|please|right now|currently|"
    r"this (?:morning|afternoon|evening|week)))*[?.!,\s]*$",
    re.IGNORECASE,
)
# Spoken English often inserts "the" ("in the New York"). That used to
# trip the negative lookahead, so the city was dropped and weather
# silently used GPS — seen live on "what's the weather in the New York".
_NAMED_PLACE_RE = re.compile(
    r"\b(?:in|at|for|of|over)\s+(?:the\s+)?(?!my\b|this\b|here\b|our\b|your\b|there\b)"
    r"([A-Za-z\u0980-\u09FF][A-Za-z\u0980-\u09FF\s.'-]{0,40})",
    re.IGNORECASE,
)
_LEADING_PLACE_RE = re.compile(
    r"^(.+?)\s+(?:weather|forecast|temperature)\s*$",
    re.IGNORECASE,
)
_NOT_A_PLACE = {
    "the", "a", "an", "my", "our", "your", "this", "that", "here",
    "current", "today", "today's", "todays", "local", "now", "please",
}
# "what's the weather" must not become the place "what's the".
_NOT_A_PLACE_WORDS = _NOT_A_PLACE | {
    "what", "what's", "whats", "how", "how's", "hows", "is",
    "tell", "check", "give", "get", "me",
}

CATEGORY_LABELS = {
    "restaurant": "restaurants",
    "hotel": "hotels",
    "gas_station": "gas stations",
    "cafe": "cafes",
    "pharmacy": "pharmacies",
    "movie_theater": "movie theaters",
    "hospital": "hospitals",
    "atm": "ATMs",
    "supermarket": "supermarkets",
    "bank": "banks",
    "parking": "parking",
    "park": "parks",
    "shopping_mall": "malls",
}

# Maps common spoken phrasings onto the fixed category vocabulary in
# location_client._GOOGLE_TYPES — deliberately a plain dict lookup, not
# LLM extraction (see the plan's Context section for why: this is a small
# closed vocabulary, so a keyword match is strictly safer and cheaper).
_CATEGORY_KEYWORDS = {
    "restaurant": "restaurant",
    "restaurants": "restaurant",
    "food": "restaurant",
    "eat": "restaurant",
    "ৰেষ্টুৰেণ্ট": "restaurant",
    "ৰেষ্টোৰেণ্ট": "restaurant",
    "ৰেষ্টোৰেন্ট": "restaurant",
    "লাঞ্চ": "restaurant",
    "रेस्तरां": "restaurant",
    "hotel": "hotel",
    "hotels": "hotel",
    "motel": "hotel",
    "হোটেল": "hotel",
    "होटल": "hotel",
    "gas station": "gas_station",
    "gas": "gas_station",
    "petrol": "gas_station",
    "fuel": "gas_station",
    "cafe": "cafe",
    "coffee": "cafe",
    "ক্যাফে": "cafe",
    "pharmacy": "pharmacy",
    "pharmacies": "pharmacy",
    "chemist": "pharmacy",
    "drugstore": "pharmacy",
    "movie theater": "movie_theater",
    "movie theatre": "movie_theater",
    "movie theatres": "movie_theater",
    "movie theaters": "movie_theater",
    "cinema": "movie_theater",
    "cinemas": "movie_theater",
    "theatre": "movie_theater",
    "theaters": "movie_theater",
    "theater": "movie_theater",
    "hospital": "hospital",
    "hospitals": "hospital",
    "atm": "atm",
    "atms": "atm",
    "supermarket": "supermarket",
    "supermarkets": "supermarket",
    "grocery": "supermarket",
    "groceries": "supermarket",
    "mall": "shopping_mall",
    "malls": "shopping_mall",
    "shopping mall": "shopping_mall",
    "bank": "bank",
    "banks": "bank",
    "parking": "parking",
    "car park": "parking",
    "park": "park",
    "parks": "park",
}


def detect_category(text: str) -> str | None:
    """Word-boundary match, longest phrase first.

    Substring matching treated 'theater' as 'eat' → restaurants. Seen live:
    nearest movie theater was answered as if it were food, or as GPS only.
    Assamese has no \b, so Indic keywords are a plain substring.
    """
    raw = text or ""
    lowered = raw.lower()
    if re.search(r"চিনেমা|মুভি|ছবি\s*হল|cinema|movie theater|movie theatre", raw, re.I):
        return "movie_theater"
    if re.search(r"ৰেষ্ট[োু]ৰেণ্ট|restaurant", raw, re.I):
        return "restaurant"
    if re.search(r"লাঞ্চ|খাবলৈ|বাহিৰত যাম", raw) and not re.search(
        r"হোটেল|চিনেমা|theater|cinema|পেট্ৰল", raw, re.I
    ):
        return "restaurant"
    for keyword, category in sorted(_CATEGORY_KEYWORDS.items(), key=lambda kv: -len(kv[0])):
        if re.search(r"[\u0980-\u09FF]", keyword):
            if keyword in raw:
                return category
        elif re.search(rf"\b{re.escape(keyword)}\b", lowered):
            return category
    return None


def _place_phrase(loc: dict) -> str:
    city = (loc.get("city") or "").strip()
    return f" in {city}" if city else ""


def extract_weather_place(text: str) -> str | None:
    """A city the user named, or None to use current GPS.

    'here' / 'my location' are not cities. Never invent one.
    """
    if not text or _HERE_RE.search(text):
        return None
    stripped = _WEATHER_TRAIL_RE.sub("", text.strip()).strip()
    match = _NAMED_PLACE_RE.search(stripped)
    place = ""
    if match:
        place = " ".join(match.group(1).split()).strip(" .?!,")
    else:
        lead = _LEADING_PLACE_RE.match(stripped)
        if lead:
            place = " ".join(lead.group(1).split()).strip(" .?!,")
    place = _WEATHER_TRAIL_RE.sub("", place).strip(" .?!,")
    words = set(place.lower().split())
    if (
        not place
        or _HERE_RE.search(place)
        or place.lower() in _NOT_A_PLACE
        or words & _NOT_A_PLACE_WORDS
    ):
        return None
    return place


def describe_here() -> str:
    """Town name only. No GPS lecture."""
    loc = lc.get_location()
    city = (loc.get("city") or "").strip()
    if city:
        return f"You're in {city}."
    return "I couldn't tell which town you're in right now."


def weather_horizon(text: str) -> str:
    """today | tomorrow | week — from the user's words, not a guess."""
    t = (text or "").lower()
    if re.search(r"\b(this week|next (few )?days|forecast)\b", t):
        return "week"
    if re.search(r"\btomorrow\b", t):
        return "tomorrow"
    return "today"


# How alike two place names must be to count as the same place.
# Tuned on the real failures: "johar"/"jorhat" = 0.73, "johra"/"jorhat" =
# 0.73, while "bangalore"/"jorhat" = 0.25 and "delhi"/"jorhat" = 0.18.
_SAME_PLACE = 0.70
# Below this, whatever the geocoder returned is not what was asked for.
_GEOCODE_TRUSTED = 0.60


def _alike(a: str, b: str) -> float:
    return difflib.SequenceMatcher(
        None, (a or "").lower().strip(), (b or "").lower().strip()
    ).ratio()


def resolve_named_place(name: str, here: dict | None):
    """Turn a spoken place name into somewhere real, or say it can't.

    Returns ("local", None) to use the current GPS position,
    ("named", geocode_hit) for a place elsewhere, or ("unknown", None).

    This exists because speech-to-text mangles the user's OWN town more
    than any other word, and the geocoder answers anyway. Measured live:
    "Jorhat" came through as "Johar" and "Johra", which Nominatim resolved
    to Karang Baru (2,528 km away) and Badhiyana (1,621 km away). Bella then
    reported entirely genuine weather for entirely the wrong country, in a
    confident voice — indistinguishable from making it up, and worse,
    because there is nothing in the answer to suggest checking it.

    So: a name that sounds like where they already are IS where they are,
    and a geocode that comes back with a name nothing like what was asked
    is not trusted.
    """
    wanted = (name or "").strip()
    if not wanted:
        return "local", None
    city = ((here or {}).get("city") or "").strip()
    if city and _alike(wanted, city) >= _SAME_PLACE:
        # "Johar" while standing in Jorhat. Also saves a geocode call.
        return "local", None
    hit = lc.geocode_place(wanted)
    if not hit:
        return "unknown", None
    if _alike(wanted, hit.get("city") or "") < _GEOCODE_TRUSTED:
        return "unknown", None
    return "named", hit


def describe_weather(
    include_rain: bool = True, user_text: str = "", language: str | None = None
) -> str:
    named = extract_weather_place(user_text)
    here = None
    try:
        here = lc.get_location()
    except lc.LocationError:
        pass
    kind, hit = resolve_named_place(named, here) if named else ("local", None)

    if kind == "unknown":
        # Name it rather than answering for somewhere random, and offer
        # the town they are actually standing in as the likely intent.
        near = (here or {}).get("city") or ""
        if language == "as-IN":
            return f"{named} নামৰ ঠাই বিচাৰি পোৱা নগ'ল।"
        if near:
            return f"I couldn't place \u201c{named}\u201d. Did you mean {near}?"
        return f"I couldn't find a place called {named}."

    if kind == "named":
        weather = lc.get_weather(hit["lat"], hit["lon"])
        loc = {"city": hit.get("city") or named, "source": "named",
               "lat": hit["lat"], "lon": hit["lon"]}
    else:
        loc = here if here else lc.get_location()
        weather = lc.get_weather(loc["lat"], loc["lon"])
    place = _place_phrase(loc)
    horizon = weather_horizon(user_text)
    days = weather.get("days") or []
    if language == "as-IN":
        return _format_weather_as(weather, loc, horizon, include_rain, days)
    if horizon == "tomorrow" and len(days) >= 2:
        d = days[1]
        high, low = d.get("high_c"), d.get("low_c")
        temps = ""
        if high is not None and low is not None:
            temps = f", {low:.0f} to {high:.0f}\N{DEGREE SIGN}C"
        rain = ""
        if d.get("rain_chance") is not None and d["rain_chance"] >= 40:
            rain = f" Rain about {d['rain_chance']:.0f}%."
        return f"Tomorrow{place}: {d['condition']}{temps}.{rain}"
    if horizon == "week" and days:
        bits = []
        import datetime as _dt
        for i, d in enumerate(days[:3]):
            if i == 0:
                label = "Today"
            elif i == 1:
                label = "Tomorrow"
            else:
                try:
                    label = _dt.date.fromisoformat(d["date"]).strftime("%A")
                except (TypeError, ValueError):
                    label = f"Day {i + 1}"
            high = d.get("high_c")
            bit = f"{label} {d['condition']}"
            if high is not None:
                bit += f" {high:.0f}\N{DEGREE SIGN}"
            bits.append(bit)
        return f"Forecast{place}: " + "; ".join(bits) + "."
    temp = weather["temperature_c"]
    line = f'It\'s currently {temp:.0f}\N{DEGREE SIGN}C with {weather["condition"]}{place}.'
    if include_rain:
        rain = weather.get("rain_today")
        if rain:
            line += " " + rain
    return line


_AS_CITIES = {
    "jorhat": "যোৰহাট",
    "guwahati": "গুৱাহাটী",
    "gauhati": "গুৱাহাটী",
    "dispur": "দিছপুৰ",
    "dibrugarh": "ডিব্ৰুগড়",
    "tezpur": "তেজপুৰ",
    "silchar": "ছিলচৰ",
    "nagaon": "নগাঁও",
    "tinsukia": "তিনিচুকীয়া",
    "bangalore": "বেংগালুৰু",
    "bengaluru": "বেংগালুৰু",
    "delhi": "দিল্লী",
    "new delhi": "নতুন দিল্লী",
    "mumbai": "মুম্বাই",
    "kolkata": "কলকাতা",
}

_AS_COND = (
    ("thunderstorm", "ধুমুহা-বজ্ৰপাত"),
    ("heavy rain", "তীব্ৰ বৰষুণ"),
    ("light rain", "পৰুৱা বৰষুণ"),
    ("rain shower", "বৰষুণৰ জাক"),
    ("drizzle", "শিৰশিৰীয়া বৰষুণ"),
    ("rain", "বৰষুণ"),
    ("clear", "পৰিষ্কাৰ আকাশ"),
    ("sunny", "ৰ'দ"),
    ("partly cloudy", "অলপ ডাৱৰীয়া"),
    ("overcast", "ডাৱৰে ঢকা"),
    ("cloudy", "ডাৱৰীয়া"),
    ("fog", "কুঁৱলী"),
    ("mist", "কুঁৱলী"),
    ("haze", "ধোঁৱা-কুঁৱলী"),
    ("snow", "হিমপাত"),
)


def _as_city(name: str) -> str:
    key = (name or "").strip().lower()
    return _AS_CITIES.get(key, name or "")


def _as_cond(text: str) -> str:
    t = (text or "").lower()
    for en, as_ in _AS_COND:
        if en in t:
            return as_
    return text or "মিশ্ৰিত আকাশ"


def _format_weather_as(
    weather: dict, loc: dict, horizon: str, include_rain: bool, days: list
) -> str:
    city = _as_city(loc.get("city") or weather.get("city") or "")
    where = f"{city}ত " if city else ""
    if horizon == "tomorrow" and len(days) >= 2:
        d = days[1]
        high, low = d.get("high_c"), d.get("low_c")
        temps = ""
        if high is not None and low is not None:
            temps = f", {low:.0f}ৰ পৰা {high:.0f} ডিগ্ৰীছে"
        rain = ""
        chance = d.get("rain_chance")
        if chance is not None and chance >= 40:
            rain = f" বৰষুণৰ সম্ভাৱনা প্ৰায় {chance:.0f} শতাংশ।"
        return f"কাইলৈ {where}{_as_cond(d.get('condition', ''))}{temps}।{rain}"
    temp = weather["temperature_c"]
    line = f"এতিয়া {where}{temp:.0f} ডিগ্ৰীছে, {_as_cond(weather.get('condition', ''))}।"
    if include_rain:
        rain = weather.get("rain_today") or ""
        if "no rain" in rain.lower():
            line += " আজি বৰষুণৰ সম্ভাৱনা নাই।"
        elif rain:
            m = re.search(r"(\d+)\s*%", rain)
            if m:
                line += f" আজি বৰষুণৰ সম্ভাৱনা {m.group(1)} শতাংশ।"
            else:
                line += " আজি বৰষুণ হ'ব পাৰে।"
    return line


# Names from the most recent nearby search, so "call them" / "call that
# place" has something concrete to resolve instead of the model guessing.
# Names only: these come from OpenStreetMap, which almost never carries
# phone numbers here (1 of 32 within 5km), so the number itself is looked
# up through Apple Maps at call time by name.
_last_nearby_names: list[str] = []
# The place they last named from that list ("Novotel"), so "call them" /
# "share the number" does not ask "which one" after they already said it.
_last_focus_place: str | None = None


def remember_places(names: list[str]) -> None:
    global _last_nearby_names, _last_focus_place
    _last_nearby_names = [n for n in names if n and n != "(unnamed)"]
    _last_focus_place = None


def last_places() -> list[str]:
    return list(_last_nearby_names)


def focused_place() -> str | None:
    return _last_focus_place


def note_mentioned_place(text: str) -> str | None:
    """If this turn names a remembered place, that is now the one they mean."""
    global _last_focus_place
    matched = match_remembered_place(text)
    if matched:
        _last_focus_place = matched
    return matched


def match_remembered_place(target: str) -> str | None:
    """Resolve a spoken fragment ('Nova') onto the last nearby list.

    Nearby results are nearest-first, so the first substring match is the
    closest one. Two spellings of the same shop (Nova AKhol / Nova Akhol)
    therefore collapse to the nearest rather than looking ambiguous.
    """
    needle = " ".join((target or "").lower().split())
    if not needle:
        return None
    hits = [n for n in _last_nearby_names if needle in n.lower() or n.lower() in needle]
    if hits:
        return hits[0]
    # "Nova Arcol" vs "Nova AKhol" — STT mangles the second word. Share
    # any 3+ letter token and take the nearest (list is nearest-first).
    words = [w for w in needle.split() if len(w) >= 3]
    if not words:
        return None
    for name in _last_nearby_names:
        tokens = name.lower().split()
        if any(w in tokens or any(w in t or t in w for t in tokens if len(t) >= 3) for w in words):
            return name
    return None


_NAMED_NEAR_RE = re.compile(
    r"(?:is there (?:any |a )?|any |find |where(?:'s| is) (?:a |the )?)"
    r"([A-Za-z][\w'&.-]{1,24}(?:\s+[A-Za-z][\w'&.-]{1,20}){0,2})"
    r"\s+(?:close to|near|nearby|around|location)\b",
    re.IGNORECASE,
)
_NAMED_NEAR_STOP = {
    "a", "an", "the", "any", "some", "good", "nearby", "nearest", "closest",
    "restaurant", "restaurants", "hotel", "hotels", "place", "places",
    "food", "here", "there", "my", "location", "mine", "me",
    "movie", "theater", "theatre", "theaters", "theatres", "cinema",
    "cinemas", "gas", "station", "stations", "petrol", "pump",
    "hospital", "hospitals", "pharmacy", "cafe", "cafes", "mall",
    "house", "houses", "home", "homes",
}


def extract_nearby_name(text: str) -> str | None:
    """A named shop (KFC) asked for near the user, or None."""
    match = _NAMED_NEAR_RE.search(text or "")
    if not match:
        return None
    name = " ".join(match.group(1).split()).strip(" .?!,")
    name = re.sub(r"^(?:the|a|an)\s+", "", name, flags=re.I).strip()
    if not name or name.lower() in _NAMED_NEAR_STOP:
        return None
    if detect_category(name):
        return None
    return name


def _short_area(r: dict, city: str) -> str:
    addr = r.get("address") or ""
    skip = {city.lower(), "india", "assam"} if city else {"india", "assam"}
    parts = []
    for p in addr.split(","):
        p = p.strip()
        low = p.lower()
        if (
            not p
            or low in skip
            or re.match(r"^\d{6}", p)
            or re.search(r"\bassam\b|\b\d{6}\b", low)
            or re.match(r"^[A-Z0-9]{4,}\+[A-Z0-9]{2,}$", p)
        ):
            continue
        if re.match(r"^\d+(st|nd|rd|th)\s+floor\b", p, re.I):
            continue
        parts.append(p)
    return ", ".join(parts[-2:]) if parts else ""


def _format_place_line(i: int, r: dict, city: str = "") -> str:
    dist_km = r["distance_m"] / 1000
    dist_str = f"{r['distance_m']:.0f} m" if r["distance_m"] < 1000 else f"{dist_km:.1f} km"
    area = _short_area(r, city)
    bit = f"{r['name']}, about {dist_str}"
    if area:
        bit += f", {area}"
    return f"{i}. {bit}"


def _local_query(query: str, loc: dict) -> str:
    city = (loc.get("city") or "").strip()
    if city and city.lower() not in (query or "").lower():
        return f"{query} in {city}"
    return query


def _nearby_only(results: list[dict], max_m: int = 30000) -> list[dict]:
    return [r for r in results if r.get("distance_m", 10**9) <= max_m]


_CITY_TAIL_RE = re.compile(r"\bin\s+([\w'’-]+(?:\s+[\w'’-]+)?)\s*$", re.IGNORECASE)


def search_origin(user_text: str):
    """(loc, remote) — where to search, and whether it is somewhere else.

    describe_weather has resolved named cities since the "Johar" incident,
    but the places path never did: it always searched the current GPS
    position. Measured live, "the best hotels in Guwahati" answered "Near
    you in Jorhat: The Manor, about 517 m" — Jorhat hotels, 300 km from
    the city actually asked about, presented as the answer. The city was
    in the sentence twice and extract_weather_place picks it out fine;
    nothing was consulting it.

    Returns (None, False) when no city was named, meaning use GPS.
    """
    named = extract_weather_place(user_text)
    if not named:
        return None, False
    here = None
    try:
        here = lc.get_location()
    except lc.LocationError:
        pass
    kind, hit = resolve_named_place(named, here)
    if kind != "named" and _CITY_TAIL_RE.search(named or ""):
        # "Hotel Novotel in Guwahati" comes out of the extractor whole,
        # and geocoding the whole thing resolves to nothing — so the city
        # was dropped and the search silently fell back to GPS. The tail
        # after the last "in" is the city; the venue before it is what is
        # being searched FOR, not where.
        kind, hit = resolve_named_place(_CITY_TAIL_RE.search(named).group(1), here)
    if kind != "named" or not hit:
        # "local" means they named where they already are; "unknown"
        # means the geocoder did not recognise it, and searching the
        # wrong continent is exactly what resolve_named_place exists to
        # prevent. Both fall back to GPS.
        return None, False
    return {"lat": hit["lat"], "lon": hit["lon"], "city": hit.get("city") or named}, True


def describe_named_nearby(name: str, language: str | None = None,
                          origin: dict | None = None, remote: bool = False) -> str:
    loc = origin or lc.get_location()
    results = lc.search_places_google(_local_query(name, loc), loc["lat"], loc["lon"])
    if not remote:
        # The distance cap is about "near me". In another city, distance
        # from its centre is not a reason to drop a result.
        results = _nearby_only(results)
    remember_places([r.get("name", "") for r in results])
    return _nearby_speech(loc, results, name, language, remote)


def describe_nearby(category: str, radius_m: int = 3000, language: str | None = None,
                    origin: dict | None = None, remote: bool = False) -> str:
    loc = origin or lc.get_location()
    if category == "movie_theater":
        radius_m = max(radius_m, 12000)
    if remote:
        # A city-wide radius: "hotels in Guwahati" means the city, not
        # three kilometres from whichever point the geocoder returned.
        radius_m = max(radius_m, 15000)
    results = lc.search_nearby(loc["lat"], loc["lon"], category, radius_m=radius_m)
    remember_places([r.get("name", "") for r in results])
    label = CATEGORY_LABELS.get(category, category.replace("_", " "))
    return _nearby_speech(loc, results, label, language, remote)


_AS_LABELS = {
    "restaurants": "ৰেষ্টুৰেণ্ট",
    "hotels": "হোটেল",
    "movie theaters": "চিনেমা হল",
    "gas stations": "পেট্ৰল পাম্প",
    "cafes": "কেফে",
}


def _nearby_speech(loc: dict, results: list[dict], label: str,
                   language: str | None, remote: bool = False) -> str:
    """Spoken list. Data is Google Maps; the words should not sound like a dump."""
    place = (loc.get("city") or "").strip()
    top = results[:4]
    if language == "as-IN":
        city = _as_city(place) if place else "ইয়াত"
        kind = _AS_LABELS.get(label, label)
        if not top:
            return f"{city}ৰ ওচৰত Google Maps-ত {kind} পোৱা নগ'ল।"
        bits = []
        for r in top:
            dist_km = r["distance_m"] / 1000
            dist = f"{r['distance_m']:.0f} মিটাৰ" if r["distance_m"] < 1000 else f"{dist_km:.1f} কিমি"
            area = _short_area(r, place)
            bit = f"{r['name']}, প্ৰায় {dist}"
            if area:
                bit += f", {area}"
            bits.append(bit)
        if len(bits) == 1:
            return f"{city}ৰ ওচৰত {bits[0]}।"
        return f"{city}ৰ ওচৰৰ {kind}: " + "; ".join(bits) + "।"
    if not top:
        where = f" in {place}" if place else " near you"
        return f"Google Maps has no {label}{where}."
    bits = [_format_place_line(i, r, place).split(". ", 1)[-1] for i, r in enumerate(top, start=1)]
    if remote:
        # "Near you" would be a lie about a city 300 km away, and the
        # distances are from that city's centre, not from the user.
        bits = [b.split(", about ")[0] for b in bits]
        return f"In {place}: " + "; ".join(bits) + "."
    where = f" in {place}" if place else " near you"
    if len(bits) == 1:
        return f"Near you{where}: {bits[0]}."
    return f"Near you{where}: " + "; ".join(bits) + "."


def describe_text_search(query: str, language: str | None = None,
                         origin: dict | None = None, remote: bool = False) -> str:
    loc = origin or lc.get_location()
    q = _local_query(query, loc)
    results = lc.search_places_google(q, loc["lat"], loc["lon"])
    if not remote:
        results = _nearby_only(results)
    remember_places([r.get("name", "") for r in results])
    return _nearby_speech(loc, results, query, language, remote)


_DISTANCE_DEST_RE = re.compile(
    r"(?:how far (?:is|to)|how long (?:to|does it take to (?:get to|reach|drive to))|"
    r"distance (?:to|from)|directions to|route to|drive to)\s+(.+?)\s*$",
    re.IGNORECASE,
)


def extract_distance_dest(text: str) -> str | None:
    match = _DISTANCE_DEST_RE.search((text or "").strip().strip("?.!"))
    if not match:
        return None
    dest = match.group(1).strip(" .?!,")
    dest = re.sub(r"\b(from here|from my location|right now|by car|by road)\b", "", dest, flags=re.I)
    dest = dest.strip(" .?!,")
    return dest or None


def describe_distance(destination: str) -> str:
    loc = lc.get_location()
    route = lc.route_to(destination, loc["lat"], loc["lon"])
    here = (loc.get("city") or "here").strip()
    if not route:
        places = lc.search_places_google(destination, loc["lat"], loc["lon"], limit=1)
        if places:
            km = places[0]["distance_m"] / 1000
            return (
                f"{places[0]['name']} is about {km:.1f}km in a straight line from {here}. "
                "I couldn't get a driving route."
            )
        return f"Google Maps has no route from {here} to {destination}."
    km = (route["distance_m"] or 0) / 1000
    mins = round((route["duration_s"] or 0) / 60)
    if mins >= 60:
        hours, mins = divmod(mins, 60)
        eta = f"{hours}h {mins}m" if mins else f"{hours}h"
    else:
        eta = f"{mins} minutes"
    return f"{destination} is {km:.1f}km from {here}, about {eta} by car."


def maps_url(query: str, lat: float | None = None, lon: float | None = None) -> str:
    # query is often empty — "open this in maps"/"open maps" carry no place
    # name of their own (see cli.handle_open_maps, which strips the trigger
    # phrase itself before calling this). Omitting `q` rather than sending
    # the trigger phrase as a literal search string opens a plain map
    # centered on the current location instead of searching for nonsense.
    # Google by default, because every nearby lookup Bella already does goes
    # through Google Maps (location_client._google_nearby / _text_search) —
    # opening Apple Maps meant the map you were shown was a different
    # service from the one that found the place, with different names and
    # sometimes a different location for it. Set maps_provider to "apple"
    # in config.json to go back.
    from bella.config import load_config

    if (load_config().get("maps_provider") or "google").lower() == "apple":
        params = {"q": query} if query else {}
        if lat is not None and lon is not None:
            params["near" if query else "ll"] = f"{lat},{lon}"
        return "https://maps.apple.com/?" + urllib.parse.urlencode(params)

    params = {"api": "1"}
    if query:
        params["query"] = query
        if lat is not None and lon is not None:
            # Biases the search to where they actually are rather than to
            # whatever Google thinks is the most famous match.
            params["query"] = f"{query}"
            params["center"] = f"{lat},{lon}"
        return "https://www.google.com/maps/search/?" + urllib.parse.urlencode(params)
    if lat is not None and lon is not None:
        return f"https://www.google.com/maps/@{lat},{lon},15z"
    return "https://www.google.com/maps"
