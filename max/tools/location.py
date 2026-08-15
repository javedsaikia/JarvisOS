"""Spoken-friendly formatting over max/location_client.py — same
separation as max/tools/spotify.py wrapping max/spotify_client.py.
Weather and nearby-place lookups are read-only; the confirmed "open in
Apple Maps" action itself lives in cli.py next to the other confirm-gated
tools (execute_browser_tool is the direct precedent), not here.
"""
import re
import urllib.parse

from max import location_client as lc
from max.wit import season

_HERE_RE = re.compile(
    r"\b(here|my location|my city|current location|this city|this town|"
    r"outside|around here|near me|locally)\b",
    re.IGNORECASE,
)
_NAMED_PLACE_RE = re.compile(
    r"\b(?:in|at|for|over)\s+(?!the\b|my\b|this\b|here\b|our\b|your\b)"
    r"([a-zA-Z\u0980-\u09FF][a-zA-Z\u0980-\u09FF\s.'-]{1,40}?)"
    r"(?:\s+(?:today|now|please|right now))?\s*$",
    re.IGNORECASE,
)

CATEGORY_LABELS = {
    "restaurant": "restaurants",
    "hotel": "hotels",
    "gas_station": "gas stations",
    "cafe": "cafes",
    "pharmacy": "pharmacies",
}

# Maps common spoken phrasings onto the fixed category vocabulary in
# location_client.CATEGORY_TAGS — deliberately a plain dict lookup, not
# LLM extraction (see the plan's Context section for why: this is a small
# closed vocabulary, so a keyword match is strictly safer and cheaper).
_CATEGORY_KEYWORDS = {
    "restaurant": "restaurant",
    "restaurants": "restaurant",
    "food": "restaurant",
    "eat": "restaurant",
    "ৰেষ্টুৰেণ্ট": "restaurant",
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
}


def detect_category(text: str) -> str | None:
    lowered = text.lower()
    for keyword, category in _CATEGORY_KEYWORDS.items():
        if keyword in lowered:
            return category
    return None


def _place_phrase(loc: dict) -> str:
    """' in <place>', flagged when the position is only IP-approximate.

    IP geolocation resolves to the ISP's gateway city, which was reporting
    Guwahati while the user was in Jorhat — roughly 300km out, and silently,
    so the wrong weather and wrong nearby places looked authoritative. When
    the fix is not from GPS, say so rather than stating it flatly.
    """
    city = loc.get("city")
    if not city:
        return ""
    if loc.get("source") == "gps":
        return f" in {city}"
    if loc.get("source") == "named":
        return f" in {city}"
    return f" near {city}, going by your network location"


def extract_weather_place(text: str) -> str | None:
    """A city the user named, or None to use current GPS.

    'here' / 'my location' are not cities. Never invent one.
    """
    if not text or _HERE_RE.search(text):
        return None
    match = _NAMED_PLACE_RE.search(text.strip())
    if not match:
        return None
    place = " ".join(match.group(1).split()).strip(" .?!,")
    if not place or _HERE_RE.search(place):
        return None
    return place


def describe_here() -> str:
    """Town name from the live GPS fix. Never invent one."""
    loc = lc.get_location()
    city = (loc.get("city") or "").strip()
    if loc.get("source") == "gps":
        if city:
            return season(
                f"You're in {city}.",
                "That's the GPS talking.",
                "Not a guess.",
            )
        return season(
            "I have a GPS fix, but I couldn't resolve the town name.",
            "Even the map is being coy.",
        )
    if city:
        return season(
            f"Going by your network location, you're near {city}. "
            "That can be the ISP city, not where you actually are.",
            "Blame the internet, not me.",
        )
    return season(
        "I couldn't determine the town you're in.",
        "The planet has gone shy.",
    )


def describe_weather(include_rain: bool = True, user_text: str = "") -> str:
    named = extract_weather_place(user_text)
    loc = lc.get_location()
    if named:
        here = (loc.get("city") or "").strip().lower()
        if named.lower() != here:
            looked = lc.geocode_place(named)
            if looked:
                loc = looked
    weather = lc.get_weather(loc["lat"], loc["lon"])
    place = _place_phrase(loc)
    temp = weather["temperature_c"]
    line = f'It\'s currently {temp:.0f}\N{DEGREE SIGN}C with {weather["condition"]}{place}.'
    if include_rain:
        rain = weather.get("rain_today")
        if rain:
            line += " " + rain
    return line


# Names from the most recent nearby search, so "call them" / "call that
# place" has something concrete to resolve instead of the model guessing.
# Names only: these come from OpenStreetMap, which almost never carries
# phone numbers here (1 of 32 within 5km), so the number itself is looked
# up through Apple Maps at call time by name.
_last_nearby_names: list[str] = []


def remember_places(names: list[str]) -> None:
    global _last_nearby_names
    _last_nearby_names = [n for n in names if n and n != "(unnamed)"]


def last_places() -> list[str]:
    return list(_last_nearby_names)


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
    return hits[0] if hits else None


def describe_nearby(category: str, radius_m: int = 3000) -> str:
    if category not in lc.CATEGORY_TAGS:
        known = ", ".join(sorted(lc.CATEGORY_TAGS))
        return season(
            f"I don't know how to search for that yet — I can look for: {known}.",
            "My map has hobbies, but that isn't one of them.",
        )

    loc = lc.get_location()
    results = lc.search_nearby(loc["lat"], loc["lon"], category, radius_m=radius_m)
    remember_places([r.get("name", "") for r in results])
    label = CATEGORY_LABELS.get(category, category)
    place = (loc.get("city") or "").strip()
    where = f" in {place}" if place else ""
    if not results:
        return season(
            f"I couldn't find any {label}{where} within {radius_m // 1000}km.",
            "The map is empty, not my imagination.",
        )

    lines = [f"Nearby {label}{where}:"]
    for i, r in enumerate(results, start=1):
        dist_km = r["distance_m"] / 1000
        dist_str = f"{r['distance_m']:.0f}m" if r["distance_m"] < 1000 else f"{dist_km:.1f}km"
        lines.append(f"{i}. {r['name']} — {dist_str} away")
    return "\n".join(lines)


def maps_url(query: str, lat: float | None = None, lon: float | None = None) -> str:
    # query is often empty — "open this in maps"/"open maps" carry no place
    # name of their own (see cli.handle_open_maps, which strips the trigger
    # phrase itself before calling this). Omitting `q` rather than sending
    # the trigger phrase as a literal search string opens a plain map
    # centered on the current location instead of searching for nonsense.
    params = {"q": query} if query else {}
    if lat is not None and lon is not None:
        params["near" if query else "ll"] = f"{lat},{lon}"
    return "https://maps.apple.com/?" + urllib.parse.urlencode(params)
