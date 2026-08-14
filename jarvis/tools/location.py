"""Spoken-friendly formatting over jarvis/location_client.py — same
separation as jarvis/tools/spotify.py wrapping jarvis/spotify_client.py.
Weather and nearby-place lookups are read-only; the confirmed "open in
Apple Maps" action itself lives in cli.py next to the other confirm-gated
tools (execute_browser_tool is the direct precedent), not here.
"""
import urllib.parse

from jarvis import location_client as lc

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
    "hotel": "hotel",
    "hotels": "hotel",
    "motel": "hotel",
    "gas station": "gas_station",
    "gas": "gas_station",
    "petrol": "gas_station",
    "fuel": "gas_station",
    "cafe": "cafe",
    "coffee": "cafe",
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
    return f" near {city}, going by your network location"


def describe_weather() -> str:
    loc = lc.get_location()
    weather = lc.get_weather(loc["lat"], loc["lon"])
    place = _place_phrase(loc)
    temp = weather["temperature_c"]
    return f'It\'s currently {temp:.0f}\N{DEGREE SIGN}C with {weather["condition"]}{place}.'


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


def describe_nearby(category: str, radius_m: int = 3000) -> str:
    if category not in lc.CATEGORY_TAGS:
        known = ", ".join(sorted(lc.CATEGORY_TAGS))
        return f"I don't know how to search for that yet — I can look for: {known}."

    loc = lc.get_location()
    results = lc.search_nearby(loc["lat"], loc["lon"], category, radius_m=radius_m)
    remember_places([r.get("name", "") for r in results])
    label = CATEGORY_LABELS.get(category, category)
    if not results:
        return f"I couldn't find any {label} within {radius_m // 1000}km."

    lines = [f"Nearby {label}:"]
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
