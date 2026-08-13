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


def describe_weather() -> str:
    loc = lc.get_location()
    weather = lc.get_weather(loc["lat"], loc["lon"])
    place = f' in {loc["city"]}' if loc.get("city") else ""
    temp = weather["temperature_c"]
    return f'It\'s currently {temp:.0f}\N{DEGREE SIGN}C with {weather["condition"]}{place}.'


def describe_nearby(category: str, radius_m: int = 3000) -> str:
    if category not in lc.CATEGORY_TAGS:
        known = ", ".join(sorted(lc.CATEGORY_TAGS))
        return f"I don't know how to search for that yet — I can look for: {known}."

    loc = lc.get_location()
    results = lc.search_nearby(loc["lat"], loc["lon"], category, radius_m=radius_m)
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
    params = {"q": query}
    if lat is not None and lon is not None:
        params["near"] = f"{lat},{lon}"
    return "https://maps.apple.com/?" + urllib.parse.urlencode(params)
