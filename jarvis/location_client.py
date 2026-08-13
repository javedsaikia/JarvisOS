"""Location, weather, and nearby-places clients — all free, no API key,
stdlib `urllib` only. Same pattern as spotify_client.py/sarvam_client.py:
a thin wrapper around an external REST API, no SDK dependency.

Location: CoreLocationCLI (https://github.com/fulldecent/corelocationcli)
if it's already installed (`shutil.which`) — this project never installs
it automatically, only uses it if present. Falls back to IP-based
geolocation via ipwho.is (HTTPS, no key). CoreLocationCLI's exact CLI
flags are per its documented interface (`-once -format`), not verified
live in this environment since it isn't installed here; the IP fallback
IS verified live and is what actually runs on a fresh setup.

Weather: Open-Meteo (api.open-meteo.com) — verified live, no key.
Nearby places: OpenStreetMap Overpass API — verified live, no key.
"""
import json
import math
import shutil
import subprocess
import urllib.error
import urllib.parse
import urllib.request

OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"
# The main public Overpass instance (overpass-api.de) returned a real
# 504 Gateway Timeout during testing — a known characteristic of the free
# community-run instances under load, not a bug. Falling through a couple
# of known mirrors is a small, dependency-free resilience improvement
# rather than surfacing a transient server hiccup as a hard failure.
OVERPASS_URLS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
    "https://overpass.openstreetmap.ru/api/interpreter",
]
IP_GEOLOCATION_URL = "https://ipwho.is/"
_USER_AGENT = "JarvisOS/1.0 (local personal assistant)"

# Small fixed vocabulary — matches how router.py detects domains via
# keyword patterns, not free-form extraction. "or similar" categories can
# be added here as one-line entries; no code changes needed elsewhere.
CATEGORY_TAGS = {
    "restaurant": ("amenity", "restaurant"),
    "hotel": ("tourism", "hotel"),
    "gas_station": ("amenity", "fuel"),
    "cafe": ("amenity", "cafe"),
    "pharmacy": ("amenity", "pharmacy"),
}

# WMO weather codes -> plain English, per Open-Meteo's documented mapping.
# Only the common ones — an unmapped code still returns a sane default.
_WMO_CONDITIONS = {
    0: "clear sky",
    1: "mainly clear",
    2: "partly cloudy",
    3: "overcast",
    45: "fog",
    48: "depositing rime fog",
    51: "light drizzle",
    53: "moderate drizzle",
    55: "dense drizzle",
    61: "slight rain",
    63: "moderate rain",
    65: "heavy rain",
    71: "slight snow",
    73: "moderate snow",
    75: "heavy snow",
    80: "slight rain showers",
    81: "moderate rain showers",
    82: "violent rain showers",
    95: "thunderstorm",
}


class LocationError(Exception):
    pass


def _corelocation() -> dict | None:
    binary = shutil.which("CoreLocationCLI")
    if not binary:
        return None
    try:
        result = subprocess.run(
            [binary, "-once", "-format", "%latitude,%longitude"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode != 0:
            return None
        lat_str, lon_str = result.stdout.strip().split(",")
        return {"lat": float(lat_str), "lon": float(lon_str), "source": "corelocation"}
    except (subprocess.TimeoutExpired, OSError, ValueError):
        return None


def _ip_location() -> dict | None:
    req = urllib.request.Request(IP_GEOLOCATION_URL, headers={"User-Agent": _USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read())
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError):
        return None
    if not data.get("success", True) or "latitude" not in data:
        return None
    return {"lat": data["latitude"], "lon": data["longitude"], "source": "ip", "city": data.get("city")}


def get_location() -> dict:
    """Prefers CoreLocationCLI (real GPS) when installed, otherwise falls
    back to IP-based approximate location. Raises LocationError if both
    fail (e.g. offline)."""
    loc = _corelocation()
    if loc:
        return loc
    loc = _ip_location()
    if loc:
        return loc
    raise LocationError(
        "Could not determine location — CoreLocationCLI isn't installed "
        "and the IP geolocation service is unreachable (check your internet connection)."
    )


def get_weather(lat: float, lon: float) -> dict:
    params = urllib.parse.urlencode({"latitude": lat, "longitude": lon, "current_weather": "true"})
    url = f"{OPEN_METEO_URL}?{params}"
    try:
        with urllib.request.urlopen(url, timeout=10) as resp:
            data = json.loads(resp.read())
    except (urllib.error.URLError, TimeoutError) as e:
        raise LocationError(f"Could not reach Open-Meteo ({e})") from e

    current = data.get("current_weather")
    if not current:
        raise LocationError("Open-Meteo returned no current weather data.")
    code = current.get("weathercode")
    return {
        "temperature_c": current.get("temperature"),
        "windspeed_kmh": current.get("windspeed"),
        "condition": _WMO_CONDITIONS.get(code, "unknown conditions"),
    }


def _haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371000.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def search_nearby(lat: float, lon: float, category: str, radius_m: int = 3000, limit: int = 5) -> list[dict]:
    tag = CATEGORY_TAGS.get(category)
    if not tag:
        raise LocationError(f"Unknown place category: {category}")
    key, value = tag
    query = (
        f'[out:json][timeout:15];'
        f'(node["{key}"="{value}"](around:{radius_m},{lat},{lon});'
        f'way["{key}"="{value}"](around:{radius_m},{lat},{lon}););'
        f"out center {limit * 4};"
    )
    payload = urllib.parse.urlencode({"data": query}).encode()
    data = None
    last_error: Exception | None = None
    for url in OVERPASS_URLS:
        req = urllib.request.Request(url, data=payload, headers={"User-Agent": _USER_AGENT})
        try:
            with urllib.request.urlopen(req, timeout=20) as resp:
                data = json.loads(resp.read())
            break
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as e:
            last_error = e
            continue
    if data is None:
        raise LocationError(f"Could not reach the places search service ({last_error})")

    results = []
    for el in data.get("elements", []):
        tags = el.get("tags", {})
        name = tags.get("name")
        if not name:
            continue
        el_lat = el.get("lat") or el.get("center", {}).get("lat")
        el_lon = el.get("lon") or el.get("center", {}).get("lon")
        if el_lat is None or el_lon is None:
            continue
        results.append(
            {"name": name, "distance_m": _haversine_m(lat, lon, el_lat, el_lon), "lat": el_lat, "lon": el_lon}
        )
    results.sort(key=lambda r: r["distance_m"])
    return results[:limit]
