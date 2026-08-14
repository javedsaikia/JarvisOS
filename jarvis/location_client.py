"""Location, weather, and nearby-places clients — all free, no API key,
stdlib `urllib` only. Same pattern as spotify_client.py/sarvam_client.py:
a thin wrapper around an external REST API, no SDK dependency.

Location: CoreLocationCLI (https://github.com/fulldecent/corelocationcli)
if it's installed (`shutil.which`) — this project never installs it
automatically, only uses it if present. Now verified live: it needs
Location Services enabled for it in System Settings, and its real
interface is `--format` (there is no `-once`; printing once is the
default) with `%locality` giving Apple's own reverse-geocoded city name.

Falls back to IP-based geolocation via ipwho.is (HTTPS, no key) only when
GPS is genuinely unavailable. That fallback resolves to the ISP's gateway
city, which measured ~300km off here (Guwahati reported while the user was
in Jorhat), so callers are expected to surface the difference rather than
present it as a definite position — see tools/location._place_phrase.

Weather: Open-Meteo (api.open-meteo.com) — verified live, no key.
Nearby places: OpenStreetMap Overpass API — verified live, no key.
"""
import json
import math
import shutil
import subprocess
import time
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
_USER_AGENT = "Orin/1.0 (local personal assistant)"

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


def _reverse_geocode(lat: float, lon: float) -> str | None:
    """Coordinates -> a place name, via OpenStreetMap Nominatim (free, no
    key, same OSM family already used for nearby places). Only the IP
    provider hands back a city name for free; without this the GPS path —
    the accurate one — could not say where you actually are.
    """
    params = urllib.parse.urlencode(
        {"lat": lat, "lon": lon, "format": "json", "zoom": 10, "addressdetails": 1}
    )
    req = urllib.request.Request(
        f"https://nominatim.openstreetmap.org/reverse?{params}",
        headers={"User-Agent": _USER_AGENT},  # Nominatim rejects requests without one
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read())
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError):
        return None
    address = data.get("address") or {}
    # state_district is checked ahead of county deliberately. For this
    # address OSM returns county "Jorhat West" (an administrative
    # sub-division) but state_district "Jorhat", and the district is what
    # a person actually calls the place they are in. Where a real city or
    # town name exists it still wins, since those come first.
    for key in ("city", "town", "municipality", "village", "state_district", "county", "state"):
        if address.get(key):
            return address[key]
    return None


# A GPS fix stays valid far longer than a weather query takes, so a recent
# one is reused when CoreLocation transiently has nothing. Without this a
# single blip drops the answer back to the ISP's gateway city — the exact
# 300km error this whole path exists to avoid.
_GPS_CACHE: dict | None = None
_GPS_CACHE_AT: float = 0.0
_GPS_CACHE_TTL_SECONDS = 300


def _corelocation() -> dict | None:
    global _GPS_CACHE, _GPS_CACHE_AT
    binary = shutil.which("CoreLocationCLI")
    if not binary:
        return None

    # Pipe-separated, not comma: the tool ignores literal separators in
    # -format and prints space-separated, so the previous "%latitude,%longitude"
    # + split(",") produced a single field, raised ValueError on unpack, and
    # was swallowed — the GPS path silently never worked and every lookup
    # quietly fell through to IP. A pipe also survives place names
    # containing spaces. %locality is Apple's own reverse geocoding, which
    # saves a network round-trip and returns "Jorhat" directly.
    fmt = "%latitude|%longitude|%locality"
    for attempt in range(3):
        try:
            result = subprocess.run(
                [binary, "--format", fmt], capture_output=True, text=True, timeout=15
            )
        except (subprocess.TimeoutExpired, OSError):
            break
        output = result.stdout.strip()
        if result.returncode == 0 and "|" in output:
            parts = output.split("|")
            try:
                lat, lon = float(parts[0]), float(parts[1])
            except ValueError:
                break
            city = parts[2].strip() if len(parts) > 2 else ""
            location = {
                "lat": lat,
                "lon": lon,
                "source": "gps",
                "city": city or _reverse_geocode(lat, lon),
            }
            _GPS_CACHE, _GPS_CACHE_AT = location, time.monotonic()
            return location
        # kCLErrorDomain error 0 (location unknown) — CoreLocation simply
        # has no fix yet and returns immediately rather than waiting.
        # Observed clearing on its own within a couple of seconds.
        if attempt < 2:
            time.sleep(1.5)

    if _GPS_CACHE and time.monotonic() - _GPS_CACHE_AT < _GPS_CACHE_TTL_SECONDS:
        return _GPS_CACHE
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
