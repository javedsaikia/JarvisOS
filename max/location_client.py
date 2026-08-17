"""Location, weather, and nearby-places clients — all free, no API key,
stdlib `urllib` only. Same pattern as spotify_client.py/sarvam_client.py:
a thin wrapper around an external REST API, no SDK dependency.

"Where am I" / local weather / nearby: CoreLocationCLI GPS only.
Never IP geolocation — that is the ISP's gateway city (measured as
Guwahati while the user was in Jorhat). A named city ("weather in
Bangalore") is geocoded separately and does not use GPS.

CoreLocationCLI (https://github.com/fulldecent/corelocationcli) is
looked up by absolute path as well as PATH. LaunchAgents inherit a
tiny PATH (`/usr/bin:/bin:...`) that hides Homebrew, so `shutil.which`
alone made every lookup miss GPS and fall through to the ISP city.

Weather: Composio weathermap/OpenWeather when connected, else Open-Meteo
(api.open-meteo.com) — verified live, no key.
Nearby places: Google Maps via Composio. OpenStreetMap/Overpass used to
back this up and was removed: on the free community instances it answered
with the wrong town often enough that a fallback was worse than no
answer, so search_nearby now returns an empty list rather than guessing.
"""
import difflib
import json
import math
import re
import os
import shutil
import subprocess
import time
import urllib.error
import urllib.parse
import urllib.request

OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"
_USER_AGENT = "Max/1.0 (local personal assistant)"
# Launchd does not include Homebrew. Absolute paths first so GPS still
# works when PATH is `/usr/bin:/bin:/usr/sbin:/sbin`.
_CORELOCATION_CANDIDATES = (
    "/opt/homebrew/bin/CoreLocationCLI",
    "/usr/local/bin/CoreLocationCLI",
)

# WMO weather codes -> plain English, per Open-Meteo's documented mapping.
# Only the common ones — an unmapped code still returns a sane default.
_WMO_CONDITIONS = {
    0: "clear sky",
    1: "mainly clear",
    2: "partly cloudy",
    3: "overcast",
    45: "fog",
    48: "rime fog",
    51: "light drizzle",
    53: "moderate drizzle",
    55: "dense drizzle",
    56: "freezing drizzle",
    57: "dense freezing drizzle",
    61: "slight rain",
    63: "moderate rain",
    65: "heavy rain",
    66: "freezing rain",
    67: "heavy freezing rain",
    71: "slight snow",
    73: "moderate snow",
    75: "heavy snow",
    77: "snow grains",
    80: "slight rain showers",
    81: "moderate rain showers",
    82: "heavy rain showers",
    85: "slight snow showers",
    86: "heavy snow showers",
    95: "thunderstorm",
    96: "thunderstorm with hail",
    99: "thunderstorm with heavy hail",
}


def _condition_for(code) -> str:
    """Never say 'unknown conditions' out loud."""
    try:
        code = int(code)
    except (TypeError, ValueError):
        return "mixed skies"
    if code in _WMO_CONDITIONS:
        return _WMO_CONDITIONS[code]
    if code >= 95:
        return "thunderstorm"
    if code >= 80:
        return "rain showers"
    if code >= 70:
        return "snow"
    if code >= 60:
        return "rain"
    if code >= 50:
        return "drizzle"
    if code >= 40:
        return "fog"
    return "cloudy"


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
# single blip used to drop the answer onto the ISP city.
_GPS_CACHE: dict | None = None
_GPS_CACHE_AT: float = 0.0
_GPS_CACHE_TTL_SECONDS = 300


def _corelocation_binary() -> str | None:
    for path in _CORELOCATION_CANDIDATES:
        if os.path.isfile(path) and os.access(path, os.X_OK):
            return path
    return shutil.which("CoreLocationCLI")


def _corelocation() -> dict | None:
    global _GPS_CACHE, _GPS_CACHE_AT
    binary = _corelocation_binary()
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


def geocode_place(name: str) -> dict | None:
    """Named city → coordinates via Nominatim. Used only when the user
    explicitly asks for weather somewhere else — bare 'what's the weather'
    always uses GPS."""
    query = (name or "").strip()
    if not query:
        return None
    params = urllib.parse.urlencode(
        {"q": query, "format": "json", "limit": 1, "addressdetails": 1}
    )
    req = urllib.request.Request(
        f"https://nominatim.openstreetmap.org/search?{params}",
        headers={"User-Agent": _USER_AGENT},
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            hits = json.loads(resp.read())
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError):
        return None
    if not hits:
        return None
    hit = hits[0]
    try:
        lat, lon = float(hit["lat"]), float(hit["lon"])
    except (KeyError, TypeError, ValueError):
        return None
    address = hit.get("address") or {}
    city = None
    for key in ("city", "town", "municipality", "village", "state_district", "county"):
        if address.get(key):
            city = address[key]
            break
    return {
        "lat": lat,
        "lon": lon,
        "source": "named",
        "city": city or query.title(),
    }


def get_location() -> dict:
    """Current position from GPS only. Never IP / ISP city."""
    loc = _corelocation()
    if loc:
        return loc
    if not _corelocation_binary():
        raise LocationError(
            "No GPS — install CoreLocationCLI and enable Location Services for it."
        )
    raise LocationError(
        "No GPS fix yet. Enable Location Services for CoreLocationCLI "
        "in System Settings → Privacy & Security → Location Services."
    )


def get_weather(
    lat: float | None = None,
    lon: float | None = None,
    forecast_days: int = 7,
    query: str | None = None,
) -> dict:
    """Current conditions plus a short daily forecast.

    Open-Meteo first, Composio's weather toolkits only as a fallback.

    That order used to be reversed, and it cost about six seconds on every
    single weather question. Measured, same coordinates, back to back:

        _weathermap (Composio)   8.38s
        _openweather (Composio)  5.77s
        _open_meteo_weather      2.09s

    Nothing was bought with those seconds. Open-Meteo needs no key, and
    _weathermap was ALREADY calling it internally for the daily forecast
    and the rain line — so the Composio round trip was adding six seconds
    for a temperature and a description that the free call returns anyway.
    The paid-ish paths stay as a fallback so a Open-Meteo outage still
    answers, which is the only thing they were really providing.

    `query` is a city name the user said; lat/lon are used for "here".
    """
    looked = None
    if (lat is None or lon is None) and query:
        looked = geocode_place(query)
        if looked:
            lat, lon = looked["lat"], looked["lon"]

    if lat is not None and lon is not None:
        try:
            data = _open_meteo_weather(lat, lon, forecast_days)
            if looked and looked.get("city"):
                data["city"] = looked["city"]
            return data
        except LocationError as e:
            print(f"[weather] open-meteo failed ({e}) — trying Composio", flush=True)

    wm = _weathermap(lat=lat, lon=lon, query=query)
    if wm:
        return wm
    ow = _openweather(lat=lat, lon=lon, query=query)
    if ow:
        return ow
    if lat is None or lon is None:
        raise LocationError(f"no place {query}" if query else "no location for weather")
    data = _open_meteo_weather(lat, lon, forecast_days)
    if looked and looked.get("city"):
        data["city"] = looked["city"]
    return data


def _open_meteo_weather(lat: float, lon: float, forecast_days: int = 7) -> dict:
    days = max(1, min(int(forecast_days), 7))
    params = urllib.parse.urlencode(
        {
            "latitude": lat,
            "longitude": lon,
            "current_weather": "true",
            "daily": (
                "weathercode,temperature_2m_max,temperature_2m_min,"
                "precipitation_probability_max,precipitation_sum"
            ),
            "forecast_days": days,
            "timezone": "auto",
        }
    )
    url = f"{OPEN_METEO_URL}?{params}"
    try:
        with urllib.request.urlopen(url, timeout=10) as resp:
            data = json.loads(resp.read())
    except (urllib.error.URLError, TimeoutError) as e:
        raise LocationError(f"Could not reach Open-Meteo ({e})") from e

    current = data.get("current_weather")
    if not current:
        raise LocationError("Open-Meteo returned no current weather data.")
    code = current.get("weathercode", current.get("weather_code"))
    daily = data.get("daily") or {}
    rain_today = _rain_line(daily)
    return {
        "temperature_c": current.get("temperature"),
        "windspeed_kmh": current.get("windspeed"),
        "condition": _condition_for(code),
        "rain_today": rain_today,
        "days": _daily_days(daily),
        "source": "open-meteo",
        "city": None,
    }


_OW_DISABLED_UNTIL = 0.0
_OW_DISABLE_SECONDS = 2 * 60


def _kelvin_to_c(temp) -> float | None:
    try:
        value = float(temp)
    except (TypeError, ValueError):
        return None
    return value - 273.15 if value > 200 else value


def _weathermap(lat: float | None = None, lon: float | None = None, query: str | None = None) -> dict | None:
    """Composio toolkit `weathermap` (OpenWeatherMap, no user key)."""
    place = (query or "").strip()
    if not place and lat is not None and lon is not None:
        place = _reverse_geocode(lat, lon) or ""
    if not place:
        return None
    try:
        from max import composio_client
        if not composio_client.configured():
            return None
        session = composio_client.get_session()
        result = session.execute("WEATHERMAP_WEATHER", arguments={"location": place})
    except Exception as e:
        print(f"[weather] weathermap failed: {e}", flush=True)
        return None
    err = getattr(result, "error", None)
    if err:
        print(f"[weather] weathermap: {err}", flush=True)
        return None
    data = getattr(result, "data", None)
    if hasattr(data, "model_dump"):
        data = data.model_dump()
    if not isinstance(data, dict):
        return None
    info = data.get("weather_info") if isinstance(data.get("weather_info"), dict) else data
    main = info.get("main") if isinstance(info.get("main"), dict) else {}
    temp = _kelvin_to_c(main.get("temp"))
    if temp is None:
        return None
    desc = ""
    weather = info.get("weather") or []
    if weather and isinstance(weather[0], dict):
        desc = (weather[0].get("description") or "").strip()
    city = (info.get("name") or place).strip() or place
    coord = info.get("coord") if isinstance(info.get("coord"), dict) else {}
    try:
        wlat = float(coord["lat"]) if coord.get("lat") is not None else lat
        wlon = float(coord["lon"]) if coord.get("lon") is not None else lon
    except (TypeError, ValueError):
        wlat, wlon = lat, lon
    days = []
    rain_today = ""
    if wlat is not None and wlon is not None:
        try:
            extra = _open_meteo_weather(wlat, wlon)
            days = extra.get("days") or []
            rain_today = extra.get("rain_today") or ""
        except LocationError:
            pass
    return {
        "temperature_c": temp,
        "windspeed_kmh": (info.get("wind") or {}).get("speed"),
        "condition": desc or "mixed skies",
        "rain_today": rain_today,
        "days": days,
        "source": "weathermap",
        "city": city,
    }


def _openweather(lat: float | None = None, lon: float | None = None, query: str | None = None) -> dict | None:
    """Composio OpenWeather. None if not connected or the key is rejected."""
    if time.monotonic() < _OW_DISABLED_UNTIL:
        return None
    try:
        from max import composio_client
    except Exception:
        return None
    try:
        if not composio_client.configured():
            return None
        if "openweather_api" not in composio_client.active_accounts():
            return None
        session = composio_client.get_session()
    except Exception as e:
        print(f"[weather] OpenWeather session skipped: {e}", flush=True)
        return None

    loc_args: dict = {"units": "metric"}
    if query:
        loc_args["q"] = query
    elif lat is not None and lon is not None:
        loc_args["lat"] = lat
        loc_args["lon"] = lon
    else:
        return None

    current = _composio_weather_payload(
        session, "OPENWEATHER_API_GET_CURRENT_WEATHER", loc_args
    )
    if not current or "main" not in current:
        return None
    forecast = _composio_weather_payload(
        session, "OPENWEATHER_API_GET5_DAY_FORECAST", loc_args
    )
    try:
        temp = float(current["main"]["temp"])
    except (KeyError, TypeError, ValueError):
        return None
    desc = ""
    weather = current.get("weather") or []
    if weather and isinstance(weather[0], dict):
        desc = (weather[0].get("description") or "").strip()
    city = (current.get("name") or query or "").strip() or None
    days = _openweather_days(forecast) if forecast else []
    rain_today = ""
    if days:
        chance = days[0].get("rain_chance")
        if chance is not None and chance >= 40:
            rain_today = f"Rain is likely today ({chance:.0f}% chance)."
        elif chance is not None:
            rain_today = "No rain expected today."
    return {
        "temperature_c": temp,
        "windspeed_kmh": (current.get("wind") or {}).get("speed"),
        "condition": desc or "mixed skies",
        "rain_today": rain_today,
        "days": days,
        "source": "openweather",
        "city": city,
    }


def _composio_weather_payload(session, slug: str, arguments: dict) -> dict | None:
    try:
        result = session.execute(slug, arguments=arguments)
    except Exception as e:
        print(f"[weather] {slug} failed: {e}", flush=True)
        return None
    err = getattr(result, "error", None)
    if err:
        print(f"[weather] {slug}: {err}", flush=True)
        if "401" in str(err):
            global _OW_DISABLED_UNTIL
            _OW_DISABLED_UNTIL = time.monotonic() + _OW_DISABLE_SECONDS
            print("[weather] OpenWeather key rejected; using fallback for 2 minutes", flush=True)
        return None
    data = getattr(result, "data", None)
    if hasattr(data, "model_dump"):
        data = data.model_dump()
    if isinstance(data, dict):
        inner = data.get("data", data)
        if isinstance(inner, str):
            try:
                inner = json.loads(inner)
            except json.JSONDecodeError:
                return None
        if isinstance(inner, dict) and inner.get("cod") not in (None, 200, "200"):
            print(f"[weather] {slug} rejected: {inner.get('message') or inner.get('cod')}", flush=True)
            return None
        return inner if isinstance(inner, dict) else None
    return None


def _openweather_days(forecast: dict) -> list[dict]:
    items = forecast.get("list") or []
    buckets: dict[str, dict] = {}
    order: list[str] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        ts = item.get("dt")
        try:
            day = time.strftime("%Y-%m-%d", time.gmtime(int(ts)))
        except (TypeError, ValueError, OSError):
            continue
        main = item.get("main") or {}
        try:
            high = float(main.get("temp_max"))
        except (TypeError, ValueError):
            high = None
        try:
            low = float(main.get("temp_min"))
        except (TypeError, ValueError):
            low = None
        desc = ""
        w = item.get("weather") or []
        if w and isinstance(w[0], dict):
            desc = (w[0].get("description") or "").strip()
        try:
            pop = float(item.get("pop")) * 100
        except (TypeError, ValueError):
            pop = None
        if day not in buckets:
            buckets[day] = {
                "date": day,
                "high_c": high,
                "low_c": low,
                "condition": desc or "mixed skies",
                "rain_chance": pop,
            }
            order.append(day)
            continue
        slot = buckets[day]
        if high is not None:
            slot["high_c"] = high if slot["high_c"] is None else max(slot["high_c"], high)
        if low is not None:
            slot["low_c"] = low if slot["low_c"] is None else min(slot["low_c"], low)
        if pop is not None:
            slot["rain_chance"] = pop if slot["rain_chance"] is None else max(slot["rain_chance"], pop)
        if desc:
            slot["condition"] = desc
    return [buckets[d] for d in order]


def _daily_days(daily: dict) -> list[dict]:
    times = daily.get("time") or []
    highs = daily.get("temperature_2m_max") or []
    lows = daily.get("temperature_2m_min") or []
    codes = daily.get("weathercode") or daily.get("weather_code") or []
    probs = daily.get("precipitation_probability_max") or []
    out = []
    for i, day in enumerate(times):
        code = codes[i] if i < len(codes) else None
        try:
            high = float(highs[i]) if i < len(highs) and highs[i] is not None else None
        except (TypeError, ValueError):
            high = None
        try:
            low = float(lows[i]) if i < len(lows) and lows[i] is not None else None
        except (TypeError, ValueError):
            low = None
        try:
            prob = float(probs[i]) if i < len(probs) and probs[i] is not None else None
        except (TypeError, ValueError):
            prob = None
        out.append({
            "date": day,
            "high_c": high,
            "low_c": low,
            "condition": _condition_for(code),
            "rain_chance": prob,
        })
    return out


def _rain_line(daily: dict) -> str:
    """One spoken sentence about today's rain, or empty if unknown."""
    probs = daily.get("precipitation_probability_max") or []
    amounts = daily.get("precipitation_sum") or []
    try:
        prob = float(probs[0]) if probs else None
    except (TypeError, ValueError):
        prob = None
    try:
        amount = float(amounts[0]) if amounts else None
    except (TypeError, ValueError):
        amount = None
    if prob is None and amount is None:
        return ""
    if (prob is not None and prob >= 40) or (amount is not None and amount > 0.2):
        if prob is not None:
            return f"Rain is likely today ({prob:.0f}% chance)."
        return "Some rain is expected today."
    return "No rain expected today."


def _haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371000.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def search_nearby(lat: float, lon: float, category: str, radius_m: int = 3000, limit: int = 5) -> list[dict]:
    google = _google_nearby(lat, lon, category, radius_m=radius_m, limit=limit)
    if google:
        return google
    label = category.replace("_", " ")
    text = _google_text_search(
        f"{label} near me", lat, lon, radius_m=max(radius_m, 8000), limit=limit
    )
    if text:
        return text
    # Google only. OSM used to fill gaps with the wrong town.
    return []


_GOOGLE_TYPES = {
    "restaurant": "restaurant",
    "hotel": "hotel",
    "cafe": "cafe",
    "pharmacy": "pharmacy",
    "gas_station": "gas_station",
    "movie_theater": "movie_theater",
    "hospital": "hospital",
    "atm": "atm",
    "supermarket": "supermarket",
    "bank": "bank",
    "parking": "parking",
    "park": "park",
    "shopping_mall": "shopping_mall",
}

_PLACE_FIELDS = (
    "places.id,places.displayName,places.formattedAddress,places.rating,"
    "places.userRatingCount,places.nationalPhoneNumber,places.location,"
    "places.googleMapsUri,places.businessStatus,places.primaryType"
)


def _google_nearby(lat: float, lon: float, category: str, radius_m: int = 3000, limit: int = 5) -> list[dict]:
    """Composio Google Maps nearby. Empty list if not connected or the call fails."""
    gtype = _GOOGLE_TYPES.get(category)
    if not gtype:
        return []
    try:
        from max import composio_client
        if not composio_client.configured():
            return []
        if "google_maps" not in composio_client.active_accounts():
            return []
        session = composio_client.get_session()
        result = session.execute(
            "GOOGLE_MAPS_NEARBY_SEARCH",
            arguments={
                "latitude": lat,
                "longitude": lon,
                "radius": int(radius_m),
                "includedTypes": [gtype],
                "maxResultCount": max(1, min(int(limit), 10)),
                "fieldMask": _PLACE_FIELDS,
            },
        )
    except Exception as e:
        print(f"[maps] google nearby failed: {e}", flush=True)
        return []
    if getattr(result, "error", None):
        print(f"[maps] google nearby: {result.error}", flush=True)
        return []
    data = getattr(result, "data", None)
    if hasattr(data, "model_dump"):
        data = data.model_dump()
    if not isinstance(data, dict):
        return []
    places = data.get("places") or []
    out = []
    for item in places:
        if not isinstance(item, dict):
            continue
        name = item.get("displayName")
        if isinstance(name, dict):
            name = name.get("text")
        name = (name or "").strip()
        if not name:
            continue
        loc = item.get("location") or {}
        try:
            plat, plon = float(loc["latitude"]), float(loc["longitude"])
        except (KeyError, TypeError, ValueError):
            continue
        row = _place_row(item, lat, lon, plat, plon)
        ptype = (item.get("primaryType") or "").lower()
        if category == "restaurant" and ptype in ("hotel", "lodging", "motel"):
            continue
        out.append(row)
    out.sort(key=lambda r: r["distance_m"])
    return out[:limit]


def _place_row(item: dict, lat: float, lon: float, plat: float, plon: float) -> dict:
    name = item.get("displayName")
    if isinstance(name, dict):
        name = name.get("text")
    return {
        "name": (name or "").strip(),
        "distance_m": _haversine_m(lat, lon, plat, plon),
        "lat": plat,
        "lon": plon,
        "phone": item.get("nationalPhoneNumber"),
        "rating": item.get("rating"),
        "rating_count": item.get("userRatingCount"),
        "address": item.get("formattedAddress"),
        "status": item.get("businessStatus"),
        "place_id": item.get("id"),
        "maps_url": item.get("googleMapsUri"),
        "source": "google_maps",
    }


def _maps_session():
    from max import composio_client
    if not composio_client.configured():
        return None
    if "google_maps" not in composio_client.active_accounts():
        return None
    return composio_client.get_session()


def _parse_places(data, lat: float, lon: float, limit: int) -> list[dict]:
    if hasattr(data, "model_dump"):
        data = data.model_dump()
    if not isinstance(data, dict):
        return []
    places = data.get("places") or (data.get("data") or {}).get("places") or []
    out = []
    for item in places:
        if not isinstance(item, dict):
            continue
        loc = item.get("location") or {}
        try:
            plat, plon = float(loc["latitude"]), float(loc["longitude"])
        except (KeyError, TypeError, ValueError):
            continue
        row = _place_row(item, lat, lon, plat, plon)
        if row["name"]:
            out.append(row)
    out.sort(key=lambda r: r["distance_m"])
    return out[:limit]


def _google_text_search(query: str, lat: float, lon: float, radius_m: int = 8000, limit: int = 5) -> list[dict]:
    """Composio GOOGLE_MAPS_TEXT_SEARCH, biased to the GPS fix."""
    q = (query or "").strip()
    if not q:
        return []
    try:
        session = _maps_session()
        if session is None:
            return []
        result = session.execute(
            "GOOGLE_MAPS_TEXT_SEARCH",
            arguments={
                "textQuery": q,
                "latitude": lat,
                "longitude": lon,
                "radius": int(radius_m),
                "maxResultCount": max(1, min(int(limit), 10)),
                "fieldMask": _PLACE_FIELDS,
            },
        )
    except Exception as e:
        print(f"[maps] google text search failed: {e}", flush=True)
        return []
    if getattr(result, "error", None):
        print(f"[maps] google text search: {result.error}", flush=True)
        return []
    return _parse_places(getattr(result, "data", None), lat, lon, limit)


def search_places_google(query: str, lat: float, lon: float, radius_m: int = 12000, limit: int = 5) -> list[dict]:
    return _google_text_search(query, lat, lon, radius_m=radius_m, limit=limit)


_GENERIC_PLACE_WORDS = {
    "cafe", "cafes", "coffee", "hotel", "hotels", "restaurant", "restaurants",
    "bar", "bars", "pizza", "kitchen", "house", "store", "shop", "mall",
    "cinema", "theatre", "theater", "multiplex", "hospital", "clinic",
    "pharmacy", "bank", "atm", "station", "petrol", "the", "and", "for",
}


def name_matches(query: str, name: str) -> bool:
    """Is `name` plausibly the place that was asked for?

    Google Text Search always returns something. Asked for a business
    that does not exist ("Zzzqqq Cafe") it answered "Barbeque : The Smoke
    House" — a real place, a real phone number, and completely the wrong
    answer, delivered with no hint that it was a guess. Requiring the
    result to actually resemble the request is what turns that into an
    honest "I couldn't find it".
    """
    q = re.sub(r"[^a-z0-9 ]", " ", (query or "").lower())
    n = re.sub(r"[^a-z0-9 ]", " ", (name or "").lower())
    # Generic venue words match everything. "Zzzqqq Cafe" was accepted as
    # "Atomix Cafe | Rooftop Cafe in Jorhat" purely on the word "cafe" —
    # a real business, a real number, still the wrong answer. Only a
    # DISTINCTIVE token counts as a match.
    tokens = [t for t in q.split() if len(t) >= 3 and t not in _GENERIC_PLACE_WORDS]
    if not tokens:
        # Nothing distinctive was said (e.g. just "the cafe"), so fall back
        # to whole-string similarity rather than matching on a noun.
        return difflib.SequenceMatcher(None, q.strip(), n.strip()).ratio() >= 0.6
    if any(t in n for t in tokens):
        return True
    return difflib.SequenceMatcher(None, q.strip(), n.strip()).ratio() >= 0.6


# Same cap max/maps_client.py applies to MapKit results, for the same
# reason: dialling a business in another country is unrecoverable.
MAX_PHONE_DISTANCE_M = 25_000


def find_phone(query: str, lat: float, lon: float, city: str = "",
               limit: int = 8) -> dict | None:
    """Nearest place matching `query` that has a phone number, via Google.

    Google Maps is already the source for every nearby search Max does,
    and _PLACE_FIELDS has been asking for nationalPhoneNumber all along —
    the number was being fetched and thrown away while a second provider
    (Apple MapKit, max/maps_client.py) was queried for the same thing.
    One provider means one set of names, one distance, and one answer.
    MapKit stays as the fallback in cli.handle_call for when Google has
    the place but not its number.
    """
    # The town name goes IN THE QUERY. Measured: GOOGLE_MAPS_TEXT_SEARCH
    # ignores the lat/lon bias completely — a bare "KFC" from Jorhat came
    # back with stores in Alberta, Ontario, Michigan and New York, the
    # nearest 10,925 km away, while "KFC Jorhat" returned the local branch
    # at 2.3 km with its real number. The coordinates are not enough.
    wanted = (query or "").strip()
    town = (city or "").strip()
    if town and town.lower() not in wanted.lower():
        wanted = f"{wanted} {town}"
    for place in _google_text_search(wanted, lat, lon, radius_m=MAX_PHONE_DISTANCE_M, limit=limit):
        # The radius is a BIAS, not a filter — Google returns matches
        # outside it and simply ranks them lower, the same trap
        # max/maps_client.py documents for MapKit's region. Without this
        # check "call the KFC" offered a number 10,925 km away (a 403
        # Alberta line) and the confirmation prompt asked whether to dial
        # it. A wrong-continent phone call is not a recoverable mistake.
        if (place.get("phone")
                and place.get("distance_m", 10**9) <= MAX_PHONE_DISTANCE_M
                and name_matches(query, place.get("name", ""))):
            return place
    return None


def route_to(destination: str, lat: float, lon: float) -> dict | None:
    """Driving distance/time via GOOGLE_MAPS_GET_ROUTE."""
    dest = (destination or "").strip()
    if not dest:
        return None
    try:
        session = _maps_session()
        if session is None:
            return None
        result = session.execute(
            "GOOGLE_MAPS_GET_ROUTE",
            arguments={
                "origin_address": f"{lat},{lon}",
                "destination_address": dest,
                "units": "METRIC",
            },
        )
    except Exception as e:
        print(f"[maps] google route failed: {e}", flush=True)
        return None
    if getattr(result, "error", None):
        print(f"[maps] google route: {result.error}", flush=True)
        return None
    data = getattr(result, "data", None)
    if hasattr(data, "model_dump"):
        data = data.model_dump()
    if not isinstance(data, dict):
        return None
    inner = data.get("response_data") or data.get("data") or data
    routes = inner.get("routes") or data.get("routes") or []
    if not routes:
        return None
    route = routes[0] if isinstance(routes[0], dict) else {}
    meters = route.get("distanceMeters") or route.get("distance_meters")
    duration = route.get("duration") or ""
    seconds = None
    if isinstance(duration, str) and duration.endswith("s"):
        try:
            seconds = int(float(duration[:-1]))
        except ValueError:
            seconds = None
    elif isinstance(duration, (int, float)):
        seconds = int(duration)
    try:
        meters = int(meters) if meters is not None else None
    except (TypeError, ValueError):
        meters = None
    if meters is None and seconds is None:
        return None
    return {"distance_m": meters, "duration_s": seconds, "destination": dest}
