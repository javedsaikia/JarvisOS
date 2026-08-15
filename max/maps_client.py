"""Business lookup via Apple's MapKit (MKLocalSearch) — the same data the
Maps app uses. Free, no API key, no account.

Exists because OpenStreetMap, which max/location_client.py already
queries for nearby places, almost never carries phone numbers here:
measured 1 of 32 places within 5km of Jorhat. MapKit returned a number for
every one of the six restaurants it found in the same area, so it is the
only workable source for "call <that restaurant>".

MKLocalSearch treats its region as a RANKING HINT, not a filter. Searching
"Mising Kitchen" from Jorhat returned the Guwahati one, 248km away, as its
sole result. Every caller therefore gets a distance and results beyond
MAX_DISTANCE_M are dropped outright — dialling a business in a different
city would be exactly the silent-wrong-answer failure the location work
just fixed.
"""
import math
import threading
import time

import CoreLocation
import Foundation
import MapKit

# Results further than this from the user are discarded rather than
# offered. Generous enough for "the next town over", far short of the
# 248km cross-state match observed live.
MAX_DISTANCE_M = 25_000
_SEARCH_TIMEOUT_SECONDS = 20


class MapsError(Exception):
    pass


def _haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    radius = 6_371_000
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlambda / 2) ** 2
    return 2 * radius * math.asin(math.sqrt(a))


def search_places(
    query: str,
    lat: float,
    lon: float,
    radius_m: int = 10_000,
    limit: int = 5,
    max_distance_m: int = MAX_DISTANCE_M,
) -> list[dict]:
    """Look up businesses by name/category near a point.

    Returns [{name, phone, lat, lon, distance_m}] sorted nearest-first,
    with anything beyond max_distance_m removed. phone may be None.
    """
    query = (query or "").strip()
    if not query:
        return []

    request = MapKit.MKLocalSearchRequest.alloc().init()
    request.setNaturalLanguageQuery_(query)
    request.setRegion_(
        MapKit.MKCoordinateRegionMakeWithDistance(
            CoreLocation.CLLocationCoordinate2DMake(lat, lon), radius_m, radius_m
        )
    )

    finished = threading.Event()
    result: dict = {}

    def completion(response, error):
        result["response"] = response
        result["error"] = error
        finished.set()

    MapKit.MKLocalSearch.alloc().initWithRequest_(request).startWithCompletionHandler_(completion)

    # The completion handler is delivered via the run loop, so this thread
    # has to pump its own rather than just blocking on the Event.
    loop = Foundation.NSRunLoop.currentRunLoop()
    deadline = time.monotonic() + _SEARCH_TIMEOUT_SECONDS
    while not finished.is_set() and time.monotonic() < deadline:
        loop.runMode_beforeDate_(
            Foundation.NSDefaultRunLoopMode,
            Foundation.NSDate.dateWithTimeIntervalSinceNow_(0.05),
        )
    if not finished.is_set():
        raise MapsError("Map lookup timed out.")
    if result.get("error") is not None:
        raise MapsError(str(result["error"].localizedDescription()))

    response = result.get("response")
    if response is None:
        return []

    places: list[dict] = []
    for item in response.mapItems():
        coord = item.placemark().coordinate()
        distance = _haversine_m(lat, lon, coord.latitude, coord.longitude)
        if distance > max_distance_m:
            continue
        phone = item.phoneNumber()
        places.append(
            {
                "name": str(item.name()) if item.name() else "(unnamed)",
                "phone": str(phone) if phone else None,
                "lat": coord.latitude,
                "lon": coord.longitude,
                "distance_m": distance,
            }
        )
    places.sort(key=lambda p: p["distance_m"])
    return places[:limit]


def find_phone(query: str, lat: float, lon: float) -> dict | None:
    """Nearest place matching `query` that actually has a phone number."""
    for place in search_places(query, lat, lon, limit=10):
        if place["phone"]:
            return place
    return None
