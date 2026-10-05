"""Read-only listing discovery for movies, hotels, and flights.

Maxun-style: fetch a public search/listing page, pull structured fields,
return JSON. Listing pages are GET. Never posts a payment, passenger,
OTP, or login form, never follows checkout URLs, never stores booking-site
credentials.

Optional Maxun cloud robots (MAXUN_API_KEY) can replace the HTML extract
when configured. Missing key or a failed robot falls back to DuckDuckGo
HTML search of the official listing URL.
"""
from __future__ import annotations

import html
import json
import re
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone

from bella import env

_USER_AGENT = "Bella/1.0 (personal assistant; +https://github.com/javedsaikia/JarvisOS)"
_TIMEOUT = 12

# Paths that mean checkout, not a listing. Refusing these is the whole
# safety story for this phase.
_UNSAFE_URL = re.compile(
    r"(checkout|payment|paynow|pay-now|/pay\b|upi|otp|cvv|card[-_]?detail|"
    r"billing|wallet|login|signin|sign-in|password|passenger|traveller|"
    r"add[-_]?card|complete[-_]?booking)",
    re.IGNORECASE,
)

_DDG = "https://html.duckduckgo.com/html/"
_DDG_LITE = "https://lite.duckduckgo.com/lite/"

_PRICE = re.compile(
    r"(?:₹|rs\.?|inr|usd|\$|€)\s?[\d,]+(?:\.\d{1,2})?",
    re.IGNORECASE,
)
_TIME = re.compile(r"\b(?:[01]?\d|2[0-3])[:.][0-5]\d\s?(?:am|pm)?\b", re.IGNORECASE)


class DiscoveryError(Exception):
    pass


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def is_safe_listing_url(url: str) -> bool:
    if not url or not url.lower().startswith(("http://", "https://")):
        return False
    return not _UNSAFE_URL.search(url)


def _get(url: str) -> str:
    if not is_safe_listing_url(url):
        raise DiscoveryError("that page is a checkout or login, not a listing")
    req = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=_TIMEOUT) as resp:
            final = resp.geturl() or url
            if not is_safe_listing_url(final):
                raise DiscoveryError("that page redirected to checkout or login")
            return resp.read().decode("utf-8", "replace")
    except TimeoutError as e:
        raise DiscoveryError("listing page timed out") from e
    except urllib.error.URLError as e:
        raise DiscoveryError(f"could not reach listing page ({e})") from e


def _clean(text: str) -> str:
    text = html.unescape(re.sub(r"<[^>]+>", " ", text or ""))
    return re.sub(r"\s+", " ", text).strip()


def _ddg_url(href: str) -> str:
    href = html.unescape(href or "")
    if "uddg=" in href:
        parsed = urllib.parse.parse_qs(urllib.parse.urlparse(href).query)
        href = (parsed.get("uddg") or [href])[0]
    if href.startswith("//"):
        href = "https:" + href
    return href


def _items_from_pairs(links: list[tuple[str, str]], snippets: list[str], limit: int) -> list[dict]:
    items: list[dict] = []
    for i, (href, title_html) in enumerate(links):
        url = _ddg_url(href)
        title = _clean(title_html)
        snippet = _clean(snippets[i] if i < len(snippets) else "")
        if not title or not is_safe_listing_url(url):
            continue
        price = _PRICE.search(snippet)
        items.append({
            "title": title[:160],
            "snippet": snippet[:280],
            "url": url,
            "price": price.group(0) if price else None,
            "times": _TIME.findall(snippet)[:6],
        })
        if len(items) >= limit:
            break
    return items


def ddg_search(query: str, limit: int = 6) -> list[dict]:
    """Public search-result extract. Listing pages, not a booking form."""
    q = urllib.parse.urlencode({"q": query})
    try:
        body = _get(_DDG_LITE + "?" + q)
    except DiscoveryError:
        raise
    links = re.findall(
        r"<a\b[^>]*href=['\"]([^'\"]+)['\"][^>]*class=['\"]result-link['\"][^>]*>(.*?)</a>",
        body,
        re.IGNORECASE | re.DOTALL,
    )
    snippets = re.findall(
        r"<td[^>]*class=['\"]result-snippet['\"][^>]*>(.*?)</td>",
        body,
        re.IGNORECASE | re.DOTALL,
    )
    items = _items_from_pairs(links, snippets, limit)
    if items:
        return items

    # html.duckduckgo.com sometimes serves the same results with different
    # class names. GET only — a query string, not a form submit.
    try:
        body = _get(_DDG + "?" + q)
    except DiscoveryError:
        return []
    links = re.findall(
        r"<a\b[^>]*class=['\"]result__a['\"][^>]*href=['\"]([^'\"]+)['\"][^>]*>(.*?)</a>"
        r"|<a\b[^>]*href=['\"]([^'\"]+)['\"][^>]*class=['\"]result__a['\"][^>]*>(.*?)</a>",
        body,
        re.IGNORECASE | re.DOTALL,
    )
    pairs = []
    for a, b, c, d in links:
        href, title = (a, b) if a else (c, d)
        if href:
            pairs.append((href, title))
    snippets = re.findall(
        r"class=['\"]result__snippet['\"][^>]*>(.*?)</(?:a|td|span)>",
        body,
        re.IGNORECASE | re.DOTALL,
    )
    return _items_from_pairs(pairs, snippets, limit)


def _maxun_extract(kind: str, query: str, listing_url: str) -> list[dict] | None:
    """Optional Maxun robot. None means 'not configured or failed'."""
    keys = env.load_env()
    api_key = (keys.get("MAXUN_API_KEY") or "").strip()
    robot = (keys.get("MAXUN_ROBOT_ID") or "").strip()
    if not api_key or not robot:
        return None
    base = (keys.get("MAXUN_API_URL") or "https://app.maxun.dev").rstrip("/")
    url = f"{base}/api/robots/{urllib.parse.quote(robot)}/run"
    if not is_safe_listing_url(url):
        return None
    body = json.dumps({
        "kind": kind,
        "query": query,
        "url": listing_url,
        "readOnly": True,
    }).encode()
    req = urllib.request.Request(
        url,
        data=body,
        headers={
            "User-Agent": _USER_AGENT,
            "Content-Type": "application/json",
            "x-api-key": api_key,
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=_TIMEOUT) as resp:
            data = json.loads(resp.read().decode("utf-8", "replace"))
    except (TimeoutError, urllib.error.URLError, json.JSONDecodeError, ValueError):
        return None
    rows = data.get("data") or data.get("items") or data.get("result") or []
    if not isinstance(rows, list):
        return None
    out: list[dict] = []
    for row in rows[:8]:
        if not isinstance(row, dict):
            continue
        item_url = str(row.get("url") or row.get("link") or listing_url)
        if not is_safe_listing_url(item_url):
            continue
        out.append({
            "title": str(row.get("title") or row.get("name") or "")[:160],
            "snippet": str(row.get("snippet") or row.get("description") or "")[:280],
            "url": item_url,
            "price": row.get("price"),
            "times": row.get("times") or [],
            "extra": {k: row[k] for k in row if k not in {
                "title", "name", "snippet", "description", "url", "link",
                "price", "times",
            }},
        })
    return out or None


def _official_item(title: str, listing_url: str) -> dict:
    return {
        "title": title,
        "snippet": "Official listing page. Live data — names, times, and prices can change.",
        "url": listing_url,
        "price": None,
        "times": [],
    }


def _pack(kind: str, query: str, listing_url: str, items: list[dict],
          extra: dict | None = None, official_title: str = "") -> dict:
    if listing_url and is_safe_listing_url(listing_url):
        if not any((it.get("url") or "") == listing_url for it in items):
            items = [_official_item(official_title or query, listing_url)] + list(items)
    return {
        "kind": kind,
        "query": query,
        "live": True,
        "disclaimer": "Live web data. Names, times, and prices can change.",
        "fetched_at": _now_iso(),
        "listing_url": listing_url,
        "source": "public listing search",
        "items": items,
        **(extra or {}),
    }


def _search_or_empty(kind: str, query: str, listing_url: str, extra: dict | None = None) -> dict:
    items: list[dict] = []
    try:
        items = _maxun_extract(kind, query, listing_url) or []
    except Exception:
        items = []
    if not items:
        try:
            items = ddg_search(query)
        except DiscoveryError:
            items = []
    return _pack(kind, query, listing_url, items, extra, official_title=query)


def city_slug(city: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", (city or "").lower()).strip("-")
    return slug or "india"


def get_movie_showtimes(city: str, theater_or_query: str = "") -> dict:
    """Public cinema board for a city / named hall. No booking."""
    city = (city or "").strip()
    theater = (theater_or_query or "").strip()
    where = theater or city or "India"
    listing = f"https://in.bookmyshow.com/explore/home/{city_slug(city or where)}"
    query = f"{where} movie showtimes today site:in.bookmyshow.com"
    packed = _search_or_empty(
        "movies", query, listing,
        extra={"city": city, "theater": theater},
    )
    packed["items"] = packed.get("items") or []
    if packed["items"] and packed["items"][0].get("url") == listing:
        packed["items"][0]["title"] = f"BookMyShow — {where}"
    return packed


def search_hotels(city_or_area: str, constraints: str = "") -> dict:
    """Hotel listing search. Prices if the page shows them. No checkout."""
    place = (city_or_area or "").strip()
    extra = (constraints or "").strip()
    listing = (
        "https://www.google.com/travel/hotels/"
        + urllib.parse.quote(place or "India")
    )
    query = f"{place} hotels {extra} listing".strip()
    packed = _search_or_empty(
        "hotels", query, listing,
        extra={"city_or_area": place, "constraints": extra},
    )
    if packed.get("items") and packed["items"][0].get("url") == listing:
        packed["items"][0]["title"] = f"Hotels in {place or 'this area'}"
    return packed


def search_flights(
    origin: str,
    dest: str,
    date: str = "",
    return_date: str = "",
    constraints: str = "",
) -> dict:
    """Flight options for a route. Listing only — no passenger form."""
    origin = (origin or "").strip()
    dest = (dest or "").strip()
    date = (date or "").strip()
    return_date = (return_date or "").strip()
    constraints = (constraints or "").strip()
    q = f"Flights from {origin} to {dest}"
    if date:
        q += f" on {date}"
    if return_date:
        q += f" return {return_date}"
    listing = "https://www.google.com/travel/flights?q=" + urllib.parse.quote(q)
    search = f"{q} {constraints} tickets".strip()
    packed = _search_or_empty(
        "flights", search, listing,
        extra={
            "origin": origin,
            "dest": dest,
            "date": date,
            "return_date": return_date,
            "constraints": constraints,
        },
    )
    if packed.get("items") and packed["items"][0].get("url") == listing:
        label = f"{origin} to {dest}"
        if date:
            label += f" on {date}"
        packed["items"][0]["title"] = label + " — Google Flights"
    return packed
