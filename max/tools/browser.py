"""Confirmed browser control via Playwright (sync API — matches every
other file in max/tools/, which are all synchronous; bridge.py already
knows how to run sync tool calls via loop.run_in_executor, so no async
plumbing is needed here).

A single persistent Chromium instance, launched lazily on first use (not
at Max startup) and reused across turns — this is "a controlled
instance" the user can see and that remembers logins, not a fresh
incognito window spawned per command and not the user's actual daily
browser. State-changing actions (open_url/click/type_text) are NOT
confirmed here — that gate lives in cli.execute_browser_tool, one layer
up, same separation as max/tools/shell.py (runs unconditionally; the
router/cli layer is what confirms first).
"""
import atexit
import re
from pathlib import Path

from playwright.sync_api import Error as PlaywrightError
from playwright.sync_api import sync_playwright

PROFILE_DIR = Path(__file__).parent.parent / "browser_profile"


class BrowserError(Exception):
    pass


_playwright = None
_context = None


def _get_page():
    global _playwright, _context
    if _context is None:
        PROFILE_DIR.mkdir(exist_ok=True)
        try:
            _playwright = sync_playwright().start()
            # headless=False: this is meant to be a real, visible session
            # the user can glance at and trust, not a hidden background
            # process — the whole point of "controlled instance" over
            # "uncontrolled window" is that it's the one predictable place
            # browser actions happen, not that it's invisible.
            _context = _playwright.chromium.launch_persistent_context(str(PROFILE_DIR), headless=False)
        except PlaywrightError as e:
            raise BrowserError(f"Could not start the browser: {e}") from e
        atexit.register(_shutdown)
    pages = _context.pages
    return pages[0] if pages else _context.new_page()


def _shutdown() -> None:
    global _playwright, _context
    try:
        if _context:
            _context.close()
    except Exception:
        pass
    try:
        if _playwright:
            _playwright.stop()
    except Exception:
        pass


def _clean_target(target: str) -> str:
    # "the Submit button" -> "Submit" — strips words describing the
    # ELEMENT TYPE, not its actual accessible name/label, which is what
    # Playwright's locators match against.
    cleaned = re.sub(r"\b(the|button|link|field|box|input)\b", "", target, flags=re.IGNORECASE).strip()
    return cleaned or target


def _first_unique_match(strategies, target: str):
    for strategy in strategies:
        try:
            locator = strategy()
            count = locator.count()
        except PlaywrightError:
            continue
        if count == 1:
            return locator
    # Zero or multiple matches both refuse rather than guess — clicking
    # the wrong one of several matches is worse than just failing clearly.
    raise BrowserError(f'Could not find exactly one match for "{target}" on the current page.')


def _find_clickable(page, target: str):
    cleaned = _clean_target(target)
    return _first_unique_match(
        [
            lambda: page.get_by_role("button", name=cleaned, exact=False),
            lambda: page.get_by_role("link", name=cleaned, exact=False),
            lambda: page.get_by_text(cleaned, exact=False),
        ],
        target,
    )


def _find_fillable(page, target: str):
    cleaned = _clean_target(target)
    return _first_unique_match(
        [
            lambda: page.get_by_label(cleaned, exact=False),
            lambda: page.get_by_placeholder(cleaned, exact=False),
            lambda: page.get_by_role("textbox", name=cleaned, exact=False),
        ],
        target,
    )


def open_url(url: str) -> str:
    if not re.match(r"^https?://", url, re.IGNORECASE):
        url = "https://" + url
    try:
        page = _get_page()
        page.goto(url, timeout=15000)
        title = page.title()
    except PlaywrightError as e:
        raise BrowserError(f"Could not open {url}: {e}") from e
    return f"Opened {url}" + (f' — "{title}".' if title else ".")


def click(target: str) -> str:
    try:
        page = _get_page()
        locator = _find_clickable(page, target)
        locator.click(timeout=5000)
    except PlaywrightError as e:
        raise BrowserError(f'Could not click "{target}": {e}') from e
    return f'Clicked "{target}" on the current page.'


def type_text(target: str, text: str) -> str:
    try:
        page = _get_page()
        locator = _find_fillable(page, target)
        locator.fill(text, timeout=5000)
    except PlaywrightError as e:
        raise BrowserError(f'Could not type into "{target}": {e}') from e
    return f'Typed "{text}" into "{target}".'


def get_page_text(max_chars: int = 4000) -> str:
    try:
        page = _get_page()
        text = page.inner_text("body")
    except PlaywrightError as e:
        raise BrowserError(f"Could not read the page: {e}") from e
    text = " ".join(text.split())
    if len(text) > max_chars:
        return text[:max_chars] + f"... (truncated, {len(text)} chars total)"
    return text


def get_current_url() -> str:
    try:
        return _get_page().url
    except PlaywrightError as e:
        raise BrowserError(f"Could not read the current URL: {e}") from e
