"""Native screen capture and on-device OCR for "what's on my screen?".

Three problems had to be solved for that question to actually work, and
none of them are about the model:

1. **The browser can't see the desktop.** The HUD is a web page, so
   anything it could capture is the tab it lives in. Capture therefore
   happens here, in the Python process, through macOS's own window
   server — the same choice already made for AppleScript and screencapture
   elsewhere in this project.

2. **JARVIS was the thing on screen.** With the HUD open (often
   fullscreen), a plain screenshot is a picture of JARVIS, and the honest
   answer to "what's on my screen" became "a glowing blue orb". Rather
   than minimising the window and hoping, this composites the screen from
   the window list *with the HUD window left out*
   (CGWindowListCreateImageFromArray), so everything else — including
   other windows of the same browser — is captured exactly as it is, with
   no hiding, no delay and no flicker. Minimise/restore remains as the
   fallback for when this path is unavailable.

3. **A small vision model cannot read a 3456x2234 screenshot.** Measured:
   moondream on a real capture returned "a computer screen with a black
   background and white text", because it downsamples to a few hundred
   pixels internally and every UI label turns to mush. macOS's own Vision
   framework does accurate on-device OCR of the same image in ~0.9s for
   free, which turns the question into a text question that the local
   model already handles well. The vision model stays as the path for
   screens that genuinely have little text (a photo, a video).

Everything here is local: no image ever leaves the machine on this path.
"""
import subprocess
import tempfile
import time
from pathlib import Path

# Quartz ships with pyobjc, already a dependency (CoreLocation/MapKit).
# Vision adds on-device OCR from the same family. Both are optional at
# import time: without them this module falls back to `screencapture` and
# no OCR, so a missing framework degrades the feature instead of breaking
# the process.
try:
    import Quartz
    import CoreFoundation

    HAVE_QUARTZ = True
except ImportError:  # pragma: no cover - depends on the install
    HAVE_QUARTZ = False

try:
    import Vision
    import Foundation

    HAVE_VISION = True
except ImportError:  # pragma: no cover - depends on the install
    HAVE_VISION = False


class ScreenCaptureError(Exception):
    pass


class ScreenPermissionError(ScreenCaptureError):
    """Screen Recording permission is missing — user action required."""


PERMISSION_HELP = (
    "macOS is not letting JARVIS see the screen. Grant \"Screen Recording\" to the "
    "app that started JARVIS (Terminal, or whatever you ran ./start_jarvis.sh from) "
    "in System Settings > Privacy & Security > Screen Recording, then restart JARVIS."
)

# Layers above the normal window layer are system chrome (menu bar, Dock,
# notification banners, the screen-sharing indicator). They are captured
# anyway as part of the screen; this constant is only used when deciding
# which windows are real application windows worth naming in the context.
NORMAL_WINDOW_LAYER = 0


def available() -> bool:
    """True when the native (window-excluding) capture path can be used."""
    return HAVE_QUARTZ


def list_windows() -> list[dict]:
    """On-screen windows, newest/frontmost first, as plain dicts."""
    if not HAVE_QUARTZ:
        return []
    options = Quartz.kCGWindowListOptionOnScreenOnly | Quartz.kCGWindowListExcludeDesktopElements
    raw = Quartz.CGWindowListCopyWindowInfo(options, Quartz.kCGNullWindowID) or []
    windows = []
    for w in raw:
        windows.append({
            "id": w.get("kCGWindowNumber"),
            "app": w.get("kCGWindowOwnerName") or "",
            "title": w.get("kCGWindowName") or "",
            "layer": w.get("kCGWindowLayer", 0),
        })
    return windows


def has_permission() -> bool:
    """Heuristic Screen Recording check, done without capturing anything.

    Window *titles* of other applications are privileged in the same way
    the pixels are: without the grant, macOS still lists every window but
    blanks kCGWindowName for anything this process doesn't own. So "no
    other app has a readable title" is a reliable stand-in for "not
    granted", and unlike a trial capture it costs nothing and shows no
    screen-recording indicator.
    """
    if not HAVE_QUARTZ:
        return True  # can't tell; the screencapture path reports its own error
    windows = list_windows()
    if not windows:
        return True
    return any(w["title"] and w["layer"] == NORMAL_WINDOW_LAYER for w in windows)


def _cgimage_to_jpeg(image, quality: float = 0.7) -> bytes:
    data = CoreFoundation.CFDataCreateMutable(None, 0)
    dest = Quartz.CGImageDestinationCreateWithData(data, "public.jpeg", 1, None)
    if dest is None:
        raise ScreenCaptureError("could not create a JPEG encoder")
    Quartz.CGImageDestinationAddImage(
        dest, image, {Quartz.kCGImageDestinationLossyCompressionQuality: quality}
    )
    if not Quartz.CGImageDestinationFinalize(dest):
        raise ScreenCaptureError("JPEG encoding failed")
    return bytes(data)


def _scaled(image, max_width: int):
    """Downscale a CGImage to `max_width`, preserving aspect ratio.

    Only used for the vision-model path — OCR always gets the full-size
    image, because shrinking it is exactly what makes small UI text
    unreadable.
    """
    width = Quartz.CGImageGetWidth(image)
    height = Quartz.CGImageGetHeight(image)
    if width <= max_width:
        return image
    scale = max_width / width
    new_w, new_h = int(width * scale), int(height * scale)
    color_space = Quartz.CGColorSpaceCreateDeviceRGB()
    ctx = Quartz.CGBitmapContextCreate(
        None, new_w, new_h, 8, 0, color_space, Quartz.kCGImageAlphaPremultipliedLast
    )
    if ctx is None:
        return image
    Quartz.CGContextSetInterpolationQuality(ctx, Quartz.kCGInterpolationHigh)
    Quartz.CGContextDrawImage(ctx, Quartz.CGRectMake(0, 0, new_w, new_h), image)
    return Quartz.CGBitmapContextCreateImage(ctx) or image


def _capture_cgimage(exclude_window_ids: set[int]):
    """Composite the screen from the window list, minus `exclude_window_ids`."""
    options = Quartz.kCGWindowListOptionOnScreenOnly | Quartz.kCGWindowListExcludeDesktopElements
    raw = Quartz.CGWindowListCopyWindowInfo(options, Quartz.kCGNullWindowID) or []
    keep = [w["kCGWindowNumber"] for w in raw if w.get("kCGWindowNumber") not in exclude_window_ids]
    if not keep:
        return None
    return Quartz.CGWindowListCreateImageFromArray(
        Quartz.CGRectInfinite, keep, Quartz.kCGWindowImageDefault
    )


def _capture_with_screencapture() -> bytes:
    """Fallback: macOS's screenshot CLI. Captures everything, including the
    HUD — which is why callers hide the HUD around this path."""
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
        path = Path(f.name)
    try:
        result = subprocess.run(
            ["screencapture", "-x", "-t", "jpg", str(path)],
            capture_output=True, text=True, timeout=15,
        )
        if result.returncode != 0 or not path.exists():
            stderr = (result.stderr or "").strip()
            if "could not create image from display" in stderr.lower():
                raise ScreenPermissionError(PERMISSION_HELP)
            raise ScreenCaptureError(f"screencapture failed: {stderr or 'unknown error'}")
        return path.read_bytes()
    finally:
        path.unlink(missing_ok=True)


def capture(hud_marker: str = "", max_width: int = 0) -> tuple[bytes, dict]:
    """Capture the screen as JPEG bytes, plus what was captured.

    `hud_marker` is matched against window titles; every match is left out
    of the composite. That is how the JARVIS HUD keeps itself out of its
    own screenshot without being minimised.

    Returns (jpeg_bytes, meta) where meta records the method used, the
    excluded windows, and the frontmost application — real context that
    costs nothing to collect and makes the eventual answer far more
    specific than pixels alone.
    """
    started = time.monotonic()
    if not has_permission():
        raise ScreenPermissionError(PERMISSION_HELP)

    meta: dict = {
        "method": "screencapture", "excluded": [],
        "frontmost": "", "frontmost_title": "", "windows": [],
    }

    def describe(app_windows: list[dict]) -> None:
        """Window-server facts about what was captured: which app is in
        front and what else is open. Not a guess — this is the same list
        the Dock and Mission Control read."""
        named = [w for w in app_windows if w["title"]]
        meta["windows"] = [{"app": w["app"], "title": w["title"]} for w in named][:12]
        meta["frontmost"] = app_windows[0]["app"] if app_windows else ""
        meta["frontmost_title"] = app_windows[0]["title"] if app_windows else ""

    if HAVE_QUARTZ:
        windows = list_windows()
        app_windows = [w for w in windows if w["layer"] == NORMAL_WINDOW_LAYER]
        describe(app_windows)

        excluded_ids = set()
        if hud_marker:
            for w in windows:
                if hud_marker.lower() in (w["title"] or "").lower():
                    excluded_ids.add(w["id"])
                    meta["excluded"].append(f"{w['app']}: {w['title']}")
            # Everything derived from the window list describes what was
            # actually captured, so the excluded window must drop out of
            # the context too — otherwise the answer cites a window that
            # is deliberately absent from the image.
            describe([w for w in app_windows if w["id"] not in excluded_ids])

        image = _capture_cgimage(excluded_ids)
        if image is not None:
            meta["method"] = "quartz-exclude" if excluded_ids else "quartz"
            meta["width"] = Quartz.CGImageGetWidth(image)
            meta["height"] = Quartz.CGImageGetHeight(image)
            if max_width:
                image = _scaled(image, max_width)
            meta["seconds"] = round(time.monotonic() - started, 2)
            return _cgimage_to_jpeg(image), meta

    jpeg = _capture_with_screencapture()
    meta["seconds"] = round(time.monotonic() - started, 2)
    return jpeg, meta


def ocr(jpeg_bytes: bytes) -> list[dict]:
    """On-device text recognition (macOS Vision). Empty list if unavailable.

    Accurate-level recognition on a full-resolution screen capture takes
    about a second and never leaves the machine — no model to pull, no
    tokens, no network.

    Each entry carries the line's height as a fraction of the screen
    alongside its text. Vision hands back the bounding boxes for free, and
    height is a good proxy for prominence: the heading someone is actually
    reading is drawn larger than the menu bar around it. That is what
    makes a truthful one-line summary possible without asking a model to
    guess which line mattered.
    """
    if not (HAVE_VISION and HAVE_QUARTZ) or not jpeg_bytes:
        return []
    data = Foundation.NSData.dataWithBytes_length_(jpeg_bytes, len(jpeg_bytes))
    source = Quartz.CGImageSourceCreateWithData(data, None)
    if source is None:
        return []
    image = Quartz.CGImageSourceCreateImageAtIndex(source, 0, None)
    if image is None:
        return []

    handler = Vision.VNImageRequestHandler.alloc().initWithCGImage_options_(image, None)
    request = Vision.VNRecognizeTextRequest.alloc().init()
    request.setRecognitionLevel_(Vision.VNRequestTextRecognitionLevelAccurate)
    request.setUsesLanguageCorrection_(True)
    ok, _ = handler.performRequests_error_([request], None)
    if not ok:
        return []
    lines = []
    for observation in request.results() or []:
        candidates = observation.topCandidates_(1)
        if not candidates:
            continue
        text = (candidates[0].string() or "").strip()
        if not text:
            continue
        box = observation.boundingBox()
        lines.append({"text": text, "height": float(box.size.height)})
    return lines


def ocr_text(lines: list[dict]) -> str:
    return "\n".join(line["text"] for line in lines)


def _reads_like_prose(text: str) -> bool:
    """Rough filter for "a human would call this a heading".

    Without it the largest text on screen is regularly a fragment of code
    or a regex from whatever editor is open — true, but useless read back
    out loud. Three or more words, and mostly letters, is enough to tell
    a sentence from a symbol soup.
    """
    words = text.split()
    if len(words) < 3:
        return False
    letters = sum(c.isalpha() or c.isspace() for c in text)
    return letters / len(text) >= 0.8


def prominent_lines(lines: list[dict], count: int = 3, min_chars: int = 12) -> list[str]:
    """The largest text on screen — usually the heading being read.

    Short fragments are skipped: the biggest glyphs on a Mac are often a
    clock or a single-word toolbar label, which say nothing about what the
    screen is showing.
    """
    candidates = [
        l for l in lines
        if len(l["text"]) >= min_chars and _reads_like_prose(l["text"])
    ]
    candidates.sort(key=lambda l: l["height"], reverse=True)
    seen, out = set(), []
    for line in candidates:
        text = line["text"]
        if text.lower() in seen:
            continue
        seen.add(text.lower())
        out.append(text)
        if len(out) == count:
            break
    return out
