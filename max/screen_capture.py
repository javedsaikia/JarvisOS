"""Native screen capture and on-device OCR for "what's on my screen?".

Three problems had to be solved for that question to actually work, and
none of them are about the model:

1. **The browser can't see the desktop.** The HUD is a web page, so
   anything it could capture is the tab it lives in. Capture therefore
   happens here, in the Python process, through macOS's own window
   server — the same choice already made for AppleScript and screencapture
   elsewhere in this project.

2. **Max was the thing on screen.** With the HUD open (often
   fullscreen), a plain screenshot is a picture of Max, and the honest
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
import json
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
    "macOS is not letting Max see the screen. Grant \"Screen Recording\" to the "
    "app that started Max (Terminal, or whatever you ran ./start_max.sh from) "
    "in System Settings > Privacy & Security > Screen Recording, then restart Max."
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
        bounds = w.get("kCGWindowBounds") or {}
        windows.append({
            "id": w.get("kCGWindowNumber"),
            "app": w.get("kCGWindowOwnerName") or "",
            "title": w.get("kCGWindowName") or "",
            "layer": w.get("kCGWindowLayer", 0),
            "x": float(bounds.get("X", 0)),
            "y": float(bounds.get("Y", 0)),
            "w": float(bounds.get("Width", 0)),
            "h": float(bounds.get("Height", 0)),
        })
    return windows


def frontmost_window_bounds(app: str = "") -> tuple[float, float, float, float] | None:
    """(x, y, w, h) of the frontmost real window, or None.

    For questions about the thing someone is working in, the rest of the
    desktop is noise: asked what to play next, the model opened with "I
    see your chess board on top of some code in VS Code" — accurate, and
    not what was asked. Layer 0 only, so the menu bar and Dock are not
    mistaken for the window.
    """
    candidates = [
        w for w in list_windows()
        if w["layer"] == 0 and w["w"] >= 200 and w["h"] >= 200
        and (not app or w["app"].lower() == app.lower())
    ]
    if not candidates:
        return None
    w = candidates[0]
    return w["x"], w["y"], w["w"], w["h"]


def has_permission() -> bool:
    """True when this process may capture other apps' pixels.

    Prefer the official preflight API (macOS 10.15+). The older title
    heuristic is the fallback: without the grant, other apps' window
    names come back blank.
    """
    if HAVE_QUARTZ:
        try:
            return bool(Quartz.CGPreflightScreenCaptureAccess())
        except AttributeError:
            pass
        windows = list_windows()
        if not windows:
            return True
        return any(w["title"] and w["layer"] == NORMAL_WINDOW_LAYER for w in windows)
    return True


def ensure_permission() -> bool:
    """Ask macOS for Screen Recording if we do not already have it.

    CGRequestScreenCaptureAccess shows the system prompt attached to
    *this* process (Terminal when Max is started with ./start_max.sh).
    Returns True if capture is allowed afterwards.
    """
    if not HAVE_QUARTZ:
        return True
    try:
        if Quartz.CGPreflightScreenCaptureAccess():
            return True
        granted = bool(Quartz.CGRequestScreenCaptureAccess())
        if granted:
            return True
    except AttributeError:
        if has_permission():
            return True
    _open_screen_recording_settings()
    return has_permission()


def _open_screen_recording_settings() -> None:
    for url in (
        "x-apple.systempreferences:com.apple.preference.security?Privacy_ScreenCapture",
        "x-apple.systempreferences:com.apple.settings.PrivacySecurity.extension?Privacy_ScreenCapture",
    ):
        try:
            subprocess.run(["open", url], capture_output=True, timeout=5)
            return
        except (OSError, subprocess.TimeoutExpired):
            continue


def _cursor_position() -> tuple[float, float] | None:
    if not HAVE_QUARTZ:
        return None
    try:
        event = Quartz.CGEventCreate(None)
        loc = Quartz.CGEventGetLocation(event)
        return float(loc.x), float(loc.y)
    except Exception:
        return None


def point_cursor(x: float, y: float) -> None:
    """Put the mouse on (x, y) in global top-left display coordinates."""
    if not HAVE_QUARTZ:
        return
    try:
        Quartz.CGWarpMouseCursorPosition((float(x), float(y)))
        Quartz.CGAssociateMouseAndMouseCursorPosition(True)
    except Exception:
        return


def glide_cursor(x: float, y: float, steps: int = 14, duration: float = 0.45) -> None:
    """Move the pointer in a visible arc so the user can follow it.

    A single warp is too fast to see. This is the 'virtual cursor' that
    walks across the real desktop onto the control being named.
    """
    import time

    dest = (float(x), float(y))
    start = _cursor_position() or dest
    if steps < 2:
        point_cursor(*dest)
        return
    for i in range(1, steps + 1):
        t = i / steps
        # Ease-out so it settles on the target instead of flying past.
        e = 1 - (1 - t) * (1 - t)
        point_cursor(start[0] + (dest[0] - start[0]) * e, start[1] + (dest[1] - start[1]) * e)
        time.sleep(duration / steps)
    point_cursor(*dest)


REGION_PATH = Path("/tmp/max_region.json")


def select_region(timeout: int = 45) -> dict | None:
    """Let the user circle part of the screen. Returns the rect, or None.

    Blocks while they draw — that is the point; the turn is waiting on
    their answer the same way it waits on a spoken yes/no. `None` covers
    both "they cancelled" and "the selector could not run", because the
    caller does the same thing either way: fall back to the whole screen.
    """
    import sys

    REGION_PATH.unlink(missing_ok=True)
    try:
        result = subprocess.run(
            [sys.executable, "-m", "max.region_select", "--out", str(REGION_PATH)],
            cwd=str(Path(__file__).resolve().parent.parent),
            capture_output=True,
            timeout=timeout,
        )
    except (subprocess.TimeoutExpired, OSError) as e:
        print(f"[screen] region select unavailable: {e}", flush=True)
        return None
    if result.returncode != 0:
        print(f"[screen] region select exited {result.returncode}", flush=True)
        return None
    try:
        data = json.loads(REGION_PATH.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(data, dict) or data.get("cancelled"):
        return None
    if not (data.get("w") and data.get("h")):
        return None
    return data


def _downscale_jpeg(path: str, max_width: int) -> bytes:
    """Shrink a JPEG on disk with `sips`, returning the bytes, or b"".

    sips rather than Quartz here because the file already exists on disk —
    this is one subprocess against decoding and re-encoding by hand, and
    it is the same tool the OS uses.
    """
    try:
        result = subprocess.run(
            ["sips", "-Z", str(int(max_width)), path],
            capture_output=True, timeout=15,
        )
        if result.returncode != 0:
            return b""
        return Path(path).read_bytes()
    except (subprocess.TimeoutExpired, OSError):
        return b""


def capture_region(x: float, y: float, w: float, h: float, max_width: int = 0) -> bytes:
    """JPEG of one rectangle of the screen, in global top-left points.

    Uses `screencapture -R`, which takes exactly these coordinates, rather
    than cropping a full capture: no scale-factor arithmetic to get wrong
    on a Retina display, and it grabs a fraction of the pixels.

    `max_width` matters when the result is going to a vision model. On a
    Retina display this returns the full backing-store resolution, so a
    "smaller" crop of one window came out at 370KB against 121KB for the
    whole screen downscaled — and the request timed out twice on a chess
    board that answers in seconds at 1024px.
    """
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
        path = f.name
    try:
        result = subprocess.run(
            ["screencapture", "-x", "-t", "jpg",
             "-R", f"{x:.0f},{y:.0f},{w:.0f},{h:.0f}", path],
            capture_output=True, timeout=20,
        )
        if result.returncode != 0:
            raise ScreenCaptureError(
                (result.stderr or b"").decode(errors="replace").strip()
                or "screencapture failed on that region"
            )
        data = Path(path).read_bytes()
        if not data:
            raise ScreenCaptureError("that region came back empty")
        if max_width:
            data = _downscale_jpeg(path, max_width) or data
        return data
    except subprocess.TimeoutExpired as e:
        raise ScreenCaptureError("screencapture timed out") from e
    finally:
        Path(path).unlink(missing_ok=True)


def show_marker(x: float, y: float, label: str = "", seconds: float = 4.0) -> None:
    """Draw Max's pointer at a global CG point. Returns immediately.

    Spawned rather than drawn in-process: the overlay needs an
    NSApplication run loop on the main thread, which the voice loop owns
    (see max/cursor_overlay.py). Fire-and-forget — a marker that fails to
    appear must never break the spoken answer that goes with it.
    """
    import sys

    try:
        subprocess.Popen(
            [sys.executable, "-m", "max.cursor_overlay",
             "--x", f"{float(x):.1f}", "--y", f"{float(y):.1f}",
             "--label", (label or "")[:40], "--seconds", f"{float(seconds):.1f}"],
            cwd=str(Path(__file__).resolve().parent.parent),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
    except OSError as e:
        print(f"[screen] could not show the pointer: {e}", flush=True)


def ocr_line_point(line: dict, image_width: int, image_height: int) -> tuple[float, float] | None:
    """Global CG (top-left, y-down) centre of an OCR line, or None.

    Split out of point_at_ocr_line so the marker and the real pointer are
    aimed at the same computed spot rather than each doing the maths.
    """
    if not image_width or not image_height:
        return None
    scale = 1.0
    try:
        import AppKit
        screen = AppKit.NSScreen.mainScreen()
        if screen is not None:
            scale = float(screen.backingScaleFactor()) or 1.0
    except Exception:
        pass
    cx = (line.get("x", 0) + line.get("width", 0) / 2) * image_width
    cy_from_bottom = (line.get("y", 0) + line.get("height", 0) / 2) * image_height
    return cx / scale, (image_height - cy_from_bottom) / scale


def point_at_window(window: dict) -> None:
    """Point at the centre of a window from list_windows()."""
    w, h = window.get("w") or 0, window.get("h") or 0
    if w < 40 or h < 40:
        return
    glide_cursor(window["x"] + w / 2, window["y"] + h / 2)


def point_at_ocr_line(
    line: dict, image_width: int, image_height: int, label: str = "", marker: bool = True
) -> tuple[float, float] | None:
    """Point at an OCR line. Vision boxes are bottom-left, 0–1 of the image.

    Moves the real pointer AND draws Max's marker on the same spot: the
    real one so the user can click straight away without hunting, the
    marker so they can tell which movement was Max's. Returns where it
    pointed, or None if the line could not be placed.
    """
    point = ocr_line_point(line, image_width, image_height)
    if point is None:
        return None
    x, y = point
    if marker:
        # Drawn first, so the ring is already on screen as the pointer
        # glides into it rather than appearing after everything stopped.
        show_marker(x, y, label)
    glide_cursor(x, y)
    return point


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
    of the composite. That is how the Max HUD keeps itself out of its
    own screenshot without being minimised.

    Returns (jpeg_bytes, meta) where meta records the method used, the
    excluded windows, and the frontmost application — real context that
    costs nothing to collect and makes the eventual answer far more
    specific than pixels alone.
    """
    started = time.monotonic()
    if not ensure_permission():
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
        meta["windows"] = [
            {"app": w["app"], "title": w["title"],
             "x": w.get("x", 0), "y": w.get("y", 0),
             "w": w.get("w", 0), "h": w.get("h", 0)}
            for w in named
        ][:12]
        # The frontmost APP is app_windows[0]'s owner, but its first window
        # is not necessarily the one being looked at. Measured on Safari:
        # a 66x20 helper window titled "Window" sits in front of the real
        # 1515x996 page titled "Assamese language - Wikipedia", so the
        # title came back as "Window" and — worse — frontmost_bounds became
        # that 66x20 sliver, which is the box the teaching pointer scopes
        # its search to. Take the largest window belonging to the app in
        # front instead; that is the one a person means.
        front = app_windows[0] if app_windows else {}
        app_name = front.get("app") or ""
        same_app = [w for w in app_windows if w.get("app") == app_name]
        real = max(
            same_app,
            key=lambda w: (w.get("w", 0) or 0) * (w.get("h", 0) or 0),
            default=front,
        )
        # Only prefer it when it is actually a window rather than a widget,
        # so a genuinely small frontmost window still describes itself.
        if (real.get("w", 0) or 0) * (real.get("h", 0) or 0) > 40000:
            front = real
        meta["frontmost"] = app_name
        meta["frontmost_title"] = front.get("title") or ""
        meta["frontmost_bounds"] = {
            "x": front.get("x", 0), "y": front.get("y", 0),
            "w": front.get("w", 0), "h": front.get("h", 0),
        }

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
        lines.append({
            "text": text,
            "height": float(box.size.height),
            "x": float(box.origin.x),
            "y": float(box.origin.y),
            "width": float(box.size.width),
        })
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
