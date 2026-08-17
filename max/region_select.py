"""Circle something on screen with a crayon, and Max looks at just that.

"What is this?" about a whole 3456x2234 desktop is a bad question — the
answer is diluted by everything else on it. Pointing at the thing first
makes it a good question, and circling is how a person points at a region
rather than a point.

The stroke is drawn as translucent red powder rather than a clean vector
line, for two reasons. It reads instantly as annotation instead of as UI,
so nobody wonders whether they have broken something. And a soft grainy
trail stays legible over both a white document and a dark photo, where a
thin hairline disappears into one or the other.

Unlike max/cursor_overlay.py this window must RECEIVE mouse events, so it
is not click-through and it does take focus for the few seconds it is up.
That is the trade: while circling, the drag belongs to Max, not to the app
underneath. Escape or a right-click cancels and gives it straight back.

Writes the chosen rectangle to a JSON file and exits, so the caller does
not need a pipe:

    python3 -m max.region_select --out /tmp/max_region.json

    {"cancelled": false, "x": 120.0, "y": 340.0, "w": 480.0, "h": 260.0,
     "points": 143}

Coordinates are global, top-left origin, y down — the same space
screencapture -R and Quartz use, so the caller can crop with them directly.
"""
from __future__ import annotations

import argparse
import json
import math
import random

import AppKit
import Foundation
import objc

# Crayon look. Each sampled point becomes a few soft dabs at slight
# offsets; overlapping dabs build up density the way wax does.
DABS_PER_POINT = 4
DAB_RADIUS = 5.5
DAB_ALPHA = 0.11
JITTER = 3.4
CRAYON_RGB = (0.95, 0.18, 0.18)

# Below this the "circle" was a stray click, not a deliberate selection.
MIN_REGION = 24.0
HINT = "Circle what you want me to look at   ·   Esc to cancel"


class CrayonView(AppKit.NSView):
    def initWithFrame_(self, frame):
        self = objc.super(CrayonView, self).initWithFrame_(frame)
        if self is None:
            return None
        self._dabs = []          # (x, y, radius) in view coords
        self._points = []        # sampled cursor path
        self._drawing = False
        self._done = None        # set to a dict when finished
        return self

    def acceptsFirstResponder(self):
        return True

    def isOpaque(self):
        return False

    # --- input ----------------------------------------------------------

    @objc.python_method
    def _add(self, point):
        """Record a path point and lay down its powder."""
        self._points.append((point.x, point.y))
        # Deterministic jitter: seeded from the point index so redraws
        # reproduce the same grain instead of shimmering every frame.
        rng = random.Random(len(self._points) * 7919)
        for _ in range(DABS_PER_POINT):
            self._dabs.append((
                point.x + rng.uniform(-JITTER, JITTER),
                point.y + rng.uniform(-JITTER, JITTER),
                DAB_RADIUS * rng.uniform(0.55, 1.15),
            ))

    @objc.python_method
    def _interpolate(self, a, b):
        """Fill the gap between two mouse samples so a fast drag still
        leaves a continuous trail rather than a dotted one."""
        dx, dy = b.x - a[0], b.y - a[1]
        distance = math.hypot(dx, dy)
        steps = int(distance // 3)
        for i in range(1, steps):
            t = i / steps
            self._add(Foundation.NSMakePoint(a[0] + dx * t, a[1] + dy * t))

    def mouseDown_(self, event):
        self._drawing = True
        self._dabs = []
        self._points = []
        self._add(self.convertPoint_fromView_(event.locationInWindow(), None))
        self.setNeedsDisplay_(True)

    def mouseDragged_(self, event):
        if not self._drawing:
            return
        point = self.convertPoint_fromView_(event.locationInWindow(), None)
        if self._points:
            self._interpolate(self._points[-1], point)
        self._add(point)
        self.setNeedsDisplay_(True)

    def mouseUp_(self, event):
        self._drawing = False
        self._finish()

    def rightMouseDown_(self, event):
        self._cancel()

    def keyDown_(self, event):
        if event.keyCode() == 53:  # Escape
            self._cancel()

    # --- result ---------------------------------------------------------

    @objc.python_method
    def _cancel(self):
        self._done = {"cancelled": True}
        AppKit.NSApp().stop_(None)
        # stop_ only takes effect on the next event, and there may not be
        # one if the user has let go of the mouse.
        AppKit.NSApp().postEvent_atStart_(_wake_event(), True)

    @objc.python_method
    def _finish(self):
        if len(self._points) < 3:
            self._cancel()
            return
        xs = [p[0] for p in self._points]
        ys = [p[1] for p in self._points]
        x0, x1 = min(xs), max(xs)
        y0, y1 = min(ys), max(ys)
        if (x1 - x0) < MIN_REGION or (y1 - y0) < MIN_REGION:
            self._cancel()
            return
        height = self.bounds().size.height
        # View coords are bottom-left, y up. Callers crop with
        # screencapture -R, which is top-left, y down.
        self._done = {
            "cancelled": False,
            "x": float(x0),
            "y": float(height - y1),
            "w": float(x1 - x0),
            "h": float(y1 - y0),
            "points": len(self._points),
        }
        AppKit.NSApp().stop_(None)
        AppKit.NSApp().postEvent_atStart_(_wake_event(), True)

    # --- drawing --------------------------------------------------------

    def drawRect_(self, rect):
        # A barely-there wash so the user can see the mode is armed
        # without the screen underneath being obscured.
        AppKit.NSColor.colorWithCalibratedWhite_alpha_(0.0, 0.06).set()
        AppKit.NSRectFill(rect)

        r, g, b = CRAYON_RGB
        AppKit.NSColor.colorWithCalibratedRed_green_blue_alpha_(r, g, b, DAB_ALPHA).set()
        for x, y, radius in self._dabs:
            AppKit.NSBezierPath.bezierPathWithOvalInRect_(
                Foundation.NSMakeRect(x - radius, y - radius, radius * 2, radius * 2)
            ).fill()

        if not self._points:
            self._draw_hint()

    @objc.python_method
    def _draw_hint(self):
        bounds = self.bounds()
        text = Foundation.NSString.stringWithString_(HINT)
        attrs = {
            AppKit.NSFontAttributeName: AppKit.NSFont.systemFontOfSize_(15.0),
            AppKit.NSForegroundColorAttributeName:
                AppKit.NSColor.colorWithCalibratedWhite_alpha_(1.0, 0.85),
        }
        size = text.sizeWithAttributes_(attrs)
        pad = 14.0
        box = Foundation.NSMakeRect(
            (bounds.size.width - size.width) / 2 - pad,
            bounds.size.height * 0.09,
            size.width + pad * 2,
            size.height + pad,
        )
        bg = AppKit.NSBezierPath.bezierPathWithRoundedRect_xRadius_yRadius_(box, 9.0, 9.0)
        AppKit.NSColor.colorWithCalibratedWhite_alpha_(0.05, 0.8).set()
        bg.fill()
        text.drawAtPoint_withAttributes_(
            Foundation.NSMakePoint(box.origin.x + pad, box.origin.y + pad / 2), attrs
        )


def _wake_event():
    return AppKit.NSEvent.otherEventWithType_location_modifierFlags_timestamp_windowNumber_context_subtype_data1_data2_(
        AppKit.NSEventTypeApplicationDefined,
        Foundation.NSMakePoint(0, 0), 0, 0, 0, None, 0, 0, 0,
    )


def select(out_path: str) -> int:
    app = AppKit.NSApplication.sharedApplication()
    app.setActivationPolicy_(AppKit.NSApplicationActivationPolicyAccessory)

    screen = AppKit.NSScreen.screens()[0]
    frame = screen.frame()
    window = AppKit.NSWindow.alloc().initWithContentRect_styleMask_backing_defer_(
        frame, AppKit.NSWindowStyleMaskBorderless, AppKit.NSBackingStoreBuffered, False
    )
    window.setOpaque_(False)
    window.setBackgroundColor_(AppKit.NSColor.clearColor())
    window.setLevel_(AppKit.NSScreenSaverWindowLevel)
    window.setIgnoresMouseEvents_(False)
    window.setAcceptsMouseMovedEvents_(True)
    window.setCollectionBehavior_(
        AppKit.NSWindowCollectionBehaviorCanJoinAllSpaces
        | AppKit.NSWindowCollectionBehaviorFullScreenAuxiliary
    )

    view = CrayonView.alloc().initWithFrame_(
        Foundation.NSMakeRect(0, 0, frame.size.width, frame.size.height)
    )
    window.setContentView_(view)
    # Unlike the other overlays this one has to take focus — it cannot
    # receive a drag otherwise. Focus goes back the moment it closes.
    app.activateIgnoringOtherApps_(True)
    window.makeKeyAndOrderFront_(None)
    window.makeFirstResponder_(view)
    app.run()

    result = view._done or {"cancelled": True}
    Foundation.NSString.stringWithString_(json.dumps(result)).writeToFile_atomically_encoding_error_(
        out_path, True, Foundation.NSUTF8StringEncoding, None
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Circle a region of the screen")
    parser.add_argument("--out", required=True, help="where to write the chosen rect as JSON")
    args = parser.parse_args()
    return select(args.out)


if __name__ == "__main__":
    raise SystemExit(main())
