"""Max's pointer: a big, obviously-not-yours marker drawn over the desktop.

Why this is a separate process
------------------------------
Drawing on screen needs an NSWindow, and an NSWindow needs an
NSApplication run loop **on the main thread**. The voice loop owns its
main thread and never gives it back, so an overlay hosted in-process
would either never draw or would have to take the thread away from
listening. The same reasoning already keeps spoken audio in `afplay`
rather than in-process (see voice_loop._ack_tone_path): a separate
process cannot disturb the loop that matters.

So this is a small standalone program. It draws a ring at a point,
pulses for a few seconds, and exits. Nothing calls back into Max.

Why it exists at all
--------------------
Max used to "point" by warping the real system mouse. That works, but the
user cannot tell Max's movement from their own hand — the arrow that
lands on the Bold button looks exactly like the arrow they were already
holding. Teaching needs the opposite: a mark that is unmistakably the
teacher's, not the student's. So this is deliberately much larger than a
cursor, coloured, animated, and labelled with what it is pointing at.

Click-through (`setIgnoresMouseEvents_`) and non-activating: it must never
take focus or swallow a click, because the whole point is that the user
then clicks the thing themselves.

    python3 -m max.cursor_overlay --x 640 --y 400 --label "Bold" --seconds 4

Coordinates are global, top-left origin, y down — the same space
Quartz.CGWarpMouseCursorPosition uses, so callers can pass exactly what
they passed to screen_capture.point_cursor().
"""
from __future__ import annotations

import argparse
import math

import AppKit
import Foundation
import objc

# Ring geometry, in points. Sized against a macOS arrow cursor, which is
# roughly 16x24: the outer ring is an order of magnitude bigger so there
# is no chance of mistaking one for the other at a glance.
OUTER_RADIUS = 46.0
INNER_RADIUS = 21.0
CANVAS = 200.0
FPS = 30.0

# Max's accent, matching the HUD's cyan rather than any system colour —
# a system-blue ring reads as a macOS focus ring, which is precisely the
# confusion this is meant to avoid.
ACCENT = (0.15, 0.85, 0.95)


def _screen_height() -> float:
    """Height of the primary display, for CG (y-down) -> NS (y-up)."""
    screens = AppKit.NSScreen.screens()
    if not screens:
        return 0.0
    # The primary screen is the one whose origin is (0, 0); global NS
    # coordinates are measured from its bottom-left.
    primary = screens[0]
    for screen in screens:
        frame = screen.frame()
        if frame.origin.x == 0 and frame.origin.y == 0:
            primary = screen
            break
    return float(primary.frame().size.height)


class MarkerView(AppKit.NSView):
    """The ring itself. Pulses so the eye catches it, fades out at the end."""

    def initWithFrame_label_seconds_(self, frame, label, seconds):
        self = objc.super(MarkerView, self).initWithFrame_(frame)
        if self is None:
            return None
        self._label = label or ""
        self._seconds = float(seconds)
        self._elapsed = 0.0
        return self

    def tick_(self, timer):
        self._elapsed += 1.0 / FPS
        if self._elapsed >= self._seconds:
            AppKit.NSApp().terminate_(None)
            return
        self.setNeedsDisplay_(True)

    def _alpha(self) -> float:
        """Fade in fast, hold, fade out over the last 0.6s."""
        remaining = self._seconds - self._elapsed
        if self._elapsed < 0.15:
            return self._elapsed / 0.15
        if remaining < 0.6:
            return max(0.0, remaining / 0.6)
        return 1.0

    def drawRect_(self, rect):
        AppKit.NSColor.clearColor().set()
        AppKit.NSRectFill(rect)

        bounds = self.bounds()
        cx = bounds.size.width / 2.0
        cy = bounds.size.height / 2.0
        alpha = self._alpha()
        # One pulse per second, kept shallow: a marker that changes size
        # dramatically is harder to aim at than one that just breathes.
        pulse = 1.0 + 0.10 * math.sin(self._elapsed * 2.0 * math.pi)
        r, g, b = ACCENT

        def ring(radius: float, width: float, a: float) -> None:
            path = AppKit.NSBezierPath.bezierPathWithOvalInRect_(
                Foundation.NSMakeRect(cx - radius, cy - radius, radius * 2, radius * 2)
            )
            path.setLineWidth_(width)
            AppKit.NSColor.colorWithCalibratedRed_green_blue_alpha_(r, g, b, a * alpha).set()
            path.stroke()

        # Soft halo, then the two definite rings.
        ring(OUTER_RADIUS * pulse + 7, 10.0, 0.16)
        ring(OUTER_RADIUS * pulse, 4.0, 0.95)
        ring(INNER_RADIUS, 3.0, 0.75)

        # Crosshair ticks pointing at the exact spot, with a gap in the
        # middle so the thing being pointed at stays visible.
        AppKit.NSColor.colorWithCalibratedRed_green_blue_alpha_(r, g, b, 0.9 * alpha).set()
        tick = AppKit.NSBezierPath.bezierPath()
        tick.setLineWidth_(3.0)
        gap, reach = 7.0, INNER_RADIUS - 4
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            tick.moveToPoint_(Foundation.NSMakePoint(cx + dx * gap, cy + dy * gap))
            tick.lineToPoint_(Foundation.NSMakePoint(cx + dx * reach, cy + dy * reach))
        tick.stroke()

        if self._label:
            self._draw_label(cx, cy, alpha)

    def _draw_label(self, cx: float, cy: float, alpha: float) -> None:
        """The control's name, under the ring, so the spoken cue and the
        thing on screen are tied together without the user hunting."""
        text = Foundation.NSString.stringWithString_(self._label)
        attrs = {
            AppKit.NSFontAttributeName: AppKit.NSFont.boldSystemFontOfSize_(15.0),
            AppKit.NSForegroundColorAttributeName:
                AppKit.NSColor.colorWithCalibratedWhite_alpha_(1.0, alpha),
        }
        size = text.sizeWithAttributes_(attrs)
        pad_x, pad_y = 9.0, 5.0
        box = Foundation.NSMakeRect(
            cx - size.width / 2 - pad_x,
            cy - OUTER_RADIUS - size.height - pad_y * 2 - 10,
            size.width + pad_x * 2,
            size.height + pad_y * 2,
        )
        bg = AppKit.NSBezierPath.bezierPathWithRoundedRect_xRadius_yRadius_(box, 7.0, 7.0)
        AppKit.NSColor.colorWithCalibratedWhite_alpha_(0.05, 0.88 * alpha).set()
        bg.fill()
        r, g, b = ACCENT
        AppKit.NSColor.colorWithCalibratedRed_green_blue_alpha_(r, g, b, 0.9 * alpha).set()
        bg.setLineWidth_(1.5)
        bg.stroke()
        text.drawAtPoint_withAttributes_(
            Foundation.NSMakePoint(box.origin.x + pad_x, box.origin.y + pad_y), attrs
        )


def show(x: float, y: float, label: str = "", seconds: float = 4.0) -> int:
    """Draw the marker centred on global CG point (x, y) and block until done."""
    app = AppKit.NSApplication.sharedApplication()
    # Accessory: no Dock icon, no menu bar, and critically no focus steal —
    # the user is meant to keep working in whatever app they are learning.
    app.setActivationPolicy_(AppKit.NSApplicationActivationPolicyAccessory)

    ns_y = _screen_height() - float(y)
    frame = Foundation.NSMakeRect(
        float(x) - CANVAS / 2, ns_y - CANVAS / 2, CANVAS, CANVAS
    )
    window = AppKit.NSWindow.alloc().initWithContentRect_styleMask_backing_defer_(
        frame, AppKit.NSWindowStyleMaskBorderless, AppKit.NSBackingStoreBuffered, False
    )
    window.setOpaque_(False)
    window.setBackgroundColor_(AppKit.NSColor.clearColor())
    window.setHasShadow_(False)
    # Above normal windows and full-screen apps, so it can point at a
    # control in a maximised Word window.
    window.setLevel_(AppKit.NSScreenSaverWindowLevel)
    window.setIgnoresMouseEvents_(True)
    window.setCollectionBehavior_(
        AppKit.NSWindowCollectionBehaviorCanJoinAllSpaces
        | AppKit.NSWindowCollectionBehaviorFullScreenAuxiliary
        | AppKit.NSWindowCollectionBehaviorStationary
    )

    view = MarkerView.alloc().initWithFrame_label_seconds_(
        Foundation.NSMakeRect(0, 0, CANVAS, CANVAS), label, seconds
    )
    window.setContentView_(view)
    # Regardless: makeKeyAndOrderFront would pull focus away from the app
    # the user is being taught.
    window.orderFrontRegardless()

    timer = Foundation.NSTimer.timerWithTimeInterval_target_selector_userInfo_repeats_(
        1.0 / FPS, view, "tick:", None, True
    )
    Foundation.NSRunLoop.currentRunLoop().addTimer_forMode_(
        timer, Foundation.NSRunLoopCommonModes
    )
    app.run()
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Draw Max's pointer on screen")
    parser.add_argument("--x", type=float, required=True, help="global x, top-left origin")
    parser.add_argument("--y", type=float, required=True, help="global y, top-left origin, y down")
    parser.add_argument("--label", default="", help="control name to show under the ring")
    parser.add_argument("--seconds", type=float, default=4.0)
    args = parser.parse_args()
    return show(args.x, args.y, args.label, args.seconds)


if __name__ == "__main__":
    raise SystemExit(main())
