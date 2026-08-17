"""The instruction panel: what Max just said, in readable text on screen.

Spoken guidance disappears the moment it is said. Following "Command B,
and there's a bold B in the Home tab" means holding a sentence in your
head while hunting for a button — so the same words are also put on screen,
next to the work, to be re-read at the user's own pace. Voice and text
carry the same instruction; neither is a summary of the other.

Design constraints, all of them from how it is meant to be used:

- **One panel, ever.** A second window stacking on the first is how this
  becomes clutter instead of help. A pidfile makes the process a
  singleton, and new instructions rewrite the existing panel in place.
- **It follows the conversation, not a stopwatch.** It stays while the
  task is live, and goes away on its own when the user switches to a
  different app (the lesson is over), when Max's session ends, or after a
  hard ceiling so nothing can be left stranded on screen forever.
- **Never in the way.** Click-through, non-activating, bottom-right, and
  it never takes focus from the app being taught.

Same separate-process reasoning as max/cursor_overlay.py: AppKit needs the
main thread's run loop, and the voice loop owns its own main thread.

State comes from a JSON file that max/instructions.py writes, polled a few
times a second — the same pattern max/control.py already uses to get a
setting from the web UI into the voice loop, and for the same reason: the
writer and the reader are different processes, and either may restart.

    python3 -m max.instruction_panel
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import AppKit
import Foundation
import objc

STATE_PATH = Path("/tmp/max_instruction.json")
PID_PATH = Path("/tmp/max_instruction_panel.pid")

WIDTH = 460.0
MARGIN = 28.0
PAD = 20.0
POLL_SECONDS = 0.25
# Nothing stays on screen forever, whatever the state file claims — a
# writer that dies mid-lesson must not strand a panel on the desktop.
MAX_LIFETIME_SECONDS = 240.0

ACCENT = (0.15, 0.85, 0.95)


class PanelView(AppKit.NSView):
    def initWithFrame_(self, frame):
        self = objc.super(PanelView, self).initWithFrame_(frame)
        if self is None:
            return None
        self._title = "MAX"
        self._body = ""
        self._alpha = 0.0
        return self

    def setContent_title_(self, body, title):
        self._body = body or ""
        self._title = title or "MAX"
        self.setNeedsDisplay_(True)

    def setAlpha_(self, value):
        self._alpha = max(0.0, min(1.0, value))
        self.setNeedsDisplay_(True)

    # --- text attributes ------------------------------------------------

    @objc.python_method
    def _body_attrs(self):
        style = AppKit.NSMutableParagraphStyle.alloc().init()
        style.setLineBreakMode_(AppKit.NSLineBreakByWordWrapping)
        style.setLineSpacing_(4.0)
        # Glow, not a drop shadow: a soft halo in the accent colour behind
        # each glyph keeps the text legible over a bright document as well
        # as a dark one, which a plain black shadow does not.
        glow = AppKit.NSShadow.alloc().init()
        glow.setShadowBlurRadius_(9.0)
        glow.setShadowOffset_(Foundation.NSMakeSize(0, 0))
        r, g, b = ACCENT
        glow.setShadowColor_(
            AppKit.NSColor.colorWithCalibratedRed_green_blue_alpha_(r, g, b, 0.75 * self._alpha)
        )
        return {
            AppKit.NSFontAttributeName: AppKit.NSFont.systemFontOfSize_(15.0),
            AppKit.NSForegroundColorAttributeName:
                AppKit.NSColor.colorWithCalibratedWhite_alpha_(0.97, self._alpha),
            AppKit.NSParagraphStyleAttributeName: style,
            AppKit.NSShadowAttributeName: glow,
        }

    @objc.python_method
    def _title_attrs(self):
        r, g, b = ACCENT
        glow = AppKit.NSShadow.alloc().init()
        glow.setShadowBlurRadius_(10.0)
        glow.setShadowOffset_(Foundation.NSMakeSize(0, 0))
        glow.setShadowColor_(
            AppKit.NSColor.colorWithCalibratedRed_green_blue_alpha_(r, g, b, 0.9 * self._alpha)
        )
        return {
            AppKit.NSFontAttributeName: AppKit.NSFont.boldSystemFontOfSize_(11.0),
            AppKit.NSForegroundColorAttributeName:
                AppKit.NSColor.colorWithCalibratedRed_green_blue_alpha_(r, g, b, self._alpha),
            AppKit.NSKernAttributeName: 1.8,
            AppKit.NSShadowAttributeName: glow,
        }

    def bodyHeightForWidth_(self, width):
        if not self._body:
            return 0.0
        text = Foundation.NSString.stringWithString_(self._body)
        rect = text.boundingRectWithSize_options_attributes_(
            Foundation.NSMakeSize(width, 10000),
            AppKit.NSStringDrawingUsesLineFragmentOrigin,
            self._body_attrs(),
        )
        return float(rect.size.height)

    def drawRect_(self, rect):
        AppKit.NSColor.clearColor().set()
        AppKit.NSRectFill(rect)
        bounds = self.bounds()

        card = Foundation.NSMakeRect(0, 0, bounds.size.width, bounds.size.height)
        path = AppKit.NSBezierPath.bezierPathWithRoundedRect_xRadius_yRadius_(card, 14.0, 14.0)
        # Transparent black: dark enough to read white text on, sheer
        # enough that the document underneath is still visible — the user
        # is supposed to keep working while reading this.
        AppKit.NSColor.colorWithCalibratedWhite_alpha_(0.04, 0.78 * self._alpha).set()
        path.fill()
        r, g, b = ACCENT
        AppKit.NSColor.colorWithCalibratedRed_green_blue_alpha_(r, g, b, 0.55 * self._alpha).set()
        path.setLineWidth_(1.5)
        path.stroke()

        inner_w = bounds.size.width - PAD * 2
        title = Foundation.NSString.stringWithString_(self._title)
        t_attrs = self._title_attrs()
        t_size = title.sizeWithAttributes_(t_attrs)
        title.drawAtPoint_withAttributes_(
            Foundation.NSMakePoint(PAD, bounds.size.height - PAD - t_size.height), t_attrs
        )

        body_h = self.bodyHeightForWidth_(inner_w)
        body = Foundation.NSString.stringWithString_(self._body)
        body.drawWithRect_options_attributes_(
            Foundation.NSMakeRect(
                PAD,
                bounds.size.height - PAD - t_size.height - 10 - body_h,
                inner_w,
                body_h,
            ),
            AppKit.NSStringDrawingUsesLineFragmentOrigin,
            self._body_attrs(),
        )


class PanelController(Foundation.NSObject):
    def initWithWindow_view_(self, window, view):
        self = objc.super(PanelController, self).init()
        if self is None:
            return None
        self._window = window
        self._view = view
        self._stamp = ""
        self._shown_at = 0.0
        self._for_app = ""
        self._visible = False
        self._fade = 0.0
        self._started = time.monotonic()
        return self

    # --- state ----------------------------------------------------------

    @objc.python_method
    def _read_state(self) -> dict:
        try:
            return json.loads(STATE_PATH.read_text())
        except (OSError, json.JSONDecodeError):
            return {}

    @objc.python_method
    def _frontmost_app(self) -> str:
        try:
            app = AppKit.NSWorkspace.sharedWorkspace().frontmostApplication()
            return str(app.localizedName() or "") if app else ""
        except Exception:
            return ""

    def tick_(self, timer):
        # A hard ceiling on the whole process, independent of the file.
        if time.monotonic() - self._started > MAX_LIFETIME_SECONDS:
            AppKit.NSApp().terminate_(None)
            return

        state = self._read_state()
        wanted = bool(state.get("visible")) and bool(state.get("text"))

        if wanted:
            stamp = str(state.get("stamp") or "")
            if stamp != self._stamp:
                # New instruction: rewrite this panel rather than opening
                # another one.
                self._stamp = stamp
                self._for_app = str(state.get("app") or "")
                self._shown_at = time.monotonic()
                self._layout(str(state.get("text") or ""), str(state.get("title") or "MAX"))
                self._visible = True
            elif self._for_app and self._frontmost_app() not in ("", self._for_app):
                # They left the app this was about — the lesson is over.
                wanted = False
            elif time.monotonic() - self._shown_at > float(
                state.get("seconds") or MAX_LIFETIME_SECONDS
            ):
                wanted = False

        target = 1.0 if wanted else 0.0
        if abs(self._fade - target) > 0.001:
            step = 0.18 if target > self._fade else 0.12
            self._fade += step if target > self._fade else -step
            self._fade = max(0.0, min(1.0, self._fade))
            self._view.setAlpha_(self._fade)
            if self._fade > 0 and not self._window.isVisible():
                self._window.orderFrontRegardless()
            elif self._fade <= 0.001 and self._window.isVisible():
                self._window.orderOut_(None)
                self._visible = False
        elif not wanted and self._visible:
            self._visible = False

    @objc.python_method
    def _layout(self, text: str, title: str) -> None:
        self._view.setContent_title_(text, title)
        inner_w = WIDTH - PAD * 2
        body_h = self._view.bodyHeightForWidth_(inner_w)
        height = PAD * 2 + 14 + 10 + body_h
        screen = AppKit.NSScreen.screens()[0] if AppKit.NSScreen.screens() else None
        visible = screen.visibleFrame() if screen else Foundation.NSMakeRect(0, 0, 1440, 900)
        x = visible.origin.x + visible.size.width - WIDTH - MARGIN
        y = visible.origin.y + MARGIN
        self._window.setFrame_display_(
            Foundation.NSMakeRect(x, y, WIDTH, height), True
        )
        self._view.setFrame_(Foundation.NSMakeRect(0, 0, WIDTH, height))


def _already_running() -> bool:
    try:
        pid = int(PID_PATH.read_text().strip())
    except (OSError, ValueError):
        return False
    if pid == os.getpid():
        return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def main() -> int:
    if _already_running():
        return 0
    PID_PATH.write_text(str(os.getpid()))

    app = AppKit.NSApplication.sharedApplication()
    app.setActivationPolicy_(AppKit.NSApplicationActivationPolicyAccessory)

    window = AppKit.NSWindow.alloc().initWithContentRect_styleMask_backing_defer_(
        Foundation.NSMakeRect(0, 0, WIDTH, 120),
        AppKit.NSWindowStyleMaskBorderless,
        AppKit.NSBackingStoreBuffered,
        False,
    )
    window.setOpaque_(False)
    window.setBackgroundColor_(AppKit.NSColor.clearColor())
    window.setHasShadow_(True)
    window.setLevel_(AppKit.NSScreenSaverWindowLevel)
    window.setIgnoresMouseEvents_(True)
    window.setCollectionBehavior_(
        AppKit.NSWindowCollectionBehaviorCanJoinAllSpaces
        | AppKit.NSWindowCollectionBehaviorFullScreenAuxiliary
        | AppKit.NSWindowCollectionBehaviorStationary
    )

    view = PanelView.alloc().initWithFrame_(Foundation.NSMakeRect(0, 0, WIDTH, 120))
    window.setContentView_(view)

    controller = PanelController.alloc().initWithWindow_view_(window, view)
    timer = Foundation.NSTimer.timerWithTimeInterval_target_selector_userInfo_repeats_(
        POLL_SECONDS, controller, "tick:", None, True
    )
    Foundation.NSRunLoop.currentRunLoop().addTimer_forMode_(
        timer, Foundation.NSRunLoopCommonModes
    )
    try:
        app.run()
    finally:
        try:
            if PID_PATH.read_text().strip() == str(os.getpid()):
                PID_PATH.unlink(missing_ok=True)
        except OSError:
            pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
