// Stop drawing when nobody is looking.
//
// Every canvas on this HUD — the orb, the rings, the telemetry panels, the
// Spotify progress bar — runs its own requestAnimationFrame loop, and none
// of them used to care whether the page was visible. That is fine on an
// idle machine and expensive on this one: measured while Bella was being
// used from Microsoft Word, WindowServer was at 74% and the HUD's WebKit
// GPU process at 69%, with a load average of 7. Speech-to-text on the same
// CPU went from a 0.69s baseline to 5.9s, 7.1s and once 25.2s.
//
// So Bella's own dashboard was a large part of why Bella could not hear. The
// browser throttles rAF for a fully backgrounded tab, but not for a window
// that is merely unfocused or occluded — which is exactly the situation
// this HUD is in whenever it is doing its job and the user is working in
// another app.
//
// paused() is the check each loop makes before doing any work. It keeps
// requesting frames (so it resumes instantly) but does no drawing.
export function paused(): boolean {
  return document.hidden;
}

/** Run `fn` on every frame the page is actually visible for. */
export function onVisibleFrame(fn: () => void): void {
  const loop = () => {
    if (!paused()) fn();
    requestAnimationFrame(loop);
  };
  requestAnimationFrame(loop);
}
