---
status: accepted
---

# The follow camera is mouse-driven via out-of-band X11 pointer-warp

## Context & Decision

The viewer's [Follow camera](../../scripts/CONTEXT.md) lets the operator orbit/zoom around the
focused ant. We want **mouse motion to orbit and the scroll wheel to zoom** in the two fixed
states (auto-cycle, manual-follow), replacing the old `Q/E/T/G` orbit + `Z/X` zoom keys.

The blocker: **vlearn exposes no mouse API** (the entire `GymRender` input surface is `is_key_down`,
menu widgets, line shapes, and camera get/`reset_camera`), and it ships as **prebuilt binary wheels**
(`vlearn-0.3.9`, compiled `vlearn_bindings.so`) — there is no source to patch the bindings. VSim's own
left-drag rotates the camera *internally*, and that pan is **not** fully cancelled by our per-frame
`reset_camera` (the view still drifts). So we must both **read** the mouse and **suppress** VSim's
motion from outside vlearn.

Decision — read+eat the mouse **out-of-band via `python-xlib`** (`_MouseInput` in
`transformer_rl/train_utils.py`):

- **Eat motion by pinning the cursor.** While following (and only while the VSim window holds input
  focus), each frame we `query_pointer` → take the pixel delta from window-centre → drive
  azimuth/elevation → `warp_pointer` the cursor **back to centre**. VSim still receives the motion
  but it nets to ~zero, so its pan self-cancels. No grab.
- **Clicks pass through.** Because we never grab, button clicks still reach VSim's ImGui menu — but
  the cursor is pinned while following, so **widgets are only clickable in free-cam** (press `F`).
- **Scroll** (transient X buttons 4/5) is read **read-only via the RECORD extension** on a daemon
  thread (its own `Display`); per-frame polling would drop wheel ticks. RECORD doesn't consume the
  events, which is fine — we own `radius` and overwrite the camera each frame regardless.
- **Focus-gated.** No warp unless the VSim window is focused, so alt-tab never traps the desktop
  cursor. Degrades to a silent no-op if there's no X display / window.
- A `Mouse sens` `UserSlider` scales pixels→radians (set in free-cam, since the cursor is pinned
  while following). The control legend is embedded in the widget labels: `UserText`/`UserSeparator`
  exist in the 0.3.9 bindings but this build's renderer doesn't draw them, and registering one blanks
  the whole menu panel — so keys stay in the labels (vsim still has no working text/label widget).

## Considered Options

- **Patch the vlearn bindings to expose mouse delta + a "suppress built-in camera" flag.** Rejected:
  vlearn is distributed as binary wheels with no source here; this is an upstream request, not a
  local change.
- **`XGrabPointer` active grab + `XAllowEvents(ReplayPointer)` to forward clicks.** Rejected as the
  default: it genuinely eats motion but needs fiddly event-replay plumbing to keep the menu clickable.
  Kept as the fallback if pointer-warp ever leaves visible jitter.
- **`pynput` global listener with `suppress=True`.** Rejected: on Xorg, suppression is all-or-nothing
  (the return-`False`-to-consume trick is Windows/macOS only), so it would freeze clicks to the menu
  widgets too; and it adds a second dependency.
- **`evdev` + `EVIOCGRAB`.** Rejected: an exclusive device grab blacks out the mouse for the *whole
  desktop* (needs `uinput` re-injection to stay usable) and needs `/dev/input` permissions.
- **Drag-to-orbit (pin only while a button is held).** Rejected in favour of hover-orbit
  (continuous pin): hover-orbit eats the pan whether VSim moves on drag or bare motion, at the cost
  of widgets being unclickable while following (covered by the `Q/E` group + `Z/X` env keys + `F` to free the cursor).
- **Keep the `Q/E/T/G/Z/X` keys as a fallback.** Rejected: mouse + scroll fully replace them; the
  discrete selection keys stay (left-hand `Q/E` group, `Z/X` env, `F` free-cam, `C` auto-cycle —
  the only selection input while the cursor is pinned).

## Consequences

- New runtime dependency: `python-xlib` (already needed by the video recorder; now declared in
  `pyproject.toml`). X11-only — on Wayland the `_MouseInput` no-ops and orbit falls back to nothing
  (would need a separate path).
- While following, the menu widgets and the `Mouse sens` slider can't be clicked (cursor pinned);
  set them in free-cam. Selection is via keys.
- The cursor is hidden (XFixes, reference-counted) while following and shown again on entering
  free-cam or losing focus; if XFixes is unavailable it stays visible (orbit still works).
- The viewer starts with `capped_step(True)` (real-time-capped playback, not free-running).
- A gym rebuild (`delete_gym`/`create_gym` — the codesign env resamples bodies each episode) returns
  a **new `GymRender`**, orphaning previously-registered menu items on the dead one. `FollowCamera`
  detects the swap in `update()` (`env.gym_render is not self._panel_render`) and re-registers the
  panel on the live render. Camera control was unaffected because `update()` re-reads `env.gym_render`
  every frame.
</content>
</invoke>
