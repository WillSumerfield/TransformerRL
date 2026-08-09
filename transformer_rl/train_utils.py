"""Shared boilerplate for rl_games-based ant training scripts."""
from __future__ import annotations

import math
import os
import random
import sys
import threading
from datetime import datetime
from pathlib import Path

import yaml

from . import runtime
from .morphology import seed_body

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_VIDEOS_DIR = _PROJECT_ROOT / "videos"

# Camera position relative to the followed robot (world axes, Y up). The look
# direction is auto-derived to point back at the robot.
CAMERA_OFFSET = (2.5, 2.5, 2.5)

# Give the SDL render window a stable, unique WM_CLASS so _find_xwindow can
# target it. Without this SDL defaults WM_CLASS to the interpreter name
# ("python3.11"), which matches nothing and makes the recorder grab the wrong
# window (e.g. the editor). Must be set before vlearn creates the window.
os.environ.setdefault("SDL_VIDEO_X11_WMCLASS", "vsim_render")

# Title/class fragments used to identify the vsim render window.
_VSIM_WINDOW_TITLES = ["vsim_render", "vsim", "vlearn"]


class _PlayLimiter:
    """Wraps env; stops play after max_steps total steps."""

    def __init__(self, env, max_steps: int):
        self._env = env
        self._max = max_steps
        self._count = 0

    def __getattr__(self, name):
        return getattr(self._env, name)

    def step(self, actions):
        result = self._env.step(actions)
        self._count += 1
        if self._count >= self._max:
            self._env.render_finished = True
        return result


def _str_to_bool(s: str) -> bool:
    return s.lower() == "true"


def _deep_merge(base: dict, over: dict) -> dict:
    """Recursive dict merge; `over` wins on conflict. Returns a new dict."""
    out = dict(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def _load_config(path: Path) -> dict:
    """Load a config, resolving its `extends:` parent chain (parent merged UNDER child).

    Lets a family of configs (e.g. the codesign arms) share ONE copy of the blocks they don't
    vary -- an arm that never names `generator:` cannot silently drift from its siblings' copy.
    """
    with open(path) as f:
        cfg = yaml.safe_load(f)
    parent = cfg.pop("extends", None)
    return cfg if parent is None else _deep_merge(_load_config(path.parent / parent), cfg)


def _resolve_task(config: dict, default: str | None = None) -> tuple[str, type]:
    """(registry key, Task class) from the config's `env.task`.

    The task is a CONFIG fact: an entry point names an algorithm and never a task, so this is the
    only place either entry point learns which Task it is running. Resolved through the package's
    registry exactly as `module_library` is -- no class argument, no import path, because the key
    also has to mean the same thing in a stamped config.yaml and in a checkpoint.

    `default` is for reading configs stamped into run dirs BEFORE `env.task` existed. Every such run
    is an ant run -- ant was the only task -- so "ant" is a fact about those stamps, not a guess.
    Never pass a default when reading a config a human wrote: there, a missing task is an error.
    """
    from codesigner.components.tasks import REGISTRY
    key = config.get("env", {}).get("task", default)
    if key not in REGISTRY:
        raise SystemExit(f"env.task must be one of {sorted(REGISTRY)}; got {key!r}")
    return key, REGISTRY[key]


def _compose_identity(task: str, family: str | None, experiment: str) -> dict:
    """The identity fields base.yaml leaves out, composed from the config's task and the algorithm
    the entry point names.

    `family` groups an algorithm's runs one level above the experiment dir; None puts the experiment
    straight under `runs/<task>`. Composition -- rather than a per-script constant -- is what lets
    the same script write `runs/ant_codesign/...` and `runs/grasp_codesign/...`.
    """
    root = f"runs/{task}_{family}" if family else f"runs/{task}"
    return {
        "env_name": f"{task}-env",
        "name": f"{task}_{experiment}",
        "train_dir": f"{root}/{experiment}",
    }


def _adjust_minibatch(cfg: dict, n_envs: int, h_len: int) -> None:
    requested = cfg["minibatch_size"]
    batch = h_len * n_envs
    n_batches = (batch + requested - 1) // requested
    mb = batch // n_batches if n_batches > 1 else batch
    if batch % mb != 0:
        print(f"Error: batch ({batch}) not divisible by minibatch ({mb})")
        sys.exit(1)
    if mb != requested:
        print(f"Warning: minibatch_size {requested} does not divide batch {batch} "
              f"(num_actors={n_envs} * horizon={h_len}); snapped to {mb} ({n_batches} minibatches).")
    cfg["minibatch_size"] = mb


def _find_xwindow(display, title_fragments):
    """Find the vsim render window by title/class fragment, falling back to largest mapped window."""
    from Xlib import X, error as Xerror

    root = display.screen().root
    candidates = []

    def collect(win, depth=0):
        try:
            attrs = win.get_attributes()
            if attrs.map_state != X.IsViewable:
                return
            name = win.get_wm_name() or ""
            cls = win.get_wm_class() or ()
            combined = (name + " " + " ".join(cls)).lower()
            if any(t.lower() in combined for t in title_fragments):
                geom = win.get_geometry()
                candidates.append((geom.width * geom.height, win))
                return  # don't recurse into matched window
        except Xerror.BadWindow:
            return
        try:
            for child in win.query_tree().children:
                collect(child, depth + 1)
        except Xerror.BadWindow:
            pass

    collect(root)
    if candidates:
        candidates.sort(key=lambda x: x[0], reverse=True)
        return candidates[0][1]

    # No match. Deliberately no "largest window" fallback: it grabs whatever
    # else is on screen (e.g. the editor). A missing match means the window is
    # on a non-visible workspace (unviewable) or SDL_VIDEO_X11_WMCLASS didn't
    # apply.
    return None


class _VideoRecorder:
    """Captures the vsim X11 window each render_callback and writes to mp4."""

    def __init__(self, path: Path, max_frames: int, stop_env: bool):
        self.path = path
        self.max_frames = max_frames
        self.stop_env = stop_env
        self._writer = None
        self._display = None
        self._window = None
        self._size = None
        self._frames = 0
        self.done = False
        self._warned = False

    def _warn_once(self, msg):
        if not self._warned:
            print(f"[video] {msg}")
            self._warned = True

    def _init(self):
        from Xlib import display as xdisplay
        self._display = xdisplay.Display()

    def _get_window(self):
        if self._window is not None:
            return self._window
        win = _find_xwindow(self._display, _VSIM_WINDOW_TITLES)
        if win:
            self._window = win
        return win

    def _capture(self):
        import numpy as np
        from Xlib import X
        win = self._get_window()
        if win is None:
            self._warn_once(
                "render window not found — is it on a visible workspace? "
                "(matching WM_CLASS 'vsim_render')"
            )
            return None
        try:
            # Unviewable windows (on a non-visible workspace) yield black frames;
            # skip them so the video only contains real, rendered frames.
            if win.get_attributes().map_state != X.IsViewable:
                self._warn_once(
                    "render window is hidden (non-visible workspace) — frames "
                    "skipped; keep it on a visible workspace to record."
                )
                return None
            geom = win.get_geometry()
            raw = win.get_image(0, 0, geom.width, geom.height, X.ZPixmap, 0xFFFFFFFF)
        except Exception:
            self._window = None  # window may have moved; retry next frame
            return None
        img = np.frombuffer(raw.data, dtype=np.uint8).reshape(geom.height, geom.width, 4)
        # Drop alpha → BGR; copy to a writable C-contiguous array (VideoWriter requires it).
        # Crop to even dims so the mp4 encoder doesn't choke.
        h = geom.height - (geom.height & 1)
        w = geom.width - (geom.width & 1)
        return np.ascontiguousarray(img[:h, :w, :3])

    def make_callback(self, env):
        def callback():
            if self.done:
                return
            try:
                import cv2
                if self._display is None:
                    self._init()
                frame = self._capture()
                if frame is None:
                    return
                if self._writer is None:
                    h, w = frame.shape[:2]
                    self._size = (w, h)
                    fps = round(1.0 / env.dt) if hasattr(env, "dt") else 60
                    self.path.parent.mkdir(parents=True, exist_ok=True)
                    self._writer = cv2.VideoWriter(
                        str(self.path),
                        cv2.VideoWriter_fourcc(*"mp4v"),
                        fps,
                        self._size,
                    )
                    print(f"Recording video → {self.path}  ({fps} fps, max {self.max_frames} frames)")
                if (frame.shape[1], frame.shape[0]) != self._size:
                    frame = cv2.resize(frame, self._size)
                self._writer.write(frame)
                self._frames += 1
                if self._frames >= self.max_frames:
                    self._writer.release()
                    self.done = True
                    print(f"Video saved: {self.path}")
                    if self.stop_env:
                        env.render_finished = True
            except Exception as e:
                print(f"[video] capture error: {e}")
        return callback


class _MouseInput:
    """Out-of-band X11 mouse reader for the follow camera.

    vlearn exposes no mouse API and ships as binary wheels, so we read the pointer
    directly via Xlib. In follow mode we warp the cursor back to the render
    window's centre every frame: VSim still sees the motion but it nets to zero
    (its pan self-cancels = "eaten") while we read the pre-warp delta to drive
    orbit. Scroll (transient X buttons 4/5) is monitored read-only via the RECORD
    extension on a daemon thread (no grab, so VSim still gets the events). Active
    only while the VSim window holds input focus, so alt-tab never traps the
    desktop cursor. Degrades to a no-op if there's no X display / window.
    """

    def __init__(self, window_titles):
        self._titles = window_titles
        self._disp = None
        self._win = None
        self._warped = False          # suppress the jump on the first active frame
        self._scroll = 0
        self._lock = threading.Lock()
        self._xfixes = False          # cursor hide/show available
        self._cursor_hidden = False
        try:
            from Xlib import display as xdisplay
            self._disp = xdisplay.Display()
            self._win = _find_xwindow(self._disp, window_titles)
        except Exception as e:
            print(f"[camera] mouse disabled (no X display: {e})")
            return
        try:
            from Xlib.ext import xfixes  # noqa: F401  (registers xfixes_* methods)
            self._disp.xfixes_query_version()
            self._xfixes = True
        except Exception:
            pass                       # cursor stays visible; orbit still works
        threading.Thread(target=self._scroll_loop, daemon=True).start()

    # --- scroll: read-only global RECORD tap on a daemon thread ---
    def _scroll_loop(self):
        try:
            from Xlib import X, display as xdisplay
            from Xlib.ext import record
            from Xlib.protocol import rq
        except Exception:
            return
        d = xdisplay.Display()
        if not d.has_extension("RECORD"):
            print("[camera] RECORD ext missing — scroll-zoom disabled")
            return
        ctx = d.record_create_context(0, [record.AllClients], [{
            "core_requests": (0, 0), "core_replies": (0, 0),
            "ext_requests": (0, 0, 0, 0), "ext_replies": (0, 0, 0, 0),
            "delivered_events": (0, 0),
            "device_events": (X.ButtonPress, X.ButtonPress),
            "errors": (0, 0), "client_started": False, "client_died": False,
        }])

        def handler(reply):
            if reply.category != record.FromServer or not reply.data:
                return
            data = reply.data
            while data:
                ev, data = rq.EventField(None).parse_binary_value(data, d.display, None, None)
                if ev.type == X.ButtonPress:
                    if ev.detail == 4:      # wheel up
                        self._add_scroll(1)
                    elif ev.detail == 5:    # wheel down
                        self._add_scroll(-1)

        d.record_enable_context(ctx, handler)  # blocks for the session

    def _add_scroll(self, n):
        with self._lock:
            self._scroll += n

    def _drain_scroll(self):
        with self._lock:
            s, self._scroll = self._scroll, 0
        return s

    def _focused(self):
        """True iff the VSim window (or a descendant) holds input focus."""
        try:
            w = self._disp.get_input_focus().focus
        except Exception:
            return False
        for _ in range(8):
            if not hasattr(w, "id"):
                return False
            if self._win is not None and w.id == self._win.id:
                return True
            try:
                w = w.query_tree().parent
            except Exception:
                return False
            if w is None:
                return False
        return False

    def _hide_cursor(self, hidden):
        """Hide/show the cursor (XFixes, reference-counted per client), only on a
        state change so the ref-count stays balanced."""
        if not self._xfixes or hidden == self._cursor_hidden:
            return
        try:
            target = self._win or self._disp.screen().root
            if hidden:
                target.xfixes_hide_cursor()
            else:
                target.xfixes_show_cursor()
            self._disp.sync()
            self._cursor_hidden = hidden
        except Exception:
            pass

    def poll(self, active):
        """Return (dx, dy, scroll_ticks). dx/dy are pixels since last frame (0 on
        the first active frame). Warps the cursor to window-centre and hides it
        while active; a no-op (zeros, cursor shown) when inactive, window-less, or
        VSim isn't focused."""
        if self._disp is None:
            return 0, 0, 0
        if self._win is None:
            self._win = _find_xwindow(self._disp, self._titles)
        if not active or self._win is None or not self._focused():
            self._warped = False
            self._hide_cursor(False)
            self._drain_scroll()        # discard so it doesn't pile up while idle
            return 0, 0, 0
        try:
            geom = self._win.get_geometry()
            root = self._disp.screen().root
            c = self._win.translate_coords(root, geom.width // 2, geom.height // 2)
            p = root.query_pointer()
            root.warp_pointer(c.x, c.y)
            self._disp.sync()
        except Exception:
            self._win = None            # window moved/closed — re-find next frame
            self._warped = False
            self._hide_cursor(False)
            return 0, 0, 0
        self._hide_cursor(True)
        ticks = self._drain_scroll()
        if not self._warped:            # first active frame: seed centre, no delta
            self._warped = True
            return 0, 0, ticks
        return p.root_x - c.x, p.root_y - c.y, ticks


class FollowCamera:
    """Drives the viewer camera, with operator override via mouse + keys + GUI panel.

    Three viewing states (see scripts/CONTEXT.md "Follow camera"):
      - auto-cycle : hops to a random robot each episode (the unattended default).
      - manual-follow : locked to one chosen group+env, persists across resets.
      - free-cam : camera detached; the renderer's built-in WASD/drag fly it.
    auto/manual are mutually exclusive; free-cam is an orthogonal overlay that
    leaves the underlying mode untouched (toggling it off resumes that mode).

    Orbit/zoom is mouse-driven in the two fixed states: the cursor is pinned to
    the window (warped back to centre each frame, see _MouseInput) and its motion
    orbits the focused robot while the scroll wheel zooms. Because the cursor is
    pinned, the GUI widgets aren't clickable while following — press F (free-cam)
    to release the cursor and use the panel. In free-cam the mouse hands back to
    the renderer's built-in WASD/drag.

    Discrete keys (single-step, rising-edge), all left-hand and clear of vsim's
    built-in WASD/P/O (camera move / pause / single-step), which we can't poll:
      Q / E  prev / next group        Z / X  prev / next env
      F      toggle free-cam          C      back to auto-cycle
    The viewpoint in fixed states is a spherical offset (azimuth, elevation,
    radius) around the focused robot; mouse/scroll mutate it and it persists as the
    focus hops between robots. The GUI panel mirrors and also drives the discrete
    state (bidirectional); keys win over a same-frame widget click.
    """

    MAX_FRAMES_PER_MORPH = 250
    _KEYS = ("q", "e", "z", "x", "f", "c", "r", "t")  # discrete (rising-edge), left-hand
                                                    # r/t = prev/next epoch (policy switch)
    MOUSE_SENS_BASE = math.radians(0.15)            # radians per pixel at sens=1.0
    SCROLL_ZOOM = 1.1                               # radius multiply per wheel tick
    ELEV_MIN = math.radians(0.0)                  # don't look up from below the robot
    ELEV_MAX = math.radians(85.0)                   # clamp near top to avoid pole flip
    RADIUS_MIN, RADIUS_MAX = 1.0, 30.0

    def __init__(self, env, offset_xyz=CAMERA_OFFSET, switch=None):
        import vlearn as v
        self._v = v
        self.env = env
        self.switch = switch        # PolicySwitch | None (live epoch/run switching)
        # Spherical viewpoint around the focus, seeded from the static offset so
        # the default view is unchanged. azimuth in XZ plane, elevation above it.
        ox, oy, oz = offset_xyz
        self.radius = (ox * ox + oy * oy + oz * oz) ** 0.5
        self.azimuth = math.atan2(oz, ox)
        self.elevation = math.asin(oy / self.radius)
        self.sets = env.follow_sets()
        self.n_groups = len(self.sets)
        self.epm = len(self.sets[0]) if self.sets else 1

        self.mode = "auto"          # "auto" | "manual"
        self.free = False
        self.gi = 0                 # group index
        self.ei = 0                 # env index within group
        self._order = []
        self._last_set = None
        self._last_progress = -1
        self._start_time = 0
        self._prev_keys = {k: False for k in self._KEYS}
        self._pick_new_auto()

        self._mouse = _MouseInput(_VSIM_WINDOW_TITLES)
        self._build_panel()
        print("[camera] mouse orbit + scroll zoom (follow)  Q/E group  Z/X env  F free-cam  C auto-cycle")

    def _offset_look(self):
        """Camera offset (focus->eye) and look direction from the spherical state."""
        v = self._v
        horiz = self.radius * math.cos(self.elevation)
        ox = horiz * math.cos(self.azimuth)
        oz = horiz * math.sin(self.azimuth)
        oy = self.radius * math.sin(self.elevation)
        look = v.Vec3(-ox / self.radius, -oy / self.radius, -oz / self.radius)
        return v.Vec3(ox, oy, oz), look

    # --- GUI panel (bidirectional). The control legend is embedded in the widget
    # labels: UserText/UserSeparator exist in the bindings but this vlearn build's
    # renderer doesn't draw them (registering one blanks the whole menu), so keys
    # live in the labels. Widgets are only clickable in free-cam (the cursor is
    # pinned while following). Re-run by update() whenever the gym is rebuilt (the
    # codesign env resamples bodies each episode -> new gym_render). ---
    def _build_panel(self) -> None:
        v, r = self._v, self.env.gym_render
        prev_sens = self.w_sens.get_value() if hasattr(self, "w_sens") else 1.0
        self.w_group = v.UserCombo("Group (Q/E)", [str(i) for i in range(self.n_groups)], self.gi)
        self.w_env = v.UserCombo("Env (Z/X)", [str(i) for i in range(self.epm)], self.ei)
        self.w_free = v.UserCheckbox("Free cam (F) - frees mouse for GUI", self.free)
        self.w_auto = v.UserCheckbox("Auto-cycle (C)", self.mode == "auto")
        self.w_sens = v.UserSlider("Mouse sens - move=orbit, scroll=zoom", 0.05, 2.0, prev_sens)
        for w in (self.w_group, self.w_env, self.w_free, self.w_auto, self.w_sens):
            r.register_menu_item(w)
        self._panel_render = r          # the gym_render these widgets are registered on
        if self.switch is not None:
            self._build_switch_widgets()
        self._sync_panel()

    # --- policy-switch widgets: a Run combo (model dir only) + an Epoch combo.
    # Rebuilt from scratch by _build_panel on every gym swap; the Epoch combo also
    # rebuilds on a run change (UserCombo items are fixed at construction). ---
    def _build_switch_widgets(self) -> None:
        v, r = self._v, self.env.gym_render
        self.w_run = None
        if self.switch.has_runs:
            self.w_run = v.UserCombo("Run", self.switch.run_labels(), self.switch.run_idx)
            r.register_menu_item(self.w_run)
            self._run_syncedidx = self.switch.run_idx
        self.w_epoch = None             # fresh render: don't unregister the dead one
        self._build_epoch_combo(self.switch.run_idx)

    def _build_epoch_combo(self, run_idx: int) -> None:
        v, r = self._v, self.env.gym_render
        if getattr(self, "w_epoch", None) is not None:
            r.unregister_menu_item(self.w_epoch)
        labels = self.switch.epoch_labels(run_idx)
        idx = self.switch.epoch_idx if run_idx == self.switch.run_idx else 0
        idx = min(idx, len(labels) - 1)
        self.w_epoch = v.UserCombo("Epoch (R/T)", labels, idx)
        r.register_menu_item(self.w_epoch)
        self._epoch_syncedidx = idx

    def _sync_panel(self) -> None:
        self.w_group.set_current_index(self.gi)
        self.w_env.set_current_index(self.ei)
        self.w_free.set_value(self.free)
        self.w_auto.set_value(self.mode == "auto")
        self._synced = (self.gi, self.ei, self.free, self.mode == "auto")

    # --- auto-cycle selection ---
    def _next_set(self) -> int:
        if not self._order:
            self._order = list(range(self.n_groups))
            random.shuffle(self._order)
            if len(self._order) > 1 and self._order[0] == self._last_set:
                self._order.append(self._order.pop(0))
        return self._order.pop(0)

    def _pick_new_auto(self) -> None:
        self.gi = self._last_set = self._next_set()
        self.ei = random.randrange(self.epm)
        self._start_time = int(self.env.progress_buf[self._cur_idx()].item())
        self._last_progress = -1

    def _cur_idx(self) -> int:
        return self.sets[self.gi][self.ei]

    def _enter_manual(self) -> None:
        # Grab whatever is currently followed (gi/ei already point at it) and lock.
        self.mode = "manual"
        self.free = False

    # --- per-frame update ---
    def update(self) -> None:
        r = self.env.gym_render

        # A gym rebuild (codesign resamples bodies each episode) swaps gym_render,
        # orphaning our menu on the dead one — re-register on the live render.
        if r is not self._panel_render:
            self._build_panel()

        # Rising-edge key events.
        cur = {k: r.is_key_down(k) for k in self._KEYS}
        ev = {k: cur[k] and not self._prev_keys[k] for k in self._KEYS}
        self._prev_keys = cur

        # Widget user-changes since last sync (only meaningful if no key overrides).
        sg, se, sfree, sauto = self._synced
        u_group = self.w_group.get_current_index() != sg
        u_env = self.w_env.get_current_index() != se
        u_free = self.w_free.get_value() != sfree
        u_auto = self.w_auto.get_value() != sauto

        dg = (1 if ev["e"] else 0) - (1 if ev["q"] else 0)   # group: Q prev / E next
        de = (1 if ev["x"] else 0) - (1 if ev["z"] else 0)   # env:   Z prev / X next

        # Keys win over widget clicks. Selection > auto-toggle > free-toggle.
        if dg or de:
            self._enter_manual()
            self.gi = (self.gi + dg) % self.n_groups
            self.ei = (self.ei + de) % self.epm
        elif ev["c"]:
            self._pick_new_auto()
            self.mode, self.free = "auto", False
        elif ev["f"]:
            self.free = not self.free
        elif u_group or u_env:
            self._enter_manual()
            self.gi = self.w_group.get_current_index()
            self.ei = self.w_env.get_current_index()
        elif u_auto:
            if self.w_auto.get_value():
                self._pick_new_auto()
                self.mode, self.free = "auto", False
            else:
                self._enter_manual()
        elif u_free:
            self.free = self.w_free.get_value()

        # --- Live policy switch. Base off any un-applied request so rapid R/T
        # queue correctly; the player commits run_idx/epoch_idx when it applies. ---
        if self.switch is not None:
            s = self.switch
            base_run, base_epoch = s.pending or (s.run_idx, s.epoch_idx)
            depoch = (1 if ev["t"] else 0) - (1 if ev["r"] else 0)   # R prev / T next
            u_run = self.w_run is not None and self.w_run.get_current_index() != self._run_syncedidx
            u_epoch = self.w_epoch.get_current_index() != self._epoch_syncedidx
            if u_run:                                   # new run -> repopulate epochs, jump to best
                new_run = self.w_run.get_current_index()
                self._run_syncedidx = new_run
                self._build_epoch_combo(new_run)        # new run's epochs, best (idx 0) selected
                s.request(new_run, 0)
            elif depoch:
                n = s.n_epochs(base_run)
                new_epoch = (base_epoch + depoch) % n
                self.w_epoch.set_current_index(new_epoch)
                self._epoch_syncedidx = new_epoch
                s.request(base_run, new_epoch)
            elif u_epoch:
                new_epoch = self.w_epoch.get_current_index()
                self._epoch_syncedidx = new_epoch
                s.request(base_run, new_epoch)

        # Orbit/zoom: mouse-driven, inert in free-cam (built-in controls own it
        # there). Cursor is pinned to window-centre while following; dx/dy are the
        # pre-warp pixel deltas, scroll ticks zoom.
        dx, dy, ticks = self._mouse.poll(not self.free)
        if not self.free:
            sens = self.MOUSE_SENS_BASE * self.w_sens.get_value()
            self.azimuth += dx * sens
            self.elevation -= dy * sens          # natural Y: mouse up -> over the top
            self.elevation = max(self.ELEV_MIN, min(self.ELEV_MAX, self.elevation))
            if ticks:
                self.radius *= self.SCROLL_ZOOM ** (-ticks)   # wheel up -> zoom in
            self.radius = max(self.RADIUS_MIN, min(self.RADIUS_MAX, self.radius))

        # Auto-cycle hops to a new robot when the followed episode ends or times out.
        if self.mode == "auto":
            prog = int(self.env.progress_buf[self._cur_idx()].item())
            if prog - self._start_time > self.MAX_FRAMES_PER_MORPH or prog < self._last_progress:
                self._pick_new_auto()
                prog = int(self.env.progress_buf[self._cur_idx()].item())
                self._start_time = prog
            self._last_progress = prog

        self._sync_panel()
        if not self.free:
            offset, look = self._offset_look()
            eye = self.env.follow_world_pos(self._cur_idx()) + offset
            r.reset_camera(eye, look)


def _attach_render_callback(env, recorder=None, switch=None) -> None:
    """Compose follow-camera (if env supports it and is rendering) + video capture."""
    hooks = []
    if getattr(env, "rendering", False):
        env.gym_render.capped_step = True   # start real-time-capped, not free-running
    if getattr(env, "rendering", False) and hasattr(env, "follow_sets"):
        hooks.append(FollowCamera(env, switch=switch).update)
    if recorder is not None:
        hooks.append(recorder.make_callback(env))
    if not hooks:
        return

    def composite():
        for h in hooks:
            h()

    env.render_callback = composite


def _run_random(config, args, video_path=None, num_episodes=1) -> None:
    """Smoke path: the config's task, library and seed body, one body per env.

    Used to be config-free (hardcoded Ant + the default library). It cannot be any more -- the task
    itself now only exists in the config -- and reading the `env:` block here means `random` honours
    the same Task constructor knobs training does.
    """
    import torch
    import vlearn as v
    from codesigner.components.modular_libraries import REGISTRY as ml_registry
    _, task_class = _resolve_task(config)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    num_envs = args.num_envs or 1
    env_cfg = dict(config.get("env", {}))
    for key in ("task", "num_morphs"):
        env_cfg.pop(key, None)
    library = ml_registry[env_cfg.pop("module_library", "simple")](
        **env_cfg.pop("module_library_kwargs", {}))
    morphologies = [seed_body(library, **env_cfg.pop("base_morphology", {}))] * num_envs
    env = task_class(
        device=device,
        rendering=True,
        raise_exception=True,
        enable_scene_query=False,
        rootOffset=(v.Vec3(0, 0, 0), v.Quat(0, 0, 0, 1)),
        **env_cfg,
    )
    total = env.setup(library, num_envs, len(morphologies), morphologies, seed=args.seed)
    act_low = torch.tensor(env.action_space.low, device=device)
    act_high = torch.tensor(env.action_space.high, device=device)
    env.reset()
    recorder = None
    if video_path is not None:
        max_ep = getattr(env, "max_episode_length", 1000)
        # stop_env=False: we own this loop, so break on recorder.done instead of
        # tripping env.render_finished (which would raise via raise_exception).
        recorder = _VideoRecorder(video_path, max_frames=num_episodes * max_ep, stop_env=False)
    _attach_render_callback(env, recorder)
    from codesigner.backend.simulation import RenderFinished
    try:
        while not env.render_finished:
            if recorder is not None and recorder.done:
                break
            actions = act_low + torch.rand(total, act_low.shape[0], device=device) * (act_high - act_low)
            env.step(actions)
    except RenderFinished:
        pass  # window closed mid-step (render() raises from inside env.step)


def run_training(
    experiment: str,
    family: str | None = None,
    network: tuple | None = None,
    model: str = "continuous_a2c_logstd",
    extra_args_fn=None,
    post_config_fn=None,
) -> None:
    """Run one training/play/random session for the algorithm this entry point names.

    An entry point names an ALGORITHM (`family` + `experiment` + network + model) and never a task;
    `--config` supplies the task, and everything task-shaped -- env registration key, run name,
    output directory -- is composed from it by `_compose_identity`. There is deliberately no default
    config: the task is not something a script gets to assume.
    """
    import torch
    import vlearn as v
    import gymnasium.spaces
    from argparse import ArgumentParser
    from rl_games.common import env_configurations, vecenv
    from rl_games.common.ivecenv import IVecEnv
    from rl_games.algos_torch import model_builder as mb_module
    from rl_games.torch_runner import Runner
    from vlearn.spaces import Box, Discrete
    from vlearn.torch_utils.wrappers import NewToOldAPICompatilibity

    def convert_space(space):
        if isinstance(space, Box):
            return gymnasium.spaces.Box(low=space.low, high=space.high, shape=space.shape)
        if isinstance(space, Discrete):
            return gymnasium.spaces.Discrete(n=space.n)

    class VlearnEnv(IVecEnv):
        def __init__(self, config_dict, config_name, num_actors, **kwargs):
            self.envs = config_dict[config_name]["env_creator"](num_actors, **kwargs)
            # The count the Task actually built, not the one asked for (D17). They agree because
            # num_actors was snapped to a multiple of num_morphs before the Runner started; reading
            # it back is what makes a future disagreement visible instead of misaligning rollouts.
            self.num_actors = getattr(getattr(self.envs, "env", self.envs),
                                      "total_num_envs", num_actors)

        def step(self, actions):
            return self.envs.step(actions)

        def reset(self):
            return self.envs.reset()

        def get_env_info(self):
            env_info = {
                "observation_space": convert_space(self.envs.observation_space),
                "action_space": convert_space(self.envs.action_space),
                "value_size": getattr(getattr(self.envs, "env", self.envs), "value_size", 1),
            }
            if hasattr(self.envs, "state_space"):
                env_info["state_space"] = convert_space(self.envs.state_space)
            return env_info

    # --- Arg parsing ---
    parser = ArgumentParser()
    parser.add_argument("mode", nargs="?", choices=["train", "play", "random"], default="train")
    parser.add_argument("checkpoint", nargs="?", default=None)
    parser.add_argument("--num-episodes", type=int, default=None, dest="num_episodes")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--headless", choices=["True", "False"], default=None)
    parser.add_argument("--num_envs", type=int)
    parser.add_argument("--max_epochs", type=int)
    parser.add_argument("--horizon_length", type=int)
    parser.add_argument("--config", type=Path, required=True,
                        help="Run config (e.g. configs/ppo_ant_codesign_single.yaml). Required: it "
                             "is what names the task.")
    parser.add_argument("--name", type=str, default=None,
                        help="Run name: leaf output dir (runs/env/model/<name>) instead of timestamp.")
    parser.add_argument("--video", action="store_true")
    parser.add_argument("--timing", action="store_true", default=False,
                        help="Enable synced cuda phase timers (perf/*); adds cuda.synchronize overhead.")
    parser.add_argument("--mem_profile", action="store_true", default=False,
                        help="Enable per-node memory profiling (perf/mem_*); adds syncs + reset_peak.")
    parser.add_argument("--minibatch_size", type=int, default=None,
                        help="Override params.config.minibatch_size (e.g. 16384 to fit aux heads).")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VAL", dest="set_keys",
                        help="Override any config key by dotted path, repeatable. VAL is YAML-parsed. "
                             "e.g. --set params.config.generator.entropy_coef=0.3")
    if extra_args_fn is not None:
        extra_args_fn(parser)
    args = parser.parse_args()

    mode = args.mode
    checkpoint = args.checkpoint

    # --- Config loading ---
    # Must precede the play/random dispatch below: both need the task, and the task is only in the
    # config. Merge shared rl_games boilerplate (configs/defaults/base.yaml) UNDER the config
    # (resolved through its `extends:` chain) so per-config values win; then pin the identity fields
    # the entry point owns -- composed from the config's task, not hardcoded. See ADR-0006.
    with open(_PROJECT_ROOT / "configs" / "defaults" / "base.yaml") as f:
        base = yaml.safe_load(f)
    config = _deep_merge(base, _load_config(args.config))
    for kv in args.set_keys:                          # --set params.config.x.y=VAL
        key, _, val = kv.partition("=")
        *path, leaf = key.split(".")
        d = config
        for p in path:
            d = d.setdefault(p, {})
        d[leaf] = yaml.safe_load(val)                 # YAML-parse: 0.3 -> float, true -> bool
    task_key, env_class = _resolve_task(config)       # --set env.task=... is honoured, hence here
    identity = _compose_identity(task_key, family, experiment)
    env_name = identity["env_name"]
    name = identity["name"]

    params = config["params"]
    params["config"]["env_name"] = env_name
    params["config"]["name"] = name        # experiment-family label (drives checkpoint stems)
    params.setdefault("model", {})["name"] = model
    if network is not None:
        params.setdefault("network", {})["name"] = network[0]

    # --- play: a run/model dir enables live policy switching (see policy_switch.py);
    # a plain .pth stays single-checkpoint. base name = `name` (the checkpoint stem). ---
    switch = None
    if mode == "play" and checkpoint is not None:
        from .policy_switch import resolve_source, PolicySwitch
        source = resolve_source(Path(checkpoint), name)
        if source["mode"] != "file":
            switch = PolicySwitch(source)
            checkpoint = switch.current_path()        # placeholder; redirected post-filter
            print(f"[play] policy switching ({source['mode']} dir): "
                  f"{len(switch.runs)} run(s) found", flush=True)

    video_path = None
    if args.video:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        video_path = _VIDEOS_DIR / f"{timestamp}_{mode}.mp4"

    if mode == "random":
        _run_random(config, args, video_path=video_path, num_episodes=args.num_episodes or 1)
        return

    if "player" not in config["params"]["config"]:
        config["params"]["config"]["player"] = {}
    config["params"]["config"]["player"]["use_vecenv"] = True
    config["params"]["config"]["player"]["print_stats"] = False
    cfg = config["params"]["config"]
    cfg.setdefault("use_diagnostics", True)  # enables diagnostics/exp_var, clip_frac, rms_value
    cfg.setdefault("train_dir", identity["train_dir"])
    run_name = args.name or datetime.now().strftime("%d-%H-%M-%S")
    if mode == "train" and Path(cfg["train_dir"], run_name).exists():
        print(f"Error: run '{run_name}' already exists at {cfg['train_dir']}/{run_name}")
        sys.exit(1)
    cfg.setdefault("full_experiment_name", run_name)

    # --- Seed ---
    if args.seed is not None:
        config["params"]["seed"] = args.seed
        torch.cuda.manual_seed(args.seed)
        os.environ["PYTHONHASHSEED"] = str(args.seed)
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        # torch.use_deterministic_algorithms(True) is DELIBERATELY OFF. Under torch.compile it
        # makes inductor emit deterministic kernel variants (flash-attn backward, reductions) that
        # are numerically fragile; in bf16 (mixed_precision) they overflow -> Inf -> NaN grads ->
        # corrupted weights. Runs stay seed-stable (same sampling/init) but not bit-exact. To restore
        # bit-exact determinism you must ALSO drop bf16 (mixed_precision: false, i.e. fp32).
        # Full analysis + re-enable recipe: docs/troubleshooting/determinism_bf16_nan.md.
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        torch.cuda.set_device(0)
        torch.cuda.set_per_process_memory_fraction(1.0)

    # --- CLI overrides ---
    if args.timing:
        config["params"]["config"]["timing"] = True
    if args.mem_profile:
        config["params"]["config"]["mem_profile"] = True
    if args.minibatch_size is not None:
        config["params"]["config"]["minibatch_size"] = args.minibatch_size
    if args.num_envs is not None:
        config["params"]["config"]["num_actors"] = args.num_envs
    elif mode == "play":
        config["params"]["config"]["num_actors"] = 64   # snappy switch/rebuild; --num_envs overrides
    if args.max_epochs is not None:
        config["params"]["config"]["max_epochs"] = args.max_epochs
    if args.horizon_length is not None:
        config["params"]["config"]["horizon_length"] = args.horizon_length

    # --- Script-specific mutations (snap, extra arg application, etc.) ---
    if post_config_fn is not None:
        post_config_fn(args, config)

    # --- The run's library, seed body, and body count ---
    resolved_seed = args.seed if args.seed is not None else config["params"].get("seed")
    env_cfg = dict(config.get("env", {}))

    # ONE ModuleLibrary per run (D14): named once here, constructed once, handed to the Task at
    # setup() and to the network through `runtime`. The network used to rebuild its own from a name
    # in the network config -- two instances, free to disagree about slot count or vocabulary while
    # every derived width still looked plausible.
    from codesigner.components.modular_libraries import REGISTRY as ml_registry
    env_cfg.pop("task")                               # resolved above; not a Task ctor kwarg
    library = ml_registry[env_cfg.pop("module_library", "simple")](
        **env_cfg.pop("module_library_kwargs", {}))
    base_morphology = seed_body(library, **env_cfg.pop("base_morphology", {}))
    runtime.set_run(library=library, base_morphology=base_morphology)

    # `num_morphs` bodies share `num_envs` envs and the division ROUNDS DOWN, so the Task can build
    # fewer envs than asked for (D15-D17). rl_games sizes its rollout buffers from num_actors before
    # any env exists, so the count it is handed must already be the count the Task will reach --
    # snap it here rather than meeting the shortfall later as a shape mismatch or, worse, a rollout
    # that silently misaligns. Default: one body per env, which is what the baseline ran.
    ppo_cfg = config["params"]["config"]
    num_morphs = env_cfg.pop("num_morphs", None) or ppo_cfg["num_actors"]
    snapped = (ppo_cfg["num_actors"] // num_morphs) * num_morphs
    if snapped != ppo_cfg["num_actors"]:
        print(f"[env] num_actors {ppo_cfg['num_actors']} -> {snapped}: "
              f"{num_morphs} bodies x {ppo_cfg['num_actors'] // num_morphs} envs each", flush=True)
        ppo_cfg["num_actors"] = snapped
    if snapped < num_morphs:
        raise ValueError(f"num_actors={ppo_cfg['num_actors']} over num_morphs={num_morphs} "
                         "leaves no envs per body")

    # --- Minibatch adjustment ---
    if "horizon_length" in ppo_cfg:
        _adjust_minibatch(ppo_cfg, ppo_cfg["num_actors"], ppo_cfg["horizon_length"])
        if "central_value_config" in ppo_cfg:
            _adjust_minibatch(
                ppo_cfg["central_value_config"],
                ppo_cfg["num_actors"],
                ppo_cfg["horizon_length"],
            )

    # --- Rendering ---
    if args.video:
        args.headless = "False"
    elif args.headless is None:
        args.headless = "False" if mode == "play" else "True"
    rendering = not _str_to_bool(args.headless)

    env_kwargs = {
        "rendering": rendering,
        "raise_exception": rendering,
        "enable_scene_query": False,
        # No offset between the world frame and the scene root; every task places its own bodies.
        "rootOffset": (v.Vec3(0, 0, 0), v.Quat(0, 0, 0, 1)),
        **env_cfg,
    }

    # --- Video recorder setup ---
    recorder = None
    if args.video:
        if mode == "train":
            print("Error: --video is not supported in train mode")
            sys.exit(1)
        max_ep = env_kwargs.get("max_episode_length", 1000)
        max_frames = (args.num_episodes or 1) * max_ep
        recorder = _VideoRecorder(video_path, max_frames=max_frames, stop_env=(mode == "play"))

    # --- Env + vecenv registration ---
    num_episodes = args.num_episodes

    def create_envs(n, **kw):
        assert torch.cuda.is_available()
        device = torch.device("cuda:0")
        envs = env_class(device=device, **env_kwargs)
        # Two-phase (D7): __init__ takes simulation parameters, setup() takes the run -- the gym's
        # contact buffers are sized by the env count, so no gym exists until here. Window 0 runs on
        # the seed body in every group; the generator replaces the whole set at the first resample.
        built = envs.setup(library, n, num_morphs, [base_morphology] * num_morphs,
                           seed=resolved_seed)
        # num_actors was snapped to a multiple of num_morphs above, so the round-down has nothing
        # left to take. If that ever stops being true, rl_games' buffers are already the wrong size.
        assert built == n, f"Task built {built} envs, rl_games is sized for {n}"
        runtime.set_run(obs_layout=envs.obs_layout())
        if mode == "play":
            envs.inference_mode_post_init_callback()
        if switch is not None:                       # drop arch-incompatible runs (mixed-gen dirs)
            skipped = switch.filter_compatible(
                math.prod(envs.observation_space.shape), math.prod(envs.action_space.shape))
            if skipped:
                print("[play] skipped %d incompatible run(s): %s" % (
                    len(skipped), ", ".join(f"{n} ({why})" for n, why in skipped)), flush=True)
            print(f"[play] starting at {switch.current_label()} "
                  f"({len(switch.runs)} compatible run(s))", flush=True)
        _attach_render_callback(envs, recorder, switch)
        if mode == "play" and num_episodes is not None:
            max_ep = env_kwargs.get("max_episode_length", 1000)
            limiter = _PlayLimiter(envs, num_episodes * max_ep)
            return NewToOldAPICompatilibity(limiter)
        return NewToOldAPICompatilibity(envs)

    def make_vecenv(config_name, num_actors, **kw):
        return VlearnEnv(env_configurations.configurations, config_name, num_actors, **kw)

    env_configurations.register(
        env_name,
        {"vecenv_type": "VLEARN", "env_creator": create_envs},
    )
    vecenv.register("VLEARN", make_vecenv)

    if network is not None:
        net_name, net_builder = network
        if net_builder is not None:  # None = rl_games built-in (e.g. actor_critic); name only
            mb_module.register_network(net_name, net_builder)

    # Mask-passthrough normalizer variant (used by configs via model.name; harmless otherwise).
    from .models import TransformerMaskedNorm, MultiMorphValueBuilder, TransformerMaskedValue
    mb_module.register_model('transformer_masked_a2c_logstd', TransformerMaskedNorm)
    # PPG disjoint value net + value-only model (built by PPGAgent; harmless otherwise).
    mb_module.register_network('multimorph_limb_value', MultiMorphValueBuilder)
    mb_module.register_model('transformer_masked_value', TransformerMaskedValue)

    # --- Run ---
    if mode == "train":
        # Stamp the RESOLVED config (extends chain + base.yaml + every --set/CLI override applied)
        # into the run dir. `extends` and `--set` both resolve at load, so the yaml on disk no
        # longer answers "what knobs did this run use?" -- this file does.
        stamp = Path(cfg["train_dir"], run_name)
        stamp.mkdir(parents=True, exist_ok=True)
        with open(stamp / "config.yaml", "w") as f:
            yaml.safe_dump(config, f, default_flow_style=False, sort_keys=False)

    runner = Runner()
    # Swap in the metrics-logging agent for all continuous PPO runs (see logging_agent.py).
    from .logging_agent import LoggingA2CAgent
    runner.algo_factory.register_builder(
        'a2c_continuous', lambda **kwargs: LoggingA2CAgent(**kwargs)
    )
    from .ppg_agent import PPGAgent
    runner.algo_factory.register_builder(
        'ppg_continuous', lambda **kwargs: PPGAgent(**kwargs)
    )
    from .codesign_agent import CodesignAgent
    runner.algo_factory.register_builder(
        'codesign_continuous', lambda **kwargs: CodesignAgent(**kwargs)
    )
    # Play uses rl_games' Player, not the agent. PPG shares the standard continuous net so the
    # stock player runs it; codesign uses a custom player that samples bodies from the trained
    # generator distribution each episode (else it would just show the fixed base morph).
    # All three share SwitchMixin so play can hot-swap checkpoints (inert when switch=None).
    from .codesign_player import CodesignPlayer, SwitchablePlayer

    def _mk_player(cls):
        def build(**kwargs):
            p = cls(**kwargs)
            if switch is not None:
                p.attach_switch(switch)
            return p
        return build

    runner.player_factory.register_builder('a2c_continuous', _mk_player(SwitchablePlayer))
    runner.player_factory.register_builder('ppg_continuous', _mk_player(SwitchablePlayer))
    runner.player_factory.register_builder('codesign_continuous', _mk_player(CodesignPlayer))
    runner.load(config)

    run_args = {"train": mode == "train", "play": mode == "play"}
    if checkpoint:
        run_args["checkpoint"] = checkpoint
    if mode == "play":
        if checkpoint:
            print(f"[play] Loading model from checkpoint: {checkpoint}")
        else:
            print("[play] No checkpoint provided; running with randomly initialized model")
    from codesigner.backend.simulation import RenderFinished
    try:
        runner.run(run_args)
    except RenderFinished:
        # Intended viewer shutdown: window closed, --num-episodes cap, or video
        # budget reached. Any other Exception is a real crash and propagates.
        if recorder is not None and recorder.done:
            print("Recording complete, exiting.")
        else:
            print("Play finished, exiting.")
