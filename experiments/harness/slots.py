"""The device/process layer: what a slot is, how a run is pinned to one, and how a dead run is
classified. Shared by `scripts/tune.py` (hyperparameter trials) and `harness/launch.py` (paper
runs) — the two schedule different things onto the same 8-slice machine, and which GPU a process
lands on must have exactly one implementation or two runs will silently share a slice.

Promoted out of tune.py unchanged; the tuner imports these names and its call sites are untouched.
"""
import os
import re
import signal
import subprocess
from pathlib import Path

# ── slots ─────────────────────────────────────────────────────────────
# A SLOT is one unit of concurrency: a single pinned CUDA device running one process.
# How slots are supplied is a property of the machine, not of the caller -- the tuning server splits
# its two cards by MIG into 8, a workstation offers one per GPU -- so everything below deals in
# device STRINGS (what goes into CUDA_VISIBLE_DEVICES) and MIG is just one way to produce them.
# A run sees exactly one device and calls it cuda:0, which is why the hardcoded `cuda:0` sites in
# train_utils/algorithm need no change. slots=1 reproduces a serial launcher exactly.

_MIG_RE = re.compile(r"UUID:\s*(MIG-[^)\s]+)")
_GPU_RE = re.compile(r"^GPU\s+(\d+):.*UUID:\s*(GPU-[^)\s]+)")
_EP_RE  = re.compile(r"_ep_(\d+)_")

# Ordered most-specific first: an OOM often cascades into a generic CUDA error, so testing the
# rebuild signatures first would misfile a genuine out-of-memory as a transient fault and resume it
# forever. See docs/troubleshooting/resample_rebuild_crash.md for the rebuild signatures.
_OOM_SIG     = ("CUDA out of memory", "torch.OutOfMemoryError", "CUDA error: out of memory")
_REBUILD_SIG = ("Error deallocating pinned host memory", "CUDA error: invalid argument",
                "AcceleratorError")


class _Slot:
    __slots__ = ("name", "device")

    def __init__(self, name: str, device: str):
        self.name   = name      # short display label for the slot table
        self.device = device    # the CUDA_VISIBLE_DEVICES value


def _smi(query: str) -> str:
    """`nvidia-smi <query>` stdout, or "" if nvidia-smi is absent/fails (caller degrades)."""
    try:
        return subprocess.run(["nvidia-smi", *query.split()], capture_output=True,
                              text=True, timeout=30).stdout
    except (OSError, subprocess.SubprocessError):
        return ""


def _discover_devices() -> list[tuple[str, str]]:
    """(name, device) for every schedulable device, MIG instances if the box is partitioned else
    whole GPUs. Uses UUIDs, not indices: indices are reorderable by driver/env, UUIDs are not."""
    out = _smi("-L")
    devices, gpu_idx = [], -1
    for line in out.splitlines():
        g = _GPU_RE.match(line.strip())
        if g:
            gpu_idx = int(g.group(1))
            devices.append((f"gpu{gpu_idx}", g.group(2)))
            continue
        m = _MIG_RE.search(line)
        if m:                                    # indented MIG line: belongs to the GPU above it
            devices.append((f"g{gpu_idx}m{sum(1 for n, _ in devices if n.startswith(f'g{gpu_idx}m'))}",
                            m.group(1)))
    migs = [d for d in devices if d[0].startswith("g") and "m" in d[0][1:]]
    return migs if migs else [d for d in devices if d[0].startswith("gpu")]


def _busy_devices() -> set[str]:
    """UUIDs with a compute app on them right now. On a MIG box the driver may attribute an app to
    the PARENT GPU rather than the instance, in which case no slot matches and nothing is skipped --
    deliberately the safe direction: over-matching would mark all four slices of a card busy the
    moment one is used, and leave the scheduler with nothing to run. Unattributable apps are
    reported by _make_slots so an operator can see the check is not biting."""
    out = _smi("--query-compute-apps=gpu_uuid,pid --format=csv,noheader")
    return {ln.split(",")[0].strip() for ln in out.splitlines() if ln.strip()}


def _make_slots(spec, allow_busy: bool = False) -> list[_Slot]:
    """spec: an explicit list of device strings (used verbatim, no idle check -- the operator's
    override and the escape hatch if attribution misbehaves); an int N (first N idle discovered);
    or None/"auto" (every idle discovered device). `allow_busy` keeps devices that already carry a
    foreign compute app -- needed on a workstation, where a desktop session permanently occupies the
    only GPU, and never wanted on the compute-only tuning server."""
    if isinstance(spec, (list, tuple)) and spec:
        return [_Slot(f"s{i}", str(d)) for i, d in enumerate(spec)]

    found = _discover_devices()
    if not found:
        raise RuntimeError("no CUDA devices found via `nvidia-smi -L` (pass --slots to override)")
    busy = set() if allow_busy else _busy_devices()
    idle = [(n, d) for n, d in found if d not in busy]
    stray = busy - {d for _, d in found}
    if stray:
        print(f"[slots] {len(stray)} compute app(s) on {', '.join(sorted(stray))} could not be "
              f"attributed to a slot; idle-checking is not filtering them", flush=True)
    for n, d in found:
        if d in busy:
            print(f"[slots] skipping busy slot {n} ({d})", flush=True)
    if not idle:
        raise RuntimeError(f"all {len(found)} discovered devices are busy")
    if isinstance(spec, int) and spec > 0:
        idle = idle[:spec]
    return [_Slot(n, d) for n, d in idle]


def parse_slot_spec(spec):
    """CLI/YAML slot spec -> what _make_slots takes: "auto"/None, an int, or a device list."""
    if isinstance(spec, str):
        return None if spec == "auto" else (int(spec) if spec.isdigit() else spec.split(","))
    return spec


# ── processes ─────────────────────────────────────────────────────────

def _launch(cmd: list[str], device: str, log_path: Path, header: str):
    """Start a run pinned to `device`, streaming stdout+stderr to `log_path`. Own session
    (setsid) so the scheduler can killpg the whole cohort without the signal racing its children.
    Streams to a file rather than PIPE: the 64KB OS buffer would fill and deadlock the trainer."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logf = open(log_path, "w")
    logf.write(header)
    logf.flush()
    # Unbuffered: stdout block-buffers into a file, so a run killed (timeout, Ctrl-C) would lose
    # its last writes and the log would not be tailable while it runs. Tracebacks flush on their own
    # at interpreter exit, so _classify_exit is safe either way -- this is for liveness.
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=device, PYTHONUNBUFFERED="1")
    proc = subprocess.Popen(cmd, stdout=logf, stderr=subprocess.STDOUT, text=True,
                            env=env, start_new_session=True)
    return proc, logf


def _classify_exit(log_path: Path, tail_bytes: int = 200_000) -> str:
    """Why a run died: 'oom' (a real property of the settings -- never relaunch unchanged),
    'rebuild' (the known gym-teardown race -- resume from checkpoint), or 'unknown' (also resumed;
    a config that is genuinely broken will just fail again and exhaust its attempts)."""
    try:
        with open(log_path, "rb") as f:
            f.seek(0, os.SEEK_END)
            f.seek(max(0, f.tell() - tail_bytes))
            tail = f.read().decode("utf-8", "replace")
    except OSError:
        return "unknown"
    if any(s in tail for s in _OOM_SIG):
        return "oom"
    if any(s in tail for s in _REBUILD_SIG):
        return "rebuild"
    return "unknown"


def _kill(proc, grace: float = 30.0):
    """SIGTERM the run's whole process group, then SIGKILL whatever survives. Group-wide, not
    proc.terminate(): a run spawns its own children (compile workers, VSim threads) and signalling
    only the leader leaves them alive holding the slot's GPU memory, so the next run OOMs."""
    try:
        pgid = os.getpgid(proc.pid)
    except OSError:
        return
    for sig, wait in ((signal.SIGTERM, grace), (signal.SIGKILL, 10.0)):
        try:
            os.killpg(pgid, sig)
        except OSError:
            return
        try:
            proc.wait(timeout=wait)
            return
        except subprocess.TimeoutExpired:
            continue


# ── checkpoints ───────────────────────────────────────────────────────

def _ckpt_epoch(path: Path) -> int:
    """Epoch encoded in a periodic checkpoint's name, or -1. The filename also carries a float
    reward, so the epoch must be matched anchored on `_ep_<N>_`."""
    m = _EP_RE.search(path.name)
    return int(m.group(1)) if m else -1


def _newest_ckpt(nn_dir: Path) -> Path | None:
    """Latest periodic checkpoint, by epoch rather than mtime."""
    best, best_ep = None, -1
    for p in nn_dir.glob("last_*.pth"):
        ep = _ckpt_epoch(p)
        if ep > best_ep:
            best, best_ep = p, ep
    return best


def _purge_ckpts(nn_dir: Path, keep: int = 1) -> None:
    """Drop all but the newest `keep` checkpoints in ONE run's own nn/ dir. For TUNING only, where
    checkpoints exist solely so a crashed trial can resume -- the tuner scores from TensorBoard and
    never reads one -- so retaining every 19MB save would cost ~150GB over a full sweep. Paper runs
    must NOT purge: their checkpoints are the ladder's and specialization's input. Only unlinks
    regular files matching last_*.pth, by resolved name; never touches a directory."""
    items = [(_ckpt_epoch(p), p) for p in nn_dir.glob("last_*.pth") if _ckpt_epoch(p) >= 0]
    for _, p in sorted(items, key=lambda x: x[0], reverse=True)[max(0, keep):]:
        if p.is_file():
            p.unlink(missing_ok=True)
