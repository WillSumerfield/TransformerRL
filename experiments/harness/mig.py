"""MIG partitioning for the tuning box: turn N whole GPUs into N*k schedulable slices.

The layout is MACHINE state, not study state -- you set it once and run many studies against it --
so this is a standalone CLI, never something a launcher does as a side effect. Three reasons it
stays out of tune.py's run path: `-dci`/`-dgi` destroy instances OTHER jobs may be running on;
reconfiguration requires the card to be idle, which is exactly what a launcher cannot assume; and
it may need privileges a 4-day run has no business holding.

`slots._discover_devices` already returns MIG instances when the box is partitioned and whole GPUs
otherwise, so nothing downstream needs to know this ran. Not running it is the off switch -- on a
workstation `slots: auto` finds the one card and `--slots 1 --allow-busy` covers the desktop
session.

    python experiments/harness/mig.py --status
    python experiments/harness/mig.py --list-profiles
    python experiments/harness/mig.py --gpus 0,1 --profile 67 --count 4
    python experiments/harness/mig.py --gpus 0,1 --reset

Profile ids are per-GPU-model and NOT portable: 67 is this box's ~24GB slice. `--list-profiles`
prints what the installed cards actually offer; there is no sane default to guess.

Privileges are NOT pre-checked. Whether MIG reconfiguration needs root depends on the device and
the driver's permission setup, so a euid test would refuse on machines where it would have worked.
The commands are simply run, and a permission failure is reported as one.
"""
import argparse
import subprocess
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT))            # so `python experiments/harness/mig.py` works

# Absolute, not `from .slots import ...`: a relative import has no parent package when the file is
# run as a script, so the module-vs-script distinction would decide whether this file imports at all.
from experiments.harness.slots import _discover_devices, _smi   # noqa: E402

# nvidia-smi's wording for "you are not allowed to do this", across driver versions.
_PERM = ("insufficient permission", "permission denied", "not permitted", "requires root",
         "insufficient privileges", "operation not permitted")


def _run(args: list[str]) -> tuple[int, str]:
    """Echo, run, echo output. Echoing the command always -- these are destructive and an operator
    should be able to see, copy and re-run exactly what was issued. Output is captured rather than
    inherited so the caller can tell a permission failure from a real one, then printed either way."""
    print("  $ " + " ".join(args), flush=True)
    r = subprocess.run(args, capture_output=True, text=True)
    out = (r.stdout + r.stderr).strip()
    if out:
        print("    " + out.replace("\n", "\n    "), flush=True)
    return r.returncode, out


def _denied(rc: int, out: str, args: list[str]) -> bool:
    """True (and explains) if this failed for want of privileges rather than for a real reason."""
    if rc and any(p in out.lower() for p in _PERM):
        print(f"\nMIG reconfiguration was denied on this device -- re-run under sudo:\n"
              f"    sudo {' '.join(args)}", file=sys.stderr)
        return True
    return False


def _busy(gpu: str) -> list[str]:
    """Compute apps on GPU index `gpu`. Reconfiguring under a live job kills it, and the driver may
    attribute a MIG app to either the instance or the parent, so this asks the parent."""
    out = _smi(f"-i {gpu} --query-compute-apps=pid,process_name --format=csv,noheader")
    return [ln.strip() for ln in out.splitlines() if ln.strip()]


def _mig_enabled(gpu: str) -> bool:
    out = _smi(f"-i {gpu} --query-gpu=mig.mode.current --format=csv,noheader").strip()
    return out.lower().startswith("enabled")


def status() -> None:
    print(_smi("-L").rstrip() or "nvidia-smi unavailable")
    found = _discover_devices()
    kind = "MIG slice" if any("m" in n[1:] for n, _ in found) else "whole GPU"
    print(f"\n{len(found)} schedulable device(s) ({kind}): {', '.join(n for n, _ in found)}")


def _teardown(gpu: str) -> int:
    """Destroy compute instances then GPU instances. A non-permission failure is EXPECTED and
    ignored -- there is usually nothing to destroy -- but a denial is fatal and worth saying."""
    for sub in ("-dci", "-dgi"):
        args = ["nvidia-smi", "mig", "-i", gpu, sub]
        rc, out = _run(args)
        if _denied(rc, out, args):
            return 1
    return 0


def configure(gpus: list[str], profile: str, count: int) -> int:
    """Destroy existing instances on each GPU, then create `count` instances of `profile`.

    Teardown precedes creation because instance ids are not reusable while children exist; -C
    creates the matching compute instance per GPU instance, which is what makes a slice schedulable.
    """
    for g in gpus:
        if not _mig_enabled(g):
            print(f"GPU {g}: MIG mode is not enabled. Enable it first (needs an idle card, and on "
                  f"some drivers a reset or reboot):\n    nvidia-smi -i {g} -mig 1", file=sys.stderr)
            return 1
        if (apps := _busy(g)):
            print(f"GPU {g}: {len(apps)} compute app(s) running -- refusing to reconfigure.\n  "
                  + "\n  ".join(apps), file=sys.stderr)
            return 1

    for g in gpus:
        print(f"[mig] GPU {g}: {count} x profile {profile}")
        if _teardown(g):
            return 1
        args = ["nvidia-smi", "mig", "-i", g, "-cgi", ",".join([profile] * count), "-C"]
        rc, out = _run(args)
        if rc:
            if not _denied(rc, out, args):
                print(f"GPU {g}: create failed (rc={rc}). `--list-profiles` shows valid ids; a "
                      f"profile the card cannot fit {count} of will fail here.", file=sys.stderr)
            return rc
    print()
    status()
    return 0


def reset(gpus: list[str]) -> int:
    """Back to whole GPUs. Leaves MIG MODE on -- flipping that needs an idle card and sometimes a
    reboot, so it stays a deliberate separate step."""
    for g in gpus:
        if (apps := _busy(g)):
            print(f"GPU {g}: {len(apps)} compute app(s) running -- refusing.", file=sys.stderr)
            return 1
    for g in gpus:
        print(f"[mig] GPU {g}: destroying instances")
        if _teardown(g):
            return 1
    print()
    status()
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--gpus", default="0,1", help="comma-separated GPU indices (default 0,1)")
    p.add_argument("--profile", default="67", help="MIG profile id, per-model (default 67 ~ 24GB here)")
    p.add_argument("--count", type=int, default=4, help="instances per GPU (default 4)")
    p.add_argument("--reset", action="store_true", help="destroy instances, back to whole GPUs")
    p.add_argument("--status", action="store_true", help="show devices as the scheduler sees them")
    p.add_argument("--list-profiles", action="store_true", help="valid profile ids for these cards")
    a = p.parse_args()

    if a.status:
        return status() or 0
    if a.list_profiles:
        print(_smi("mig -lgip").rstrip() or "nvidia-smi unavailable")
        return 0

    gpus = [g.strip() for g in a.gpus.split(",") if g.strip()]
    return reset(gpus) if a.reset else configure(gpus, a.profile, a.count)


if __name__ == "__main__":
    sys.exit(main())
