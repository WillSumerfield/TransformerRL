"""MIG partitioning for the tuning box: turn N whole GPUs into N*k schedulable slices.

The layout is MACHINE state, not study state -- you set it once and run many studies against it --
so this is a standalone CLI, never something a launcher does as a side effect. Three reasons it
stays out of tune.py's run path: it needs sudo and a 4-day run should not hold a privilege it never
uses; `-dci`/`-dgi` destroy instances OTHER jobs may be running on; and reconfiguration requires the
card to be idle, which is exactly what a launcher cannot assume.

`slots._discover_devices` already returns MIG instances when the box is partitioned and whole GPUs
otherwise, so nothing downstream needs to know this ran. Not running it is the off switch -- on a
workstation `slots: auto` finds the one card and `--slots 1 --allow-busy` covers the desktop
session.

    python -m experiments.harness.mig --status
    python -m experiments.harness.mig --list-profiles
    sudo python -m experiments.harness.mig --gpus 0,1 --profile 67 --count 4 [--dry-run]
    sudo python -m experiments.harness.mig --gpus 0,1 --reset

Profile ids are per-GPU-model and NOT portable: 67 is this box's ~24GB slice. `--list-profiles`
prints what the installed cards actually offer; there is no sane default to guess.
"""
import argparse
import os
import subprocess
import sys

from .slots import _discover_devices, _smi


def _run(args: list[str], dry: bool) -> int:
    """Echo then run one nvidia-smi call. Echoing always -- these are destructive and an operator
    should be able to see, copy and re-run exactly what was issued."""
    print("  $ " + " ".join(args), flush=True)
    if dry:
        return 0
    return subprocess.run(args).returncode


def _busy(gpu: str) -> list[str]:
    """PIDs of compute apps on GPU index `gpu`. Reconfiguring under a live job kills it, and the
    driver reports a MIG app against either the instance or the parent, so this asks the parent."""
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


def configure(gpus: list[str], profile: str, count: int, dry: bool) -> int:
    """Destroy existing instances on each GPU, then create `count` instances of `profile`.

    -dci/-dgi before -cgi because instance ids are not reusable while children exist; -C creates the
    matching compute instance per GPU instance, which is what makes each slice actually schedulable.
    """
    # Preflight is ADVISORY under --dry-run: previewing the exact command sequence is the whole
    # point of the flag, and it must stay useful on a machine that cannot run MIG at all.
    for g in gpus:
        if not _mig_enabled(g):
            print(f"GPU {g}: MIG mode is not enabled. Enable it first (needs an idle card, and on "
                  f"some drivers a reset or reboot):\n    sudo nvidia-smi -i {g} -mig 1",
                  file=sys.stderr)
            if not dry:
                return 1
        if (apps := _busy(g)):
            print(f"GPU {g}: {len(apps)} compute app(s) running -- refusing to reconfigure.\n  "
                  + "\n  ".join(apps), file=sys.stderr)
            if not dry:
                return 1

    for g in gpus:
        print(f"[mig] GPU {g}: {count} x profile {profile}")
        _run(["nvidia-smi", "mig", "-i", g, "-dci"], dry)   # rc ignored: nothing to destroy is fine
        _run(["nvidia-smi", "mig", "-i", g, "-dgi"], dry)
        if (rc := _run(["nvidia-smi", "mig", "-i", g, "-cgi", ",".join([profile] * count), "-C"], dry)):
            print(f"GPU {g}: create failed (rc={rc}). `--list-profiles` shows valid ids; a profile "
                  f"the card cannot fit {count} of will fail here.", file=sys.stderr)
            return rc
    if not dry:
        print()
        status()
    return 0


def reset(gpus: list[str], dry: bool) -> int:
    """Back to whole GPUs. Leaves MIG MODE on -- flipping that needs an idle card and sometimes a
    reboot, so it stays a deliberate separate step."""
    for g in gpus:
        if (apps := _busy(g)):
            print(f"GPU {g}: {len(apps)} compute app(s) running -- refusing.", file=sys.stderr)
            return 1
    for g in gpus:
        print(f"[mig] GPU {g}: destroying instances")
        _run(["nvidia-smi", "mig", "-i", g, "-dci"], dry)
        _run(["nvidia-smi", "mig", "-i", g, "-dgi"], dry)
    if not dry:
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
    p.add_argument("--dry-run", action="store_true", help="print the nvidia-smi calls, run nothing")
    a = p.parse_args()

    if a.status:
        return status() or 0
    if a.list_profiles:
        print(_smi("mig -lgip").rstrip() or "nvidia-smi unavailable")
        return 0

    gpus = [g.strip() for g in a.gpus.split(",") if g.strip()]
    if os.geteuid() != 0 and not a.dry_run:
        print("MIG reconfiguration needs root -- re-run under sudo (or --dry-run to preview).",
              file=sys.stderr)
        return 1
    return reset(gpus, a.dry_run) if a.reset else configure(gpus, a.profile, a.count, a.dry_run)


if __name__ == "__main__":
    sys.exit(main())
