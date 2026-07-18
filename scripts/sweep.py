#!/usr/bin/env python
"""sweep.py -- enumerate-and-run grid sweep of ONE config knob over EXPLICIT hand-picked values,
crossed with seeds, launching a given train script per point (subprocess, sequential). This is the
generalized form of the warmup-length test. NOT a search: for Bayesian/Optuna range search use
scripts/tune.py -- sweep runs exactly the values you name and prints a value->reward table.

The `--script` is the alg+env (each train_ant_*.py pins its own env/network/model + default config);
`--config` is the base config; every run = that base + `--set <param>=<value>` (+ any --extra-set).

Example (the warmup-length sweep it was generalized from):
  python scripts/sweep.py \
    --script scripts/train_ant_codesign_single.py \
    --config configs/ppo_ant_codesign_single.yaml \
    --param params.config.warmup_epochs --values 64,128,256 \
    --extra-set params.config.lr_warmup=true \
    --seeds 42 --max-epochs 1500 --name-prefix warm --control
  # -> warm64_s42, warm128_s42, warm256_s42 (+ warmctrl_s42), sequential, then a reward table.

Non-destructive: if a run name already exists the train script aborts that point (rc!=0); sweep marks
it FAILED and continues. It never deletes run dirs.
"""
import argparse
import re
import subprocess
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent


def _san(v):
    """Value -> a filesystem-safe run-name fragment (label column keeps the true value)."""
    return re.sub(r"[^0-9A-Za-z]+", "", str(v))


def _best_reward(name, runs_root):
    """Max rew_<float> across the run's saved checkpoints (best achieved); None if none found.
    Handles both `..._rew_2365.99.pth` and the `..._rew__2365.99_.pth` variant."""
    best = None
    for pth in runs_root.rglob(f"{name}/nn/*.pth"):
        for m in re.finditer(r"rew_+(-?\d+\.?\d*)", pth.name):
            r = float(m.group(1))
            best = r if (best is None or r > best) else best
    return best


def main():
    ap = argparse.ArgumentParser(description="Enumerate-and-run config sweep (Optuna search = tune.py)")
    ap.add_argument("--script", required=True, help="train script = the alg+env, e.g. scripts/train_ant_codesign_single.py")
    ap.add_argument("--config", required=True, help="base config yaml")
    ap.add_argument("--param", required=True, help="dotted config key to sweep, e.g. params.config.warmup_epochs")
    ap.add_argument("--values", required=True, help="comma-separated values for --param")
    ap.add_argument("--seeds", default="42", help="comma-separated seeds (default 42)")
    ap.add_argument("--name-prefix", required=True, dest="name_prefix", help="run = {prefix}{value}_s{seed}")
    ap.add_argument("--max-epochs", type=int, default=None, dest="max_epochs")
    ap.add_argument("--num-envs", type=int, default=None, dest="num_envs")
    ap.add_argument("--headless", default="True")
    ap.add_argument("--control", action="store_true", help="also run the base config unchanged (name {prefix}ctrl_s{seed})")
    ap.add_argument("--extra-set", action="append", default=[], metavar="KEY=VAL", dest="extra_set",
                    help="extra --set applied to EVERY run (repeatable), e.g. params.config.lr_warmup=true")
    ap.add_argument("--runs-root", default="runs", dest="runs_root", help="root globbed for reward scraping")
    args = ap.parse_args()

    values = [v.strip() for v in args.values.split(",") if v.strip()]
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    leaf = args.param.split(".")[-1]

    # (label, set-value or None-for-control) x seeds -> the run plan
    points = ([("control", None)] if args.control else []) + [(v, v) for v in values]
    plan = []
    for label, val in points:
        for seed in seeds:
            tag = "ctrl" if val is None else _san(val)
            plan.append((label, val, seed, f"{args.name_prefix}{tag}_s{seed}"))

    print(f"[sweep] {len(plan)} runs | param={args.param} | values={values} | seeds={seeds} "
          f"| control={args.control} | extra_set={args.extra_set}", flush=True)

    results = []
    for i, (label, val, seed, name) in enumerate(plan, 1):
        cmd = [sys.executable, args.script, "train", "--headless", args.headless,
               "--config", args.config, "--seed", str(seed), "--name", name]
        if args.max_epochs is not None:
            cmd += ["--max_epochs", str(args.max_epochs)]
        if args.num_envs is not None:
            cmd += ["--num_envs", str(args.num_envs)]
        for kv in args.extra_set:
            cmd += ["--set", kv]
        if val is not None:
            cmd += ["--set", f"{args.param}={val}"]
        print(f"\n[sweep {i}/{len(plan)}] {name}: {' '.join(cmd)}", flush=True)
        rc = subprocess.run(cmd, cwd=_ROOT).returncode
        results.append((label, seed, name, rc))

    # value -> best-reward table
    runs_root = _ROOT / args.runs_root
    print(f"\n=== sweep results ({leaf}) ===", flush=True)
    best = None
    for label, seed, name, rc in results:
        rew = _best_reward(name, runs_root) if rc == 0 else None
        status = f"rew {rew:.1f}" if rew is not None else (f"FAILED rc={rc}" if rc else "no reward found")
        print(f"  {name:<30} {str(label):>10} s{seed}   {status}")
        if rew is not None and (best is None or rew > best[0]):
            best = (rew, name)
    if best:
        print(f"  winner: {best[1]} (rew {best[0]:.1f})")


if __name__ == "__main__":
    main()
