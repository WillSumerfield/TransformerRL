"""Q4: is the BC teacher working -- does it hand the generator a similar starting distribution,
and is that distribution roughly the same in phase 3 and phase 5?

`gen/actor_loss` cannot answer this: it is an NLL, and its floor is the teacher's OWN entropy,
which phase 5 deliberately raised by adding the type-flip axis. A higher absolute NLL against a
higher-entropy teacher can represent identical fit quality. So compare DISTRIBUTIONS instead.

Compares three populations on the SKELETON view (module counts only -- the sole view phase 3 can
express) at the end-of-pretrain checkpoint:

  teacher   `_draw_parts_counts` -- the parts-copy warmup teacher, sampled directly
  gen@pre   the generator at the end-of-pretrain checkpoint (what BC actually produced)
  gen@end   the generator at the final checkpoint (where RL took it)

Reported per population: N_limb (effective limb designs/slot), the redundancy fraction
C/sum_n H(L_n) (cross-limb structure, scale-free), and cross-population mean d_struct against the
within-population value. If BC transferred the distribution, teacher and gen@pre match on all
three and cross ~= within.

The teacher's cross-limb correlation comes from `_draw_parts_counts`'s REJECTION SAMPLING on
`_is_stable` (a global slot-pattern property); the per-slot template draws themselves are
independent. So a teacher redundancy fraction near 0 would mean rejection is doing nothing.

Binds the REAL agent methods onto a shim (no gym, no net -- `_draw_parts_counts` needs neither)
rather than reimplementing them, so the teacher under test cannot silently diverge from the one
training used. Phase 3's version of the method is identical except for clamping limb length to
`_max_len` (4) instead of phase 5's `_max_eff` (3, the forced-cap grammar); that constant is set
per phase here, which is why this runs in ONE process for both.

Usage:  python experiments/teacher_fidelity.py [--runs a,b] [--n 4096]
"""
import os; os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
import argparse
import math
import re
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT.parent / "vlearn-main" / "train"))

import numpy as np

from experiments.diversity import counts_to_repr
from experiments.diversity_p5 import (encode_population, limb_entropies, pairwise_d_struct,
                                      population_to_repr)
from transformer_rl.morphology import CANONICAL_SLOTS

RUN_DIR = _ROOT / "runs/ant_codesign/codesign_single_transformer"
# phase 3 clamps parts-copy lengths to max_limb_length (4); phase 5 to max_effectors (3).
MAX_EFF = {"phase3": 4, "phase5": 3}


class _TeacherShim:
    """Carries exactly the attributes `_draw_parts_counts` / `_is_stable` read."""

    def __init__(self, copy_prob, len_keep, prob_invalid, max_eff, device):
        import torch
        n = 8
        self._base_target = torch.tensor([2 if i in set(CANONICAL_SLOTS) else 0 for i in range(n)],
                                         dtype=torch.long, device=device)
        self._copy_prob, self._prob_invalid = copy_prob, prob_invalid
        # phase 5's method clamps to _max_eff, phase 3's (root) to _max_len -- set both so the shim
        # works under whichever branch's CodesignAgent is imported.
        self._max_eff = self._max_len = max_eff
        self._len_sigma = 0.5 / (math.sqrt(2) * torch.erfinv(torch.tensor(len_keep)).item())


def _make_teacher(cfg, max_eff, device):
    from transformer_rl.codesign_agent import CodesignAgent
    cd = cfg["params"]["config"]["generator"]
    sh = _TeacherShim(float(cd.get("copy_prob", 0.6)),
                      float(cd.get("len_keep_prob", 0.6)), float(cd.get("prob_invalid", 0.1)),
                      max_eff, device)
    # bind the REAL implementations -- never a copy
    sh._is_stable = CodesignAgent._is_stable.__get__(sh)
    sh._draw_parts_counts = CodesignAgent._draw_parts_counts.__get__(sh)
    return sh


def _stats(bodies, label):
    le = limb_entropies(bodies)
    H = le["H_limb"]
    # C = sum_n H(L_n) - H(B); H(B) here is the PLUG-IN joint (valid: the skeleton space is small
    # enough at these commitment levels, and both populations are estimated identically, so the
    # comparison is fair even where the absolute value is biased low.
    from collections import Counter
    keys = [tuple(tuple(l) if l else None for l in b) for b in bodies]
    c = np.array(list(Counter(keys).values()), dtype=float)
    p = c / c.sum()
    h_joint = float(-(p * np.log(p)).sum() + (len(c) - 1) / (2.0 * len(keys)))
    C = float(H.sum() - h_joint)
    return {"label": label, "N_limb": le["N_limb_mean"], "sumH": float(H.sum()),
            "frac": C / max(float(H.sum()), 1e-9), "n_distinct": len(set(keys))}


def _cross_within(a, b):
    """(mean within-a, mean within-b, mean cross) d_struct. Subsampled for the O(n^2) matrices."""
    k = min(len(a), len(b), 1500)
    ca, cb = encode_population(a[:k]), encode_population(b[:k])
    ml = max(ca.shape[1], cb.shape[1])
    pad = lambda x: np.pad(x, ((0, 0), (0, ml - x.shape[1])))
    ca, cb = pad(ca), pad(cb)
    wa = pairwise_d_struct(ca).mean()
    wb = pairwise_d_struct(cb).mean()
    cr = (ca[:, None, :] != cb[None, :, :]).sum(-1).mean()
    return float(wa), float(wb), float(cr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="phase3_s42,phase5_s42,21-09-23-06,21-12-02-14")
    ap.add_argument("--n", type=int, default=4096)
    ap.add_argument("--pre-epoch", type=int, default=550, help="end-of-pretrain checkpoint")
    args = ap.parse_args()

    import torch
    import yaml
    device = torch.device("cuda:0")

    for name in args.runs.split(","):
        run_dir = RUN_DIR / name
        cfg = yaml.safe_load((run_dir / "config.yaml").read_text())
        phase = "phase3" if name.startswith("phase3") else "phase5"
        teach = _make_teacher(cfg, MAX_EFF[phase], device)
        t_counts = teach._draw_parts_counts(args.n).cpu().numpy()
        t_bodies = [counts_to_repr(c) for c in t_counts]

        print(f"\n=== {name} ({phase}, teacher=parts) ===")
        rows = [_stats(t_bodies, "teacher")]

        ck = dict(sorted((int(m.group(1)), p) for p in (run_dir / "nn").glob("*_ep_*.pth")
                         if (m := re.search(r"_ep_(\d+)", p.name))))
        gens = {}
        if ck:
            from experiments.ppg_parity import _load_policy
            from task_envs.codesign_environment import _OBS_TOTAL as OB, _N_DOFS_FULL as NA
            for tag, ep in (("gen@pre", min(ck, key=lambda e: abs(e - args.pre_epoch))),
                            ("gen@end", max(ck))):
                try:
                    net, _ = _load_policy(ck[ep], cfg["params"]["network"], device,
                                          value_size=int(cfg.get("env", {}).get("value_size", 1)),
                                          obs_base=OB, n_act=NA)
                except (RuntimeError, KeyError):
                    print(f"  (cannot load {tag} ep{ep}: architecture mismatch -- wrong branch)")
                    break
                with torch.no_grad():
                    out = net.net.sample(args.n)
                cnt = out["counts"].cpu().numpy().astype(int)
                gens[tag] = ([counts_to_repr(c) for c in cnt] if "eff_sub" not in out else
                             population_to_repr(cnt, out["eff_sub"].cpu().numpy(),
                                                out["cap_sub"].cpu().numpy(),
                                                collapse_subtypes=True))
                rows.append(_stats(gens[tag], f"{tag} (ep{ep})"))
                del net; torch.cuda.empty_cache()

        print(f"  {'population':18}{'N_limb':>9}{'sumH':>9}{'C/sumH':>9}{'n_distinct':>12}")
        for r in rows:
            print(f"  {r['label']:18}{r['N_limb']:>9.2f}{r['sumH']:>9.2f}"
                  f"{r['frac']:>9.3f}{r['n_distinct']:>12d}")
        if "gen@pre" in gens:
            wa, wb, cr = _cross_within(t_bodies, gens["gen@pre"])
            verdict = ("cross ~= within => BC TRANSFERRED the distribution"
                       if cr <= 1.15 * max(wa, wb) else
                       "cross >> within => BC did NOT transfer: populations sit in different regions")
            print(f"  d_struct: within-teacher={wa:.2f}  within-gen@pre={wb:.2f}  CROSS={cr:.2f}"
                  f"   ({verdict})")


if __name__ == "__main__":
    main()
