"""Q3: does the Phase-5 SUBTYPE axis carry reward signal under the trained controller?

At a FIXED skeleton (the generator's modal count vector), enumerate subtype variants and roll the
trained control policy (deterministic mu). Separately evaluate ~32 DIFFERENT skeletons at canonical
subtypes to get the scale reference. Deliverable = subtype-spread / skeleton-spread in return.

Read-only w.r.t. the algorithm: borrows _load_policy/_rollout_return from experiments/ppg_parity.py.

  uv run python experiments/vocab_ablation.py --seeds 42 43 44
"""
import os; os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
import re
import sys
import gc
import json
import argparse
import random
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT.parent / "vlearn-main" / "train"))

import numpy as np
from codesigner.components.interfaces import ModuleType

RUN = "runs/ant_codesign/codesign_single_transformer/phase5_s{seed}"
N_SAMPLE = 4096          # bodies drawn from the converged generator (to find the modal skeleton)
EVAL_SEED = 123
N_DRAW = 8               # random per-limb subtype draws per mixed contrast


def _final_ckpt(run_dir: Path):
    nn = run_dir / "nn"
    if not nn.exists():
        return None
    epoched = [(int(m.group(1)), p) for p in nn.glob("*_ep_*.pth")
               if (m := re.search(r"_ep_(\d+)", p.name))]
    return max(epoched)[1] if epoched else None


# ---- variant construction -------------------------------------------------------

def _variants(counts, rng, ml):
    """Named subtype variants at ONE fixed skeleton `counts` (8,) of per-limb effector counts.
    Returns [(name, contrast, effector_types dict, cap_types dict)]. ml = the env's ModuleLibrary
    (subtype names come from its public `names` API; "swing"/"knee"/"bare" below are OUR choice of
    canonical body, matching AntCodesignEnv._BASE_MORPHOLOGY's own literal vocabulary, not library
    data)."""
    eff_names = ml.names(ModuleType.EFFECTOR)
    SWING, KNEE, TWIST = eff_names
    limbs = {i + 1: int(k) for i, k in enumerate(counts) if k > 0}
    BARE = "bare"

    EFF = {
        "canon":      lambda k: (["swing"] + ["knee"] * (k - 1)),   # swing then knees
        "canon_twist": lambda k: [SWING] + [TWIST] * (k - 1),   # len-matched vs canon
        "all_knee":   lambda k: [KNEE] * k,
        "all_twist":  lambda k: [TWIST] * k,                        # len-matched vs all_knee
        "all_swing":  lambda k: [SWING] * k,                        # LENGTH-CONFOUNDED
    }
    caps = list(ml.names(ModuleType.CAP))
    out = []

    def add(name, contrast, eff_key, cap_fn):
        et = {n: EFF[eff_key](k) for n, k in limbs.items()}
        out.append((name, contrast, et, {n: cap_fn(n) for n in limbs}))

    # (a) CAP axis -- cleanest single variable (cap masses equalized). Uniform cap, canonical eff.
    for c in caps:
        add(f"cap_{c}", "cap", "canon", lambda n, c=c: c)
    # cap axis re-run under an all-knee body, to check the cap effect isn't canon-specific
    for c in caps[1:]:
        add(f"cap_{c}@knee", "cap_knee", "all_knee", lambda n, c=c: c)
    # mixed caps: per-limb random draw, canonical effectors -> isolates the CAP sub-axis under
    # realistic (non-uniform) assignments.
    for j in range(N_DRAW):
        draw = {n: rng.choice(caps) for n in limbs}
        add(f"cap_mixed{j}", "cap_mixed", "canon", lambda n, d=draw: d[n])

    # (b) EFFECTOR, LENGTH-MATCHED (knee <-> twist): same module lengths, different joint axis.
    add("eff_all_knee", "eff_matched", "all_knee", lambda n: BARE)
    add("eff_all_twist", "eff_matched", "all_twist", lambda n: BARE)
    add("eff_canon_twist", "eff_matched", "canon_twist", lambda n: BARE)
    # canonical/bare (== cap_bare above) is the 4th member of this contrast; reused, not rebuilt.

    # (c) EFFECTOR, LENGTH-CONFOUNDED (swing <-> knee): 2.24x module-length change.
    add("eff_all_swing", "eff_confounded", "all_swing", lambda n: BARE)

    # (d) FAIRNESS CONTROL. The uniform variants above set ALL limbs alike -- something the
    # generator never emits, so a collapse there may be off-distribution rather than real axis
    # signal. Grade the knee->twist substitution over a FRACTION of limbs, and add fully
    # per-limb-per-depth random subtypes (the regime the controller actually trained on).
    ordered = sorted(limbs)
    for f in (1, 2, 4):
        if f >= len(ordered):
            continue
        sw = set(ordered[:f])
        et = {n: ([SWING] + [TWIST] * (k - 1) if n in sw
                  else ["swing"] + ["knee"] * (k - 1)) for n, k in limbs.items()}
        out.append((f"eff_twist_{f}limb", "eff_graded", et, {n: BARE for n in limbs}))
    rnd_eff = lambda k: [rng.choice(eff_names) for _ in range(k)]
    for j in range(N_DRAW):     # EFFECTOR sub-axis alone (caps held bare)
        out.append((f"eff_mixed{j}", "eff_mixed", {n: rnd_eff(k) for n, k in limbs.items()},
                    {n: BARE for n in limbs}))
    for j in range(N_DRAW):     # BOTH sub-axes -- the full in-distribution subtype spread
        out.append((f"both_mixed{j}", "both_mixed", {n: rnd_eff(k) for n, k in limbs.items()},
                    {n: rng.choice(caps) for n in limbs}))
    return out


# ---- main -----------------------------------------------------------------------

def _probe_env_dims(device):
    """One throwaway 1-env AntCodesignEnv, just to read the (obs_total, n_act) layout -- these are
    computed at construction from module_library.root_axes, not stale hardcoded constants."""
    import vlearn as v
    from task_envs.ant_envs.ant_codesign import AntCodesignEnv
    env = AntCodesignEnv(1, device, rendering=False, with_window=False, raise_exception=False, seed=0)
    dims = (env.observation_space.shape[0], env.action_space.shape[0])
    env.gym = None; del env; gc.collect(); v.delete_gym()
    return dims


def run_seed(seed, epm, n_skel, device, ml, obs_base, n_act):
    import torch
    import yaml
    import vlearn as v
    from experiments.ppg_parity import _load_policy, _rollout_return
    from task_envs.ant_envs.ant_codesign import AntCodesignEnv
    from task_envs.modular_libraries.simple import Morphology

    run_dir = _ROOT / RUN.format(seed=seed)
    ckpt = _final_ckpt(run_dir)
    assert ckpt is not None, f"no checkpoint in {run_dir}"
    cfg = yaml.safe_load((run_dir / "config.yaml").read_text())   # each run's OWN config
    value_size = int(cfg.get("env", {}).get("value_size", 1))
    net_params = cfg["params"]["network"]
    net, obs_norm = _load_policy(ckpt, net_params, device, value_size=value_size,
                                 obs_base=obs_base, n_act=n_act)
    print(f"[vocab] s{seed} ckpt={ckpt.name}", flush=True)

    torch.manual_seed(EVAL_SEED)          # reproducible modal skeleton across reruns
    with torch.no_grad():
        smp = net.net.sample(N_SAMPLE)
    mods = smp["counts"].cpu().numpy().astype(int)
    pats, freq = np.unique(mods, axis=0, return_counts=True)
    order = np.argsort(freq)[::-1]
    modal = pats[order[0]]
    print(f"[vocab] s{seed} modal skeleton = {modal.tolist()} "
          f"({freq[order[0]]}/{N_SAMPLE} = {freq[order[0]]/N_SAMPLE:.1%}), "
          f"{len(pats)} distinct", flush=True)
    # what subtypes does the generator actually emit? (context for the entropy claim)
    es, cs = smp["eff_sub"].cpu().numpy(), smp["cap_sub"].cpu().numpy()
    n_eff = len(ml.names(ModuleType.EFFECTOR))
    n_cap = len(ml.names(ModuleType.CAP))
    eff_hist = np.bincount(es[es >= 0].ravel(), minlength=n_eff) / max(1, (es >= 0).sum())
    cap_hist = np.bincount(cs[cs >= 0].ravel(), minlength=n_cap) / max(1, (cs >= 0).sum())

    rng = random.Random(EVAL_SEED + seed)
    variants = _variants(modal, rng, ml)
    skels = [pats[i] for i in order[:n_skel]]                    # in-distribution skeleton spread

    bodies, labels = [], []
    for name, contrast, et, ct in variants:
        bodies.append(Morphology.from_design(ml, et, ct)); labels.append((name, contrast))
    for j, s in enumerate(skels):
        counts = {i + 1: int(k) for i, k in enumerate(s) if k > 0}
        et = {n: (["swing"] + ["knee"] * (k - 1)) for n, k in counts.items()}
        bodies.append(Morphology.from_design(ml, et, {}))
        labels.append((f"skel{j}", "skeleton"))
    # skel0 == modal skeleton at canonical subtypes == variant "cap_bare": an intentional duplicate
    # body evaluated twice -> a direct read of the rollout noise floor.

    print(f"[vocab] s{seed} building {len(bodies)} bodies x {epm} envs", flush=True)
    env = AntCodesignEnv(len(bodies) * epm, device, morphologies=bodies,
                           rendering=False, raise_exception=False, seed=EVAL_SEED,
                           with_window=False, value_size=value_size)
    e = env.envs_per_morph
    ep = _rollout_return(net, obs_norm, env, device)

    rows = []
    for i, (name, contrast) in enumerate(labels):
        x = ep[i * e:(i + 1) * e]
        rows.append(dict(name=name, contrast=contrast, mean=float(x.mean()),
                         std=float(x.std()), sem=float(x.std() / np.sqrt(len(x)))))

    torch.cuda.synchronize()
    for _fn in ("end_streaming", "_check_for_cuda_errors"):
        try: getattr(env.gym, _fn)()
        except Exception: pass
    env.gym = None; del env; gc.collect()
    v.delete_gym()

    return dict(seed=seed, ckpt=ckpt.name, modal=modal.tolist(), epm=e, rows=rows,
                eff_hist=eff_hist.tolist(), cap_hist=cap_hist.tolist(),
                n_distinct=int(len(pats)), modal_frac=float(freq[order[0]] / N_SAMPLE))


def summarize(res):
    rows = res["rows"]
    by = {}
    for r in rows:
        by.setdefault(r["contrast"], []).append(r)
    get = lambda n: next(r for r in rows if r["name"] == n)
    # contrast members (cap_bare doubles as the canonical member of the effector contrasts)
    groups = {
        "cap": [r["mean"] for r in by["cap"]],
        "cap_knee": [get("eff_all_knee")["mean"]] + [r["mean"] for r in by["cap_knee"]],
        "eff_matched": [get("cap_bare")["mean"]] + [r["mean"] for r in by["eff_matched"]],
        "eff_confounded": [get("eff_all_knee")["mean"], get("eff_all_swing")["mean"]],
        "eff_graded": [get("cap_bare")["mean"]] + [r["mean"] for r in by.get("eff_graded", [])],
        "cap_mixed": [r["mean"] for r in by.get("cap_mixed", [])],
        "eff_mixed": [r["mean"] for r in by.get("eff_mixed", [])],
        "both_mixed": [r["mean"] for r in by.get("both_mixed", [])],
        "skeleton": [r["mean"] for r in by["skeleton"]],
    }
    spread = {k: dict(std=float(np.std(v)), rng=float(np.ptp(v)), n=len(v))
              for k, v in groups.items()}
    res["spread"] = spread
    res["noise"] = dict(dup_diff=abs(get("cap_bare")["mean"] - get("skel0")["mean"]),
                        median_sem=float(np.median([r["sem"] for r in rows])))
    return res


def main():
    import torch
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    ap.add_argument("--epm", type=int, default=128)
    ap.add_argument("--n-skel", type=int, default=32)
    a = ap.parse_args()
    assert torch.cuda.is_available()
    device = torch.device("cuda:0")
    from task_envs.ant_envs.ant_codesign import AntCodesignEnv
    ml = AntCodesignEnv._MODULE_LIBRARY
    obs_base, n_act = _probe_env_dims(device)
    out = []
    for s in a.seeds:
        out.append(summarize(run_seed(s, a.epm, a.n_skel, device, ml, obs_base, n_act)))
        sp = out[-1]["spread"]
        print(f"[vocab] s{s} spread std: " +
              " ".join(f"{k}={v['std']:.2f}" for k, v in sp.items()) +
              f" | ratio cap/skel={sp['cap']['std']/max(1e-9, sp['skeleton']['std']):.3f}",
              flush=True)
    dst = _ROOT / "temp" / "q3_subtype_reward.json"
    dst.parent.mkdir(exist_ok=True)
    dst.write_text(json.dumps(out, indent=1))
    print(f"[vocab] -> {dst}")


if __name__ == "__main__":
    main()
