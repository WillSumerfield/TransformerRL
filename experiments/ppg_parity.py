"""PPG vs PPO parity comparison on the full ant (variable sampled morphs).

All-in-one: (1) trains 3 seeds each of PPG (ppg_continuous) and PPO (a2c_continuous)
at the full config with synced timers, (2) aggregates over-time curves from the TB
summaries, (3) measures final-inference return via deterministic-mu rollout on ALL
stable bodies (same fixed set for every checkpoint). Report-only; the notebook
(notebooks/ppg_parity.ipynb) charts the results.

Always retrains (deletes the named run dir first). Outputs:
  data/ppg_parity/curves.npz     per-run scalar curves (step + value per tag)
  data/ppg_parity/inference.npz  per-body deterministic return, (n_seeds, n_bodies)/algo

Full run:  python experiments/ppg_parity.py
Dry run:   python experiments/ppg_parity.py --seeds 42 --max-epochs 3 --num-envs 256
"""
import os; os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
import sys
import shutil
import argparse
import subprocess
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT.parent / "vlearn-main" / "train"))

import numpy as np
import torch
import yaml
from rl_games.algos_torch.running_mean_std import RunningMeanStd
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from transformer_rl.models import MultiMorphLegTransformerBuilder
from transformer_rl.tokenize import OBS_DIM_8, LEN_DIM_8, MASK_DIM_8
from envs.ant_envs.ant_multimorph import AntMultiMorphEnv, _stable_morphologies

_OBS_TOTAL = OBS_DIM_8 + LEN_DIM_8 + MASK_DIM_8   # 139
_MASK_DIM  = MASK_DIM_8                            # 16
_N_ACT     = 16
ENVS_PER   = 128                                  # eval envs per body (averaged)
EVAL_SEED  = 123

ALGOS = {
    "ppo": dict(script="scripts/train_ant_full.py", config="configs/ppo_ant_full.yaml",
                run="runs/ant_full/full_transformer/{name}", ckpt="ant_full_transformer.pth"),
    "ppg": dict(script="scripts/train_ant_ppg.py",  config="configs/ppo_ant_ppg.yaml",
                run="runs/ant_ppg/ppg_transformer/{name}",  ckpt="ant_ppg_transformer.pth"),
}
# tags pulled from TB if present (PPG-only ones simply absent for PPO)
CURVE_TAGS = [
    "morph_reward/mean", "losses/a_loss", "losses/c_loss",
    "losses/aux_value", "losses/aux_clone_kl", "losses/aux_value_net",
    "perf/t_rollout", "perf/t_update", "perf/t_policy", "perf/t_value", "perf/t_aux",
]
DATA_DIR = _ROOT / "data" / "ppg_parity"


# ---- 1. training -----------------------------------------------------------------

def train_all(seeds, max_epochs, num_envs):
    for seed in seeds:
        name = f"s{seed}"
        for algo, spec in ALGOS.items():
            run_dir = _ROOT / spec["run"].format(name=name)
            if run_dir.exists():
                shutil.rmtree(run_dir)        # always retrain (only this exact named dir)
            cmd = [sys.executable, str(_ROOT / spec["script"]), "train",
                   "--headless", "True", "--seed", str(seed), "--name", name, "--timing"]
            if max_epochs is not None:
                cmd += ["--max_epochs", str(max_epochs)]
            if num_envs is not None:
                cmd += ["--num_envs", str(num_envs)]
            print(f"\n[parity] TRAIN {algo} {name}: {' '.join(cmd)}", flush=True)
            subprocess.run(cmd, check=True, cwd=_ROOT)


# ---- 2. curve aggregation (TB summaries) -----------------------------------------

def _event_file(run_dir: Path) -> Path | None:
    files = sorted((run_dir / "summaries").glob("events.out.tfevents.*"))
    return files[-1] if files else None

def aggregate_curves(seeds):
    out = {}
    for seed in seeds:
        name = f"s{seed}"
        for algo, spec in ALGOS.items():
            ev = _event_file(_ROOT / spec["run"].format(name=name))
            if ev is None:
                print(f"[parity] WARN no summaries for {algo}/{name}"); continue
            ea = EventAccumulator(str(ev)); ea.Reload()
            have = set(ea.Tags()["scalars"])
            for tag in CURVE_TAGS:
                if tag not in have:
                    continue
                sc = ea.Scalars(tag)
                key = f"{algo}_{name}__{tag.replace('/', '_')}"
                out[key + "__step"] = np.array([s.step for s in sc], dtype=np.int64)
                out[key + "__val"] = np.array([s.value for s in sc], dtype=np.float32)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(DATA_DIR / "curves.npz",
                        seeds=np.array(seeds), algos=np.array(list(ALGOS)), **out)
    print(f"[parity] curves -> {DATA_DIR / 'curves.npz'} ({len(out)//2} series)")


# ---- 3. final-inference return (controlled clean bodies) -------------------------

def _load_policy(ckpt: Path, net_params: dict, device):
    net = MultiMorphLegTransformerBuilder.Network(
        params=net_params, actions_num=_N_ACT, input_shape=(_OBS_TOTAL,), num_seqs=1)
    raw = torch.load(ckpt, map_location=device, weights_only=False)["model"]
    state = {k.removeprefix("_orig_mod."): v for k, v in raw.items()}
    sub = lambda p: {k.removeprefix(p): v for k, v in state.items() if k.startswith(p)}
    net.load_state_dict(sub("a2c_network."))
    obs_norm = RunningMeanStd(_OBS_TOTAL).to(device)
    obs_norm.load_state_dict(sub("running_mean_std."))
    net.to(device).eval(); obs_norm.eval()
    return net, obs_norm

@torch.no_grad()
def _rollout_return(net, obs_norm, env, device):
    n, L = env.total_num_envs, env.max_episode_length
    ep = torch.zeros(n, device=device)
    done = torch.zeros(n, dtype=torch.bool, device=device)
    obs, _ = env.reset()
    for _ in range(L):
        normed = obs_norm(obs).clone()
        normed[..., -_MASK_DIM:] = obs[..., -_MASK_DIM:]   # mask tail raw (TransformerMaskedNorm)
        mu, _, _, _ = net({"obs": normed})
        obs, rew, term, trunc, _ = env.step(mu.clamp(-1.0, 1.0))
        rew = rew.squeeze(-1) if rew.ndim > 1 else rew
        ep += rew * (~done).float()
        done |= (term | trunc)
        if bool(done.all()):
            break
    return ep.cpu().numpy()

def run_inference(seeds):
    assert torch.cuda.is_available()
    device = torch.device("cuda:0")
    bodies = _stable_morphologies()
    bcount = np.array([len(b) for b in bodies])
    env = AntMultiMorphEnv(len(bodies) * ENVS_PER, device, morphologies=bodies,
                           sample_morphs=False, rendering=False, raise_exception=False,
                           seed=EVAL_SEED, with_window=False)
    EPM = env.envs_per_morph

    results = {}
    for algo, spec in ALGOS.items():
        net_params = yaml.safe_load((_ROOT / spec["config"]).read_text())["params"]["network"]
        per_seed = []
        for seed in seeds:
            ckpt = _ROOT / spec["run"].format(name=f"s{seed}") / "nn" / spec["ckpt"]
            if not ckpt.exists():
                print(f"[parity] SKIP inference {algo}/s{seed}: missing {ckpt}"); continue
            net, obs_norm = _load_policy(ckpt, net_params, device)
            ep = _rollout_return(net, obs_norm, env, device)
            per_seed.append([ep[i * EPM:(i + 1) * EPM].mean() for i in range(len(bodies))])
            print(f"[parity] inference {algo}/s{seed}: mean return {np.mean(per_seed[-1]):.1f}")
        if per_seed:
            results[f"ret_{algo}"] = np.array(per_seed, dtype=np.float32)  # (n_seeds, n_bodies)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(DATA_DIR / "inference.npz",
                        bodies_legcount=bcount, seeds=np.array(seeds), envs_per_body=ENVS_PER,
                        **results)
    print(f"[parity] inference -> {DATA_DIR / 'inference.npz'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    ap.add_argument("--max-epochs", type=int, default=None, help="override per-run epochs (dry run)")
    ap.add_argument("--num-envs", type=int, default=None, help="override per-run num_actors (dry run)")
    ap.add_argument("--skip-train", action="store_true", help="aggregate + infer on existing runs only")
    args = ap.parse_args()

    if not args.skip_train:
        train_all(args.seeds, args.max_epochs, args.num_envs)
    aggregate_curves(args.seeds)
    run_inference(args.seeds)


if __name__ == "__main__":
    main()
