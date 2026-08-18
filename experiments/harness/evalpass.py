"""One eval pass: open a task, load a checkpoint's policy, install bodies, roll out at mu.

Lifted out of `scripts/eval.py`, which now imports it (ADR-0021). The ladder, the specialization pass
and `eval.py` all need the *same* rollout — a second copy would be the first violation of the
harness's one-implementation-per-metric rule, and it is the rollout that defines what "return" means
for every metric in the series.

Nothing here trains or writes: it builds a sim, runs deterministic actions through it, and returns
per-body arrays. `open_task` is the only expensive call (a VSim build); callers amortise it over many
checkpoints, as `eval.py` does over its runs.
"""
import numpy as np
import torch
from tqdm import tqdm

from experiments.harness.policy import _load_policy
from transformer_rl.models import _raw_tail
from transformer_rl.morphology import designs_from_arrays, seed_body
from transformer_rl.train_utils import _resolve_task

EVAL_SEED = 123          # reproducible generator draws across runs/epochs


# ---- setup ------------------------------------------------------------------------

def open_task(cfg: dict, n_envs: int, *, device, seed: int = EVAL_SEED):
    """(env, library, layout) for a run's stamped config, sized to `n_envs` bodies, one body per env.

    Task and library come from the run's OWN stamp, never from a default: a checkpoint evaluated
    against a different task reads the wrong obs offsets and fails silently rather than loudly. Runs
    stamped before `env.task` existed are ant runs by construction, hence the fallback.

    Every env starts on the canonical seed body; callers rebuild onto their own bodies with
    `install`/`install_design`. The scene is built once and resampled, because the build is the
    expensive part.
    """
    import vlearn as v
    from codesigner.components.modular_libraries import REGISTRY as ML_REGISTRY

    library = ML_REGISTRY[cfg.get("env", {}).get("module_library", "simple")]()
    _, task_class = _resolve_task(cfg, default="ant")
    env = task_class(device=device, rendering=False, raise_exception=False, with_window=False,
                     enable_scene_query=False, rootOffset=(v.Vec3(0, 0, 0), v.Quat(0, 0, 0, 1)))
    n_envs = env.setup(library, n_envs, n_envs, [seed_body(library)] * n_envs, seed=seed)
    return env, library, env.obs_layout()


def load_net(ckpt, cfg: dict, layout: dict, device):
    """(net, obs_norm) for a checkpoint, sized from the Task's published `obs_layout` (D23) rather
    than from anything this side recomputes."""
    return _load_policy(ckpt, cfg["params"]["network"], device,
                        value_size=int(cfg.get("env", {}).get("value_size", 1)),
                        obs_base=layout["obs_total"],
                        n_act=layout["n_modules"] + layout["n_root_axes"])


# ---- bodies -----------------------------------------------------------------------

def install(env, out):
    """Rebuild the scene onto the generator's designed bodies. The Task takes Morphologies now, so
    the design grid is translated on this side of the boundary."""
    env.resample(designs_from_arrays(env.module_library, out["counts"].long(),
                                     out["eff_sub"], out["cap_sub"]))


def install_design(env, counts, eff_sub, cap_sub):
    """Rebuild the scene onto ONE design, replicated across every env. The across-env spread is then
    pure measurement noise — the in-band noise floor every ladder level is read against."""
    n = env.total_num_envs
    tile = lambda t: t.reshape(1, *t.shape).expand(n, *t.shape).contiguous()
    install(env, {"counts": tile(counts), "eff_sub": tile(eff_sub), "cap_sub": tile(cap_sub)})


def sample_bodies(net, n: int, mode: str, *, seed: int = EVAL_SEED):
    """`net.sample` with the draw seeded, so two runs' populations are comparable draw-for-draw.
    Modes (all walk the identical grammar-masked MDP): `stochastic` the trained distribution,
    `greedy` its argmax, `uniform` a random policy on the same grammar."""
    torch.manual_seed(seed)
    return net.net.sample(n, mode=mode)


# ---- rollout ----------------------------------------------------------------------

@torch.no_grad()
def forward(net, obs_norm, obs):
    """Deterministic control step: normalized obs (raw {0,1} tail restored) -> (mu-clamped, V0.98)."""
    normed = obs_norm(obs).clone()
    t0, td = _raw_tail(net.net)                              # restore raw DOF-mask/type tail
    normed[..., t0:t0 + td] = obs[..., t0:t0 + td]
    mu, _, value, _ = net({"obs": normed})
    value = value.squeeze(-1) if value is not None else None
    return mu.clamp(-1.0, 1.0), value


@torch.no_grad()
def rollout(net, obs_norm, env, device, *, episodes, label=""):
    """K episodes per env on the currently-installed (fixed) bodies. EPM==1 => per-env == per-body.
    Returns per-body arrays: return (mean over K), fall_rate (terminated vs truncated), ep_len, v0
    (mean V0.98 at episode starts). Bodies are held fixed -- resample() is NOT called here.
    `label` names the source (gen/best/rnd) in the tqdm progress bar."""
    n, L = env.total_num_envs, env.max_episode_length
    z = lambda: torch.zeros(n, device=device)
    ret_sum, term_sum, len_sum = z(), z(), z()
    v0_sum, v0_n = z(), z()
    cur, curlen = z(), z()
    ep_done = torch.zeros(n, device=device)
    at_s0 = torch.ones(n, dtype=torch.bool, device=device)  # first obs of every env is an episode start

    obs, _ = env.reset()
    cap = (episodes + 2) * L
    step = 0
    # bar tracks the slowest env's episode count (min ep_done); the loop exits when it hits budget.
    bar = tqdm(total=episodes, desc=f"[eval]   {label} rollout".rstrip(),
               unit="ep", leave=False, dynamic_ncols=True)
    while step < cap:
        mn = int(ep_done.min().item())                      # == loop-exit driver; one sync/step
        bar.update(mn - bar.n)
        bar.set_postfix_str(f"step {step}/{cap}", refresh=False)
        if mn >= episodes:
            break
        mu, value = forward(net, obs_norm, obs)
        collecting = ep_done < episodes
        if value is not None:
            take = at_s0 & collecting
            v0_sum += torch.where(take, value, torch.zeros_like(value))
            v0_n += take.float()
        obs, rew, term, trunc, _ = env.step(mu)
        rew = rew.squeeze(-1) if rew.ndim > 1 else rew
        done = term | trunc
        cur += torch.where(collecting, rew, torch.zeros_like(rew))
        curlen += collecting.float()
        finish = done & collecting                          # a within-budget episode just ended
        ret_sum += torch.where(finish, cur, torch.zeros_like(cur))
        len_sum += torch.where(finish, curlen, torch.zeros_like(curlen))
        term_sum += (term & finish).float()
        ep_done += done.float()
        cur = torch.where(done, torch.zeros_like(cur), cur)
        curlen = torch.where(done, torch.zeros_like(curlen), curlen)
        at_s0 = done
        step += 1

    bar.update(episodes - bar.n)                             # fill to full (covers cap-exit)
    bar.close()
    e = float(episodes)
    return {"return": (ret_sum / e).cpu().numpy(),
            "fall_rate": (term_sum / e).cpu().numpy(),
            "ep_len": (len_sum / e).cpu().numpy(),
            "v0": (v0_sum / v0_n.clamp(min=1)).cpu().numpy()}


# ---- designs ----------------------------------------------------------------------

def canonical(counts, eff_sub, cap_sub) -> tuple:
    """One design as a hashable value: per-slot (distal→proximal effector subtypes, cap subtype), in
    SLOT ORDER. Slots are compass directions, so two designs differing only by rotation are
    different bodies here and are not merged."""
    counts = np.asarray(counts, int)
    eff, cap = np.asarray(eff_sub, int), np.asarray(cap_sub, int)
    return tuple(None if counts[s] == 0 else (tuple(int(x) for x in eff[s, :counts[s]]), int(cap[s]))
                 for s in range(counts.shape[0]))


def modal_design(out) -> tuple[int, dict]:
    """(index, stats) of the most frequent design in a `net.sample` population.

    `greedy` is argmax at every step but the MDP visits growable limbs in a RANDOM order, so greedy
    draws are not identical — `eval.py` reports `best_n_unique` for exactly this reason. The modal
    design is therefore what "the generator's committed body" has to mean: one draw would be an
    arbitrary member of a family, and the mode is stable across the ordering noise.
    """
    counts = out["counts"].cpu().numpy().astype(int)
    eff, cap = out["eff_sub"].cpu().numpy(), out["cap_sub"].cpu().numpy()
    keys = [canonical(counts[i], eff[i], cap[i]) for i in range(counts.shape[0])]
    tally: dict[tuple, list[int]] = {}
    for i, k in enumerate(keys):
        tally.setdefault(k, []).append(i)
    top = max(tally.values(), key=len)
    return top[0], {"n_unique": len(tally), "modal_share": len(top) / len(keys),
                    "n_sampled": len(keys)}
