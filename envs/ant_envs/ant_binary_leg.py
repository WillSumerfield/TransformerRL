"""AntBinaryLegEnv: experiment-only variant for the value-gradient study.

All 8 legs are always physically present and DOF-mask-active. A leg's presence is carried by a
continuous **presence probability** `p ∈ [0,1]`, fed to the policy in BOTH the leg's hip and ankle
length obs slots. The physical body is built by **Bernoulli-sampling** `p` per leg: an ON sample
builds at 1x default, an OFF sample at a small `off_scale` (0.05x) so vsim stays non-degenerate (a 0x
link is massless and won't finalize). The OBSERVATION reports `p` (NOT the sampled 0/1 and NOT the
build length), so `V(p) ≈ E_{Bernoulli(p)}[return]` and `∂V/∂p` is a smooth, in-distribution codesign
signal. Hip and ankle of a leg share the same `p`.

Sampling (training / Step 1): pick a valid bias center `S` from the pick-pool, then draw each leg's
`p` independently — `p = √U` for on-legs (`∈ S`), `p = 1 − √U` for off-legs (`U ~ Uniform(0,1)`). The
equal mixture of √U and 1−√U is marginally Uniform(0,1), so the [0,1] knob is covered uniformly.
Bodies are Bernoulli-sampled from `p`; a **guard** rejects any sampled body in the held-out set and
**redraws `p`** (same `S`) until the body is admissible. Held-out = all 5-leg topologies + 3 curated
Step-2 sets; these are also stripped from the default pick-pool so a bias center is never held-out.

Pick-pool: stable bodies (>=3 legs, gap<=135 deg) most of the time, with an `unstable_fraction` tail
of marginal/unstable topologies (down to 2 legs, or gap>135) so the critic sees bad bodies too.

Eval modes:
  - `on_sets=...`     : deterministic curated bodies (Step 2). `p ∈ {0,1}` exactly = each on-set; no
                        sampling, no guard. obs slots hold 1.0 (on) / 0.0 (off).
  - `pick_pool=..., guard=False` : fixed bias-center set with √U sampling and guard off (Step 3:
                        5-leg interpolation, builds held-out 5-leg bodies on purpose).

See docs/value_gradient_propagation.md. NOT for production training.
"""
import numpy as np
import torch

from .ant_multimorph import AntMultiMorphEnv, _stable_morphologies, _OBS_BASE, _LEN_DIM, _N_LEGS
from .build_vsim import Morphology, DEFAULT_HIP, DEFAULT_ANKLE


def _by_legcount(sets) -> dict[int, list[frozenset]]:
    out: dict[int, list[frozenset]] = {}
    for s in sets:
        out.setdefault(len(s), []).append(frozenset(s))
    return out


def _all_topologies(min_on: int, max_on: int) -> list[frozenset]:
    out = []
    for mask in range(1, 256):
        active = frozenset(i + 1 for i in range(8) if (mask >> i) & 1)
        if min_on <= len(active) <= max_on:
            out.append(active)
    return out


# Held out from training: every 5-leg topology (leg-count interpolation, Step 3) + 3 curated Step-2
# sets (topology-specific generalization, Step 2). Their trained twins {3,4,5,8}/{3,4,5}/{2,4,6,8}
# are NOT held out. See docs/value_gradient_propagation.md.
CURATED_HELDOUT = [frozenset({2, 5, 6, 7}), frozenset({5, 6, 7}), frozenset({1, 3, 5, 7})]
HELDOUT = set(_all_topologies(5, 5)) | set(CURATED_HELDOUT)


class AntBinaryLegEnv(AntMultiMorphEnv):
    """All-8-leg ant where presence is a continuous probability `p`, body Bernoulli-sampled. See
    module docstring."""

    def __init__(
        self,
        num_envs,
        device,
        *,
        off_scale: float = 0.05,
        unstable_fraction: float = 0.25,
        min_on_stable: int = 3,
        min_on_unstable: int = 2,
        max_on: int = 8,
        max_gap_deg: float = 135.0,
        on_sets: list | None = None,
        on_p: float = 1.0,
        off_p: float = 0.0,
        pick_pool: list | None = None,
        guard: bool = True,
        **kwargs,
    ):
        self._off_scale = off_scale
        self._unstable_fraction = unstable_fraction
        self._off_hip = off_scale * DEFAULT_HIP
        self._off_ankle = off_scale * DEFAULT_ANKLE
        self._on_p = on_p                          # obs p reported for on/off legs in deterministic
        self._off_p = off_p                        # curated mode (body still built exactly = on-set)
        self._guard = guard
        self._held_out = HELDOUT
        self._np_rng = np.random.default_rng(kwargs.get("seed"))

        # Pick-pool for bias centers S. Explicit pick_pool (Step 3) is used verbatim; otherwise the
        # stable/unstable topology pools with the held-out set removed (so S is never held-out).
        if pick_pool is not None:
            self._pick_pool = [frozenset(s) for s in pick_pool]
            self._stable_by_n = self._unstable_by_n = None
        else:
            self._pick_pool = None
            stable = [s for s in _stable_morphologies(min_legs=min_on_stable, max_legs=max_on,
                                                      max_gap_deg=max_gap_deg) if s not in self._held_out]
            stable_set = set(stable)
            unstable = [s for s in _all_topologies(min_on_unstable, max_on)
                        if s not in stable_set and s not in self._held_out]
            self._stable_by_n = _by_legcount(stable)
            self._unstable_by_n = _by_legcount(unstable)
        self._on_sets: list[frozenset] = []
        self._p: np.ndarray | None = None       # (total, 8) continuous presence prob (sampled modes)

        if on_sets is not None:
            # Fixed curated on-sets (Step 2): one group per on-set, deterministic build, p in {0,1}.
            self._on_sets = [frozenset(s) for s in on_sets]
            kwargs["sample_morphs"] = False
            kwargs["morphologies"] = [self._binary_morph(s) for s in self._on_sets]
        else:
            kwargs.setdefault("sample_morphs", True)
        super().__init__(num_envs, device, **kwargs)

    def _binary_morph(self, on) -> Morphology:
        full = frozenset(range(1, _N_LEGS + 1))
        on = frozenset(on)
        hip = {n: (DEFAULT_HIP if n in on else self._off_hip) for n in full}
        ank = {n: (DEFAULT_ANKLE if n in on else self._off_ankle) for n in full}
        return Morphology(full, hip, ank)

    def _pick_center(self) -> frozenset:
        # Bias center S: from explicit pick_pool (Step 3), else leg-count-uniform then topology-uniform
        # within the stable (or unstable-tail) pool.
        if self._pick_pool is not None:
            return self._pick_pool[self._morph_rng.randrange(len(self._pick_pool))]
        use_unstable = self._unstable_by_n and self._morph_rng.random() < self._unstable_fraction
        pool = self._unstable_by_n if use_unstable else self._stable_by_n
        n = self._morph_rng.choice(sorted(pool))
        return self._morph_rng.choice(pool[n])

    def _draw_p(self, center_mask: np.ndarray) -> np.ndarray:
        # Per-leg p around S: √U on-legs, 1−√U off-legs. center_mask (..., 8) bool.
        root = np.sqrt(self._np_rng.random(center_mask.shape))
        return np.where(center_mask, root, 1.0 - root)

    def _draw_morphs(self, num: int) -> list:
        centers = [self._pick_center() for _ in range(num)]
        center_mask = np.zeros((num, _N_LEGS), dtype=bool)
        for e, s in enumerate(centers):
            for n in s:
                center_mask[e, n - 1] = True

        p = self._draw_p(center_mask)
        on = self._np_rng.random((num, _N_LEGS)) < p
        if self._guard:                                  # redraw p (same S) until body is admissible
            for e in range(num):
                while frozenset(i + 1 for i in range(_N_LEGS) if on[e, i]) in self._held_out:
                    p[e] = self._draw_p(center_mask[e])
                    on[e] = self._np_rng.random(_N_LEGS) < p[e]

        self._p = p.astype(np.float32)
        self._on_sets = [frozenset(i + 1 for i in range(_N_LEGS) if on[e, i]) for e in range(num)]
        return [self._binary_morph(s) for s in self._on_sets]

    def allocate_buffers(self):
        super().allocate_buffers()
        # Decouple obs from build: obs length slots report the presence probability p (both hip and
        # ankle slots), NOT the build geometry. _global_lengths feeds the obs length block every step
        # (compute_observations), so writing p here holds it for the whole rollout.
        EPM = self.envs_per_morph
        if self._sample_morphs:
            p_env = torch.as_tensor(self._p, dtype=torch.float32, device=self.device)  # (total, 8)
        else:                                            # curated: obs p = on_p (on) / off_p (off);
            p_env = torch.full((self.total_num_envs, _N_LEGS), self._off_p,   # body still built exactly
                               dtype=torch.float32, device=self.device)
            for gi, on in enumerate(self._on_sets):
                for n in on:
                    p_env[gi * EPM:(gi + 1) * EPM, n - 1] = self._on_p
        lvec = torch.cat([p_env, p_env], dim=1)          # both length slots hold p -> (total, 16)
        self._global_lengths[:] = lvec
        self._obs_buf[:, _OBS_BASE:_OBS_BASE + _LEN_DIM] = lvec
