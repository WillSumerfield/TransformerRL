"""Phase-5 (5a stages 1+2) module-type vocabulary — the single source of truth shared by the model
(token one-hots + constrained decoder), the agent (BC teacher), and the env/builder (physics).

Torch-free on purpose: envs/ant_envs/build_vsim.py imports it.

Two SEPARATE one-hots ride each token (CONTEXT.md "Type embedding"):
  category  {root, start, effector, cap}   -- splits phase-1's single `module` type
  subtype   shared index, width 4          -- effector 0..2, cap 0..3, root/start/pad = null (zeros)

GenAct is FACTORED, not flat: a category choice {effector, cap} (positionally grammar-masked), then
a subtype choice masked to that category's kinds. logp = logp(cat) + logp(sub | cat).
"""

# --- token CATEGORY one-hot (was _N_TYPE=3 with a single `module` kind) ----------------------
CAT_ROOT, CAT_START, CAT_EFFECTOR, CAT_CAP = 0, 1, 2, 3
N_CAT = 4
# Pad slots (deeper than a limb's cap) carry an ALL-ZERO category one-hot -- a fifth "null" kind
# that costs no width and stays distinguishable from a bare cap (whose subtype one-hot is [1,0,0,0]).

# --- GenAct category action ids: which KIND of module to emit at the chosen tip ---------------
GEN_EFF, GEN_CAP = 0, 1
N_GEN_CAT = 2

# --- SUBTYPE one-hot: SHARED index space, width = max per category ----------------------------
# effector = (local joint axis, limits, fixed default length). Axes are given in the LIMB-LOCAL
# frame (x = along-limb outward, y = tangent, z = up); build_vsim rotates them by _CYL_ROT[n], so
# effector types are position-independent (verified: _ANKLE_AXIS[n] == R_z(theta_n) . (0,1,0)).
EFF_SWING, EFF_KNEE, EFF_TWIST = 0, 1, 2
N_EFF = 3
# cap = passive terminal module. bare == today's zero-morphology stop (contact stays on the last
# effector); the other three are real terminal BODIES (contact moves to the cap). Caps never
# actuate -- no DOF, no action -- so effector <-> DOF stays 1:1.
CAP_BARE, CAP_FOOT, CAP_PAD, CAP_BALL = 0, 1, 2, 3
N_CAP = 4
N_SUB = 4                       # max(N_EFF, N_CAP) -- the shared index width

EFF_NAMES = ('swing', 'knee', 'twist')
CAP_NAMES = ('bare', 'foot', 'pad', 'ball')


def canonical_eff(depth0: int) -> int:
    """Phase-1-equivalent effector type at 0-based depth: swing proximal, knee distal. With the
    canonical cap (bare), a canonical design reproduces the phase-1 body EXACTLY."""
    return EFF_SWING if depth0 == 0 else EFF_KNEE


CANON_CAP = CAP_BARE
