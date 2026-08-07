"""Token CATEGORY vocabulary -- shared/global across every ModuleLibrary (CONTEXT.md "Type
embedding"). Which SUBTYPES fill the width below, and what each one physically is, is owned by the
active ModuleLibrary (task_envs/modular_libraries/) instead -- see
docs/adr/0016-modulelibrary-abstraction.md.

Two SEPARATE one-hots ride each token:
  category  {root, start, effector, cap}   -- splits phase-1's single `module` type
  subtype   shared index, width N_SUB      -- meaning is per-ModuleLibrary

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

# --- SUBTYPE one-hot width: SHARED index space, sized to the largest ModuleLibrary in use ------
N_SUB = 4
