"""Token CATEGORY vocabulary -- shared/global across every ModuleLibrary (CONTEXT.md "Type
embedding"). Which SUBTYPES fill the subtype width, what each one physically is, and how wide the
index is are all owned by the active ModuleLibrary (`codesigner.components.modular_libraries`) --
see docs/adr/0016-modulelibrary-abstraction.md.

Two SEPARATE one-hots ride each token:
  category  {root, start, effector, cap}   -- splits phase-1's single `module` type
  subtype   shared index, width n_sub      -- meaning AND width are per-ModuleLibrary

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

# The SUBTYPE one-hot width is NOT here. It is `ModuleLibrary.subtype_width`, published by the Task
# as `obs_layout()["n_sub"]` and carried on the net as `n_sub`. A constant here would be a second
# opinion on how wide the library's own subtype block is; the obs block is sized by the library, so
# a disagreement misreads the tail rather than failing (D14, D23).
