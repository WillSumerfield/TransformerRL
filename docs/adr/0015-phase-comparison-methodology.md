---
status: accepted
---

# Phase-comparison methodology: the frozen per-phase artifact every Complexity-Escalation phase writes into

## Context & Decision

The Complexity Escalation plan runs 10 phases, each on its own branch, each changing the codesign
algorithm. Phase 0 builds the **comparison harness** the other nine write into: how each phase's
effect on runtime, performance, convergence, and morphology diversity is measured, stored, and
overlaid in one shared notebook. This is a contract — changing it later means re-running every
phase — so it is fixed here.

**Artifact.** Each phase's run writes one self-contained `data/phase_comparison/<phaseN_name>.npz`.
`data/` is **gitignored** (repo convention — all experiment outputs are scratch), so artifacts are
**local-only, never committed**; the notebook globs the local directory and overlays whatever phases
are present. It holds the phase's **best-config** run over a **fixed seed set of 5** (`[42..46]`),
storing **per-seed arrays** (needed for between-seed diversity + mode-overlap) plus mean±std. No
single-seed cherry-picking. **Regenerating a phase** = check out its commit and rerun
`phase_comparison.py`, which reproduces its local `.npz`. The per-phase *interpretation prose* lives
in the tracked notebook, so it travels across branches even though the raw arrays do not.

**Fair-comparison axis.** Held constant across phases: **total env-step budget**, env count, seed
set. Everything else is a measured *output* (runtime cannot be an input — it is a headline metric).
The budget is expressed in **env-steps, not epochs** (later phases change per-epoch cost), fixed at
**whatever 3000 epochs of the Phase-0 config comes out to** — that env-step count is the constant
every phase runs to.

**Metrics in every artifact:**
- **Performance** — eval-time **deterministic-mu control return** on the converged generator,
  reported three ways: (1) **top choice** — the argmax-likelihood ("best") body; (2) **distribution
  average** — mean over the generator's sampled bodies; (3) **top-K** — the K highest-performing
  generated bodies, each averaged across episodes to cut variance. The honest end-to-end "how good
  are the robots it designs," not the training-time `quality/R_mean` proxy. The sampled-body set
  behind (2) is also the population for the diversity metrics — one eval pass feeds both.
- **Runtime** — throughput (env-steps/sec) and peak GPU memory.
- **Convergence** — two env-step counts: **quality convergence** (`quality/R_mean` first reaches
  90% of its final-plateau mean) and **morphology convergence** (per-limb presence distribution
  stable, max |Δ on-rate| < ε over a window). "Controller got good" and "design settled" are
  distinct events.
- **Diversity** — three representation-agnostic measures, each **within-run** and **between-seed**,
  on a shared morphology representation (8 compass slots, each a distal→proximal module sequence):
  **`d_comp`** (module-histogram L1, bag-of-modules), **`d_struct`** (slot-matched sum of per-limb
  **tip-anchored** edit distances — limbs aligned at the distal tip, so `E-C` ≈ `E-E-C`), and
  **`N_modes`** (effective number of distinct designs = Hill number order q=1 over `d_struct`
  clusters; the Phase-8 branching signal). Defined in `experiments/CONTEXT.md`.

**Interpretation text** lives as markdown cells in the shared notebook (one per-phase section),
authored as each phase lands.

**Phase 0 itself** = the current presence-only ant codesign, run through this harness = the baseline
series every later phase is compared against. It generalizes `experiments/ant_codesign.py` +
`notebooks/ant_codesign.ipynb` into the phase-keyed harness.

## Considered Options

- **Single accumulating results file / re-run-all-from-reverted-commits.** Rejected: the former is
  merge-conflict-prone across branches and one bad write loses all phases; the latter means hours of
  retraining every notebook open. Frozen per-phase artifacts are phase-local and cheap to read.
- **Fixed wall-clock budget** or **train-each-to-its-own-convergence.** Rejected: wall-clock
  confounds algorithm quality with per-step cost; per-phase-to-convergence hinges on fragile
  auto-convergence detection. Fixed env-step budget keeps the learning signal comparable and makes
  runtime a clean output.
- **Headline `quality/R_mean` (training-time proxy).** Rejected: it is the generator's internal
  scaled reward, not a robot's realized return; the two can diverge. Eval-return is the honest
  number. (`R_mean` still charted as a secondary curve.)
- **Generator-entropy / Jensen-Shannon diversity.** Rejected as the diversity headline: entropy
  inflates independent-component flipping without breaking a common core (the Phase-8 concern).
  Distance-based `d_comp`/`d_struct` + Hill-number `N_modes` capture spread and multimodality
  without the entropy trap.
- **Root-aligned positional limb comparison.** Rejected: it misranks unequal-length limbs
  (`E-E-C` vs `E-C` look far when they are near). Limb distances align at the tip.
- **Per-phase interpretation as `.md` files or npz text fields.** Not chosen (notebook markdown
  cells preferred for in-place reading); accepted cost = `.ipynb`-JSON merge conflicts on rebase and
  that a reverted commit does not cleanly carry its own text.

## Consequences

- The `.npz` schema + metric definitions are a **contract**; every phase honors it, and changing it
  forces re-running all phases (the reason this is an ADR).
- **Artifacts are local scratch, not version-controlled** (`data/` is gitignored). The notebook
  overlays only phases present in the local `data/phase_comparison/`; a fresh clone shows nothing
  until phases are (re)run. Regeneration = check out the phase's commit and rerun the harness. Only
  code, the ADR, the glossary, and the notebook (incl. its interpretation prose) are committed.
- **Eval-return is only cross-comparable within one task.** Phases 7 (new tasks) and 10 (multi-task)
  break the single-return axis — the notebook must group performance by task there, not overlay
  raw returns. Runtime and diversity remain comparable across all phases.
- Runtime capture: codesign inherits `LoggingA2CAgent` phase timing (enable via config `timing`);
  **peak-memory logging may need porting** from `ppg_agent.py` into the shared logging path.
- Diversity knobs (overhang weight `w`, attribute weight `λ`, cluster τ, Hill order q) and the
  convergence thresholds (90%, ε) are **tunable after first runs**; defaults are recorded here and
  in `experiments/CONTEXT.md`.
