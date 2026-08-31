# Operationalizing spread / exploration / generalization in the joint-optimization toy

The joint-optimization toy (`experiments/joint_optimization/`) exists to measure how a designer's
and a controller's *search configuration* trade off against each other. That only means something if
"spread", "exploration", and "generalization" have mechanical definitions, and the whole result set
is defined by the ones we picked — changing any of them invalidates every number from all four
experiments. Recording them here because each was a real fork and each looks arbitrary from the code.

Both optimizers are the same object: a moving point `mu` that samples `x ~ N(mu, spread^2)`,
optionally improves each sample, evaluates, and steps toward a top-half rank-weighted recombination
of the good ones. `f` is a reward landscape and both optimizers **maximize** it.

**Spread** is the sampling std dev. It selects which hill a sample lands on, and nothing else.

**Exploration** `e` is `P(leave the sample raw)`; with probability `1-e` the sample climbs. Note this
is the *inverse* of a natural reading of "probability of climbing" — climbing is exploitation, so the
axis had to be flipped to make the name mean what it says.

**Generalization** `g` attenuates the climb: `w = exp(-||p - centre||^2 / 2g^2)` and
`x <- x + w * (target - x)`, so `g -> inf` lands exactly on target and `g -> 0` never moves. The
distance is measured in the **joint** (design, action) space for both optimizers. For the designer
the action term vanishes by construction, so it reduces to `|d - mu_d|`; for the controller it makes
competence decay on designs far from what it has recently been handed.

**Climb target** is the local peak of the hill the sample landed on — *not* the optimum of the whole
slice.

**The designer's slice** is `f(., mu_a)`, i.e. through the controller's current mean action, one
iteration stale.

## Consequences

Exploration **gates** generalization rather than sitting beside it: at `e = 1` nothing ever climbs, so
`g` has no effect and an entire face of the Experiment 4 hypercube is degenerate. Symmetrically, at
`g_c -> inf` the controller overwrites its own sample entirely and controller spread stops mattering.
Both are reported as findings, not smoothed away.

Because the update steps toward the sample cloud, larger spread produces larger steps. Spread and
step size are deliberately *not* decoupled — wide sampling genuinely buying faster coarse progress is
the trade-off Experiment 1 is about, and normalizing it out would leave that plot measuring nothing.
The recombination rate is therefore a single global constant across every experiment.

## Considered options

**Climbing toward the best point of the whole slice** instead of the local peak. Rejected: every
climbing sample converges to the same point, so the landscape's multi-scale structure is never felt
and spread only affects raw samples. Hill-local ascent is what makes spread (which hill),
generalization (how far up), and exploration (whether at all) read three genuinely different
features of the landscape.

**A gradient step with learning rate `eta`** as the climb. Rejected: `eta` is a seventh knob that
confounds directly with both spread and generalization, and one step reaches no peak, so
"perfectly general" has no limiting behaviour to anchor the axis.

**Measuring the controller's radius in action space**, `|a - mu_a|` — the literal reading of "further
points climb less". Rejected: the controller's radius would never interact with the designer's
spread, and the central hypothesis (the two optimizers need comparable radii) has almost no mechanism
by which to show up. Under joint distance, a designer ranging beyond its controller's radius gets its
good designs scored badly, which is the effect being measured.

**The designer climbing `max_a f(., a)`**, the oracle marginal. Rejected: it makes the designer's
model independent of how good the controller actually is, so the coupling survives only in
evaluation and not in the search itself.

**Orthogonalizing exploration and generalization** so no sweep cell is degenerate. Rejected: it
requires redefining "exploration" away from being a probability, and "generalization is worthless
without exploitation pressure" is a result worth showing rather than a defect to engineer around.
