# Both task-varying quantities ride the root token

The architecture's organising rule is **one token = one module = one DOF = one action scalar**, so
the obvious way to make the policy task-general was to give each of a Task's [root axes](../../CONTEXT-MAP.md)
its own token and let the shared `joint_head` decode it. We did the opposite: root-axis actions come
from a **multi-output head on the root/CLS token**, and a Task's extra observation block is
**concatenated into that same token's content**. The token stream is therefore identical on every
task — `[CLS] [start ×8] [module ×32]` — and nothing about masking, token roles, modes or the
generator's grammar changes when the task does.

## Why

The root token already read every root axis. `root_dim` is `13 + 3·n_root_axes` and carries pos,
velocity and last-action per axis; the tokenizer has always sliced the full width. The head lands on
the token that was already observing the state it drives, so no new information routing was needed —
only an output.

Keeping root axes **off** the token stream is what preserves the generator. Its design MDP is defined
over module slots: `gen_replay` stacks `L+1` prefixes for `L` module tokens, marginal-value advantage
is a difference of consecutive prefix values, and a limb's presence is derived from its effector
count. A root axis is fixed by the Task and never designed, so a root-axis token would have had to be
special-cased out of the frontier, out of the prefix stack, out of design mode and out of the
category grammar — four exceptions to buy a uniformity the generator does not want.

## Consequences

- **Action ordering is positional, not learned.** The head's outputs are concatenated *after* the
  module actions because the env's DOF buffer is `[padded module DOFs] ++ [root axes]`. Reorder
  either side and the policy drives the wrong joints, silently.
- **Root-axis actions are unmasked.** Root axes are constant per env and always active; only module
  actions are gated by the DOF mask, and `log_std` must be widened to `n_modules + n_root_axes` with
  ones appended to the mask.
- **The root token is now carrying a lot**: root state, the root-axis block, the scene description,
  the value aggregation, and up to six actions. Hammer's 24-D extra block is the stress case. If
  task performance ever looks capacity-bound at the root, splitting the scene out into its own token
  is the first thing to try — it was the rejected alternative here, and only on bandwidth grounds.
- **Checkpoints are task-locked**, which they already were: `embed_root`'s input width varies with
  `n_root_axes` before `extra_dim` enters. `policy_switch.filter_compatible` drops shape-incompatible
  runs in play mode.
- **Ant is parameter-identical.** With `n_root_axes == 0` and `extra_dim == 0` the head is not
  constructed and `embed_root` keeps its old width, so existing checkpoints load and every
  [ADR-0015](0015-phase-comparison-methodology.md) phase-comparison number stays comparable. This is
  a required invariant, not a happy accident — it is why both quantities are built conditionally
  rather than padded to a maximum width.
