# Static morphology response probe

This opt-in feasibility probe tests whether frozen contextual design tokens help
predict physical responses beyond local morphology metadata. It makes no changes
to PPO objectives, generator advantages, or model weights. It runs no extra
environment steps. This is observational prediction, not evidence of causal effects.

Enable `params.config.generator.response_probe.enabled: true` in a copy of the
normal training configuration. At each resample boundary the capture uses up to
256 distinct bodies and eight transitions per body from the late half of the last
training rollout, before post-adaptation evaluation resets the environment.
The controller has finished its window of training, but the recorded transitions
were collected with the rollout policy, not newly evaluated with its final weights.

Capture uses raw local module state (25 dimensions), the applied current action
after the trainer's clamp/rescale, and the same limb's six contact channels.
Targets are changes in joint velocity and six parent-relative velocity channels.
Only actuated modules are included: passive caps do not expose comparable dynamic
observations in this branch. Reset-crossing transitions and the last timestep are
excluded. Body identity hashes the full counts, effector subtype and cap arrays.

Static tokens are obtained from the completed `_encode_design` pass. A temporary
pre-hook captures the actual encoder input for the pre-attention baseline. The
capture runs under no-grad in eval mode and restores individual training flags;
all saved arrays are detached. It does not call the controller or its normalizer.

Run offline (shell expands the capture glob):

```sh
.venv/bin/python scripts/probe_static_response.py \
  runs/ant_codesign/codesign_single_transformer/RUN/response_probe/*.npz \
  --output /tmp/static_response_report.json --epochs 100 --seed 42
```

Use captures from one controller checkpoint/window for the initial comparison.
Combining windows changes the learned static feature coordinates and behaviour
policy; it is not a clean test of morphology alone. Use separate reports for
independent controller seeds. Existing return/pooled-feature archives are not
substitutes for consecutive state-action transitions.

The `metadata_plus_context_v1` protocol compares interaction only,
interaction + metadata (subtype, parent subtype, depth, limb slot),
interaction + metadata + pre-attention tokens, and
interaction + metadata + contextual tokens. Report keys `pre` and `post`
include metadata; older reports without this protocol label replaced metadata
with the token instead. Reports record each model's inputs explicitly.
All use equal-width padded inputs and the same two-layer 64-wide SiLU MLP,
initialization, optimizer and minibatch ordering. Standardization is fitted on
training rows only. Unique body hashes define a 70/15/15 train/validation/test
split; validation selects the training epoch. Test errors are averaged per body,
with paired body-bootstrap intervals for baseline minus contextual error.

The static-swap control uses a different held-out body with the same module
subtype and depth, preserving state/action/contact and metadata inputs. A single donor is used
for every temporal row of a query module. Rows without a donor are excluded and
coverage is reported. The swap statistic is a matched-row diagnostic.

A positive gate requires reproducible contextual improvement over both metadata
and pre-attention baselines on unseen bodies, plus degradation under static swaps.
No integration is automatic. Small test sets, omitted whole-body dynamic context,
and controller/morphology correlations limit interpretation. Capture duration and
offline training duration are reported; GPU peak memory is not yet measured.
