# Learning static representations from responses

`scripts/train_response_representation.py` tests supervised morphology encoders
offline using existing response captures. Full counts, effector subtypes and cap
subtypes are joined from the corresponding post-evaluation archive using exact
SHA-256 body identities. Missing bodies are an error; caps are never inferred.

Four arms share the interaction + metadata inputs and a 64/64 SiLU response
decoder: metadata with zero latent, trainable local metadata MLP, a fresh static
body Transformer, and the same body Transformer initialized from an earlier
controller checkpoint. Decoder initialization is identical within a probe seed.
The architecture comes from the saved run config. The checkpoint loads strictly;
the original checkpoint and live trainer are never modified. Response losses
train only the offline encoder and decoder. Encoder parameter counts differ
between local and full-body arms and are explicitly reported.

Inputs and seven velocity-delta targets match the frozen response probe.
Static encoders see only completed morphology; state, action and contact enter
the decoder. Body batches contain 32 distinct bodies, with all their sampled
rows. Losses average rows within each body, then bodies. AdamW uses 1e-3 learning
rate and gradient clipping at 1.0. The initial bounded experiment uses 200 epochs,
seeds 42/43/44 and each window separately. Body hashes define 70/15/15 splits;
train rows alone define normalization. Validation body error selects the epoch.
Paired test-body bootstrap intervals use 2,000 resamples. These are probe seeds,
not independent controller training runs. A selected epoch near the limit means
convergence remains unresolved.

Example from the repository root:

```sh
.venv/bin/python scripts/train_response_representation.py \
  --capture PATH/response_e000189_w0002.npz \
  --morphologies PATH/post_eval_window_0002.npz \
  --checkpoint PATH/earlier_checkpoint.pth \
  --config PATH/config.yaml \
  --output /tmp/learned_response.json --epochs 200 --seed 42
```

For probe_capture_run2, initialization uses epoch 50, preceding both capture
epochs 126 and 189. The historical encoder can nevertheless have encountered
some test morphologies; the body split holds out offline response supervision,
not necessarily all prior exposure. Claims of new-morphology transfer need a
dedicated structural holdout and independently trained controllers.

A positive feasibility result requires full-body encoders to improve on metadata
and the learned-local control across windows and seeds. It does not establish a
co-design improvement, causal physics, or paper novelty. No auxiliary is integrated
into PPO or generation by this experiment.
