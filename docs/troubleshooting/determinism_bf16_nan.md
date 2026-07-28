# Why `--seed` runs NaN out (deterministic algorithms × bf16 × torch.compile)

This explains why `torch.use_deterministic_algorithms(True)` is disabled in `train_utils.py`, and how
to turn full bit-exact determinism back on safely. Sibling of
[`deterministic_embedding.md`](../guides/deterministic_embedding.md) (a *speed* casualty of the same
flag); this one is a *correctness* casualty.

## The symptom

Passing `--seed` to `train_ant_codesign_single.py` produced:

- A steady stream of `NaN or Inf found in input tensor.` warnings (from `tensorboardX`, i.e. a
  *logged metric* is NaN) — starting from epoch 1, ~5/epoch, not a one-off.
- All control-policy training scalars NaN together: `losses/a_loss`, `losses/c_loss`, `info/kl`,
  `losses/bounds_loss`, `control/grad_norm`.
- Degraded learning: at resample #1 (ep 63), seeded `R_mean = 0.448` vs unseeded `1.166` — 2.6× worse
  from the *same* seed/RNG. Eventually the run destabilises.

No-seed runs are completely clean and reach 3000 epochs. The problem is `--seed`, not the model.

## Root cause: a three-way interaction

The NaN needs **all three** of these at once. Remove any one leg and it is clean (verified):

| bf16 (`mixed_precision`) | `use_deterministic_algorithms` | `torch.compile` | result |
|:---:|:---:|:---:|:---:|
| on | on | on | **NaN** |
| on | off | on | clean  ← the fix (D) |
| **off (fp32)** | on | on | clean  ← re-enable path (G) |
| on | on | off (eager) | clean |

Flag bisection isolated the trigger to **exactly** `use_deterministic_algorithms(True)` — disabling
`CUBLAS_WORKSPACE_CONFIG` or `cudnn.deterministic` alone does **not** help.

`mixed_precision: True` (bf16 autocast) is inherited from `configs/defaults/base.yaml` — a global
default, not a codesign choice. It buys tensor-core speed + half the activation memory (needed to fit
`minibatch_size: 32768` on a 16 GB GPU). It is fine on its own; the unseeded runs use it.

## Where exactly the NaN is born

Located with `TORCHINDUCTOR_NAN_ASSERTS=1` (aborts at the first compiled kernel producing NaN/Inf):

- **Flash-attn path (default):** first NaN is `buf41 = buf38[2]`, the **grad_value** output of
  `aten._scaled_dot_product_flash_attention_backward` — emitted with kernel metadata
  `'deterministic': True`. So the *backward* NaNs first; the corrupted gradient poisons the weights,
  and the *next* forward makes every loss NaN. (Explains why it's in the backward yet a_loss/c_loss
  read NaN, and why it's intermittent — data-dependent.)
- **Math path (flash disabled to test):** NaN just relocates — first Inf is `buf60`, the **FFN
  pre-activation** (`addmm` → gelu) in **bf16**, inside the same `TransformerEncoderLayer`.

Both failure sites live inside the encoder. It is not attention-specific and not a masking bug: the
control forward (`architectures.py: _encode_codesign`) passes **no** `src_key_padding_mask` at all.

## Why bf16 is the deciding factor

- **bf16 = bfloat16**: 8 exponent bits (fp32 range, max ~3.4e38) but only **7 mantissa bits**. Its
  weakness is *precision*, not range — so an Inf here means error *compounded* into a real explosion,
  not a small-range clip.
- Deep nets survive in bf16 only because reductions (matmul accumulation, softmax sums, layernorm
  variance, grad reduction) are done in **fp32** and only downcast at the edges. Both eager autocast
  and inductor's *default* path preserve this fp32 accumulation.
- `use_deterministic_algorithms(True)` forces inductor onto **deterministic kernel variants**
  (deterministic reductions; the deterministic flash-attn backward). These are numerically less
  robust. In bf16's 7-bit mantissa there is no headroom to absorb that — a value overflows to Inf,
  `Inf−Inf`/`Inf/Inf` → NaN → optimizer writes NaN into weights → everything is NaN thereafter.
- In **fp32** (23 mantissa bits, ~16M× headroom) the same deterministic kernels run without the error
  ever compounding to overflow. That is why fp32 fixes it.

## The fix in place (D)

`train_utils.py` seeds the RNG (`config seed`, `cuda.manual_seed`, `PYTHONHASHSEED`) and keeps
`CUBLAS_WORKSPACE_CONFIG` + `cudnn.deterministic`, but **does not** call
`use_deterministic_algorithms(True)`. Consequences:

- Runs are **seed-stable**: same seed → same sampled morphologies + init, trajectories match up to
  low-order kernel nondeterminism (flash-attn backward atomics). Run-to-run drift ≪ the between-seed
  std the phase-comparison already reports over 5 seeds (`[42..46]`, ADR-0015). Not bit-exact.
- Full compile + bf16 speed retained (~43.7k env-steps/s @ 4096 envs). No config or memory change.

## How to re-enable bit-exact determinism (G)

You must break the triple by dropping **bf16**, not by any attention-backend trick (forcing the math
SDPA backend does **not** help — the instability just moves to the FFN):

1. Uncomment `torch.use_deterministic_algorithms(True)` in `train_utils.py`.
2. Set `mixed_precision: false` (fp32) in the config (overrides `base.yaml`).
3. fp32 doubles activation memory and **OOMs at `minibatch_size: 32768`** on a 16 GB GPU. Either:
   - drop `minibatch_size` to `16384` (changes PPO minibatch dynamics: 4 vs 2 per epoch), **or**
   - gradient-checkpoint the (single-layer) `TransformerEncoder` to keep 32768.

Measured cost: fp32 + det + compile ≈ **28.9k** env-steps/s at mb 16384, vs 43.7k for the bf16 fix.
Note this diverges from the bf16 baseline all other phases use, so prefer it only if you have a hard
bit-exact requirement.

## Reproduce / verify

```bash
# NaN (before fix): --seed + compile + bf16
python scripts/train_ant_codesign_single.py --seed 42 --max_epochs 5 --name x --headless True
#   -> 'NaN or Inf' warnings; a_loss/c_loss/kl NaN

# Locate the kernel:
TORCHINDUCTOR_NAN_ASSERTS=1 python scripts/train_ant_codesign_single.py --seed 42 --max_epochs 3 ...
#   -> AssertionError on buf = _scaled_dot_product_flash_attention_backward(...)[2]

# Clean (the fix): use_deterministic_algorithms off -> 0 NaN, full speed
python scripts/train_ant_codesign_single.py --seed 42 --max_epochs 15 --name x --headless True
```
