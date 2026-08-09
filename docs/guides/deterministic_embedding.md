# Why the mode embedding is a matmul, not an `nn.Embedding`

This explains `MatmulEmbedding` in `transformer_rl/architectures.py` and why `mode_emb` uses it.

## The symptom

Single-network codesign training (`train_codesign_single.py`) ran **~2.4x slower than every
other run** — ~3 h per seed for 3000 epochs, vs ~55 min for the same budget on the older codesign,
PPG, or plain-control nets. Throughput was ~24k env-steps/s where the others hit ~95–100k.

The slowdown only appeared **when a seed was set** (`--seed`). With no seed it ran at full speed.

## Why a seed makes it slow

Passing `--seed` turns on *fully deterministic* GPU math (`train_utils.py`):
`torch.use_deterministic_algorithms(True)`, `cudnn.deterministic`, `CUBLAS_WORKSPACE_CONFIG`. This
guarantees a run is bit-for-bit reproducible. The cost is that any operation without a fast
*deterministic* implementation falls back to a slow one.

Profiling pinned the entire slowdown to **one operation**: the backward pass of a single
`nn.Embedding`, which became ~60% of all GPU time under determinism and was essentially free without
it. (It only shows up under `torch.compile`, which the training uses — see "the compile part" below.)

## The operation, and why it's slow

The net tags every token with a **mode** — LIVE / COMMITTED / STOP — via a tiny lookup table
(`mode_emb`, 3 rows). In the forward pass each of `B * 16` content tokens (2 effectors × 8 limbs,
for a batch of `B` robots ≈ 16k–32k) looks up its 160-dim mode vector. That's a *gather* — cheap.

The **backward** pass is the problem. To get the gradient for the 3-row table, every one of the
`B * 16 ≈ 262,000` token gradients must be **added back into whichever of the 3 rows it came from**.
That's a *scatter-add*, and ~87,000 gradients pile onto each of the 3 rows.

- **Without determinism:** the GPU does this with `atomicAdd` — thousands of threads add into the 3
  rows at once. The order is arbitrary (so tiny floating-point differences run to run), but it's one
  fast parallel pass.
- **With determinism:** arbitrary-order adds aren't reproducible, so they're banned. PyTorch
  substitutes a *fixed-order* reduction, which effectively **serializes** ~262k additions funneling
  into 3 slots. That went from sub-millisecond to ~150 ms per call.

### Why this bit us specifically

A normal embedding (e.g. word vectors) has a *large* table — thousands of rows — so gradients spread
out and no single row is heavily contended; the deterministic path is fine. Ours is the opposite
extreme: a **3-row table indexed by a huge batch**, so all the gradient traffic collides on 3 rows —
close to the worst possible case for a deterministic scatter. Nothing was done "wrong"; a 3-row
embedding is just an unusual shape that happens to hit a known PyTorch sharp edge.

Because `mode_emb` only exists in the codesign net, this is exactly why *only* codesign was slow while
every other net (which has no such per-sample tiny embedding) stayed fast.

### The compile part

In plain eager mode the slowdown is small (~13%): `nn.Embedding`'s own backward has a decent
deterministic path. But the real training wraps the net in `torch.compile`, and the compiler lowers
embedding-backward to a generic scatter (`aten::_index_put_impl_`) whose deterministic kernel is the
worst-case one. So the full ~2.4x only appears under determinism **and** compile — the combination
used in real runs.

## The fix

An embedding lookup is, by definition, a matmul against one-hot vectors: looking up row `k` equals
`onehot(k) @ W`. `MatmulEmbedding` writes it that way:

```python
class MatmulEmbedding(nn.Embedding):
    def forward(self, ids):
        return F.one_hot(ids, self.num_embeddings).to(self.weight.dtype) @ self.weight
```

- **Forward:** bit-for-bit identical to `nn.Embedding` (a one-hot combination returns exactly the
  selected row).
- **Backward:** the weight gradient becomes `one_hotᵀ @ grad` — a **GEMM** (matmul). It computes the
  exact same sums (a matmul's inner product *is* that per-row reduction), but on a highly optimized
  kernel that is both parallel and, with `CUBLAS_WORKSPACE_CONFIG` set, deterministic. No scatter.

This recovers full speed **while keeping bitwise determinism** — 245 → 94 ms/iter under
determinism+compile (≈ the 90 ms you get with determinism off). It's strictly better than just
turning determinism off: you keep reproducible seeded runs *and* the speed.

### Why not use it everywhere?

Only for **small** categorical markers (a few rows). For a real vocabulary the one-hot tensor would
be enormous, and a normal `nn.Embedding` is the right choice — its scatter backward is fine when
gradients spread across many rows.

## Verification

`nn.Embedding` vs `MatmulEmbedding`, same weights and upstream gradient:

| dtype | forward | weight gradient |
|---|---|---|
| float32 | bitwise-equal | allclose, relative diff 2.6e-6 |
| float64 | bitwise-equal | allclose, relative diff 3.2e-15 |

Forward is exact. The gradient matches to floating-point rounding, and the difference shrinks to
machine epsilon at higher precision — confirming it is the same math, differing only in summation
order (the same kind of ordering difference `atomicAdd` already introduces).
