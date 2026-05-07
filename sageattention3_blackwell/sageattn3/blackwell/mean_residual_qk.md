# Mean/Residual QK Quantization Plan

## Goal

Replace the current Q/K centering scheme with a blockwise decomposition that keeps a low-rank block direction plus per-token orthogonal residuals. For a block size `b`, choose `b + 1` so it divides the kernel tile size.

The current best low-precision candidate is `b = 7`. A `128 x 128` tile then holds 16 expanded blocks, with 112 real tokens per tile. Q/K/V padding for this path should be to 112-token logical tile boundaries before expansion.

For each block of token vectors:

```text
x_mean = pool(x_1, ..., x_b)
lambda_i = dot(x_i, x_mean) / max(norm(x_mean)^2, eps)
r_i = x_i - lambda_i * x_mean
```

The simple per-block packed layout is:

```text
[x_mean, r_1, r_2, ..., r_b, x_mean_2, r_1, r_2, ..., r_b, ...]
```

For `b = 7`, the low-precision-friendly tile layout is slot-major:

```text
[mean_0, mean_1, ..., mean_15,
 r1_0,   r1_1,   ..., r1_15,
 ...
 r7_0,   r7_1,   ..., r7_15]
```

Both `x_mean` and residual vectors are quantized with channel block scales to NVFP4. The scalar `lambda_i` values are kept separately.

## Score Reconstruction

For one Q block and one K block, let the expanded MMA accumulator be `A'`, where:

```text
A'00      = dot(q_mean, k_mean)
A'0,j+1   = dot(q_mean, k_r_j)
A'i+1,0   = dot(q_r_i, k_mean)
A'i+1,j+1 = dot(q_r_i, k_r_j)
```

The original token score is:

```text
A_ij =
    lambda_q_i * lambda_k_j * A'00
  + lambda_q_i              * A'0,j+1
  +              lambda_k_j * A'i+1,0
  +                            A'i+1,j+1
```

Mean rows and columns are structural slots, not real tokens. They must not participate in softmax or PV.

### Low-Precision Packing Caveat

The Q/K packed direction should be the block mean, not the block sum.

Using a sum vector is algebraically equivalent if the `lambda` coefficients are rescaled, but it is a poor NVFP4 operand:

```text
x_sum = b * x_mean
lambda_sum_i = lambda_mean_i / b
```

The final score can be made exact in real arithmetic, but the raw low-precision MMA terms see the larger operands:

```text
dot(x_sum, y_sum)       has roughly b^2 larger scale
dot(x_sum, y_residual)  has roughly b larger scale
```

For `b = 7` or `b = 15`, that wastes dynamic range before the later `lambda` correction can undo it. The Q/K slot 0 should therefore store `x_mean`, not `sum(x_i)`.

The residuals are signed:

```text
r_i = x_i - lambda_i * x_mean
```

This is compatible with the signed `e2m1` NVFP4 MMA path, but the quantizer must scale residual vectors by absolute max. It also means any P-side residual packing needs signed P quantization; the existing positive-softmax assumptions are not enough once we subtract a row/block mean from P.

## Fragment Layout Notes

The relevant local docs are:

- `csrc/cutlass/media/docs/cpp/cute/0t_mma_atom.md`, especially the `CLayout` discussion: an MMA atom maps `(logical_thr_id, logical_val_id)` to C coordinates.
- `csrc/cutlass/media/docs/cpp/cute/0x_gemm_tutorial.md`, especially the `partition_C` section: `ThrMMA::partition_C` gives the per-thread accumulator fragment.
- `sageattn3/blackwell/kernel_traits.h`: the current QK path uses `SM120_16x32x64_TN_VS_NVFP4`, `AtomLayoutMNK = Layout<Shape<_8, _1, _1>>`, and `Tile<128, 32, 128>`.
- `sageattn3/blackwell/cute_extension.h`: the custom NVFP4 atom has `CLayout` for `(T32,V16) -> (M16,N32)`.

A direct `partition_C(identity_tensor<128,128>)` probe for the current QK tiled MMA shows:

```text
tiled_mma size = 256 threads
tile_shape     = (128, 32, 128)
per-thread C fragment shape = (((2,4),2), 1, 4)
```

For thread 0, the logical score coordinates are:

```text
rows:    0, 8
columns: 0,1, 8,9, 16,17, 24,25, ..., 120,121
```

Threads 0..31 cover one `16 x 128` score strip. Within that warp, lanes 0..3 own row 0 with different column-pair residues, lanes 4..7 own row 1, and so on through rows 7/15. Therefore `b = 15` is the natural first block size: one `[mean, r_1, ..., r_15]` Q block lands inside one 16-row atom/warp strip, and each K block of 16 columns aligns to half of the atom's 32-column N span.

The important correction is that the mean/residual entries are not all local to the same single thread, but for `b = 15` they are local to the same warp-level MMA atom. That makes an interleaved path plausible with warp shuffles:

- `A'i+1,j+1` is the thread's local residual/residual score.
- `A'0,j+1` can be broadcast from the lane that owns Q slot 0 for the same K slot.
- `A'i+1,0` can be broadcast from the lane that owns the same Q slot and the K group mean column.
- `A'00` can be broadcast from the Q-mean/K-mean source lane for that 16-column K group.

This should not require extra K-mean loads or separate mean GEMMs. It does require a structured reconstruction pass over the accumulator fragment before softmax/masking/PV, using the known `((2,4),2) x 1 x 4` fragment layout rather than a flat loop.

### `b = 7` Slot-Major Layout

The `b = 7` slot-major layout is preferred for low-precision scaling:

```text
[means(16) | residual_1(16) | ... | residual_7(16)]
```

This makes one 128-slot expanded tile hold exactly 16 original 7-token blocks. The first 16 K/P columns are all block means, and the first 16 V rows are all block means. That lines up with the existing sequence-axis scale-factor direction:

- P scale blocks cover 16 K columns for a row fragment.
- V scale blocks cover 16 sequence positions for a fixed head-dim row.

Therefore the mean slots and residual slots do not have to share the same P/V block scale.

For QK reconstruction with both Q and K slot-major, the logical coordinates are:

```text
row = q_slot * 16 + q_group
col = k_slot * 16 + k_group
```

and the real token score for `q_slot > 0`, `k_slot > 0` is:

```text
A =
    lambda_q * lambda_k * A'[q_group,                  k_group]
  + lambda_q            * A'[q_group,                  k_slot * 16 + k_group]
  +            lambda_k * A'[q_slot * 16 + q_group,    k_group]
  +                      A'[q_slot * 16 + q_group,    k_slot * 16 + k_group]
```

A `partition_C` probe shows the cost of this layout:

```text
tid 0..31:    Q mean rows, containing A00 and q_mean * k_residual
tid 32..255:  Q residual rows, containing q_residual * k_mean and residual/residual
```

So `A'0j` and `A'00` are not warp-local to the residual rows. They need either cross-warp staging of the first 16 Q-mean rows, or a different Q layout.

An asymmetric layout may avoid this:

```text
Q: [mean, r1, ..., r7] per block, two blocks per 16-row atom
K/V: [means(16) | r1(16) | ... | r7(16)]
```

With this layout, K/P/V still get the scale-block advantage, while Q mean/residual rows are in the same warp-level 16-row atom. A probe shows the q-mean lanes and q-residual lanes for a pair of Q blocks are separated by lane groups inside the same warp, so score reconstruction can use warp shuffles rather than cross-warp shared-memory staging.

## Interleaved Expanded GEMM

The most direct data layout is to feed the expanded Q/K tiles to the existing QK MMA. This reuses the current TMA, shared-memory layouts, scale-factor loading, and NVFP4 MMA path with minimal structural changes.

Pros:

- One QK MMA stream computes all mean/residual dot products.
- Q/K quantization can reuse the existing vector quantizer after packing.
- Expanded tile size can remain 128 when `b + 1` divides 128.
- V can be expanded with zero mean slots, so the existing PV tile shape can also be reused after mean columns are masked out.
- For `b = 7`, K/P/V mean slots can be placed in their own 16-wide scale blocks.
- Key means are loaded as normal K slots, so no separate K-mean operand stream is needed.

Cons:

- Reconstructing each compact score still needs four entries from the expanded accumulator tile.
- The four entries are not all local to the same thread. For slot-major `b = 7`, Q mean terms require cross-warp access unless Q uses the asymmetric block-local layout.
- The P tile currently wastes work on structural mean rows/columns unless the softmax/PV path is taught to skip them.
- Causal masking must operate in original token coordinates, not expanded slot coordinates.
- V needs the same structural padding, with zero mean slots, so that existing PV layout machinery can be reused.

### V Mean Slot Test

Packing plain V block means into the structural mean slots was tested against a dense CUDA reference for `b = 15`, `head_dim = 128`, and one 120-token logical tile.

```text
score reconstruction max error              2.29e-05
zero V mean slots + masked mean columns      1.34e-05
packed V means + masked mean columns         1.34e-05
packed V means + raw expanded softmax        6.04e+00
```

So packing plain V means is numerically equivalent to zero padding only when the structural mean columns are masked before PV. It does not by itself avoid the P-tile waste, because allowing mean columns to participate in softmax changes the distribution.

The useful packing is instead:

```text
V'_0 = v_mean
V'_j = v_j - v_mean
P'_i0 = sum_j P_ij
P'_ij = P_ij
```

Then each block contribution is exactly preserved:

```text
P'_i0 * V'_0 + sum_j P'_ij * V'_j
  = (sum_j P_ij) * v_mean + sum_j P_ij * (v_j - v_mean)
  = sum_j P_ij * v_j
```

A dense CUDA check with `Lq = 123`, `Lk = 119`, `b = 15`, and `head_dim = 128` gives:

```text
score reconstruction max error              2.29e-05
packed V mean/residual PV max error          1.48e-05
expanded P shape                             [1, 1, 123, 128]
packed V shape                               [1, 1, 128, 128]
```

This means the interleaved kernel should not leave K-mean columns as zero-probability structural slots for PV. After softmax over real residual-token columns, it should write the per-block probability mass into the mean column and keep the residual-token probabilities in the remaining columns. The V tile should be packed as `[v_mean, v_i - v_mean]`.

### Q Mean Row Packing

The remaining waste is on the Q side: each expanded Q block has a structural mean row that does not correspond to an output token. That row can also be made useful by applying the same mean/residual idea to P before the PV MMA.

For each Q block and each K/P column:

```text
P_mean_j = mean_i(P_ij)
P'_0j    = P_mean_j
P'_ij    = P_ij - P_mean_j
```

PV then produces:

```text
O'_0 = P_mean @ V
O'_i = (P_i - P_mean) @ V
```

The original output row is recovered after PV by adding the structural row back:

```text
O_i = O'_i + O'_0
```

A dense CUDA check with random probabilities and V gives:

```text
K probability mass + V mean/residual error   3.58e-07
K probability mean + V sum/residual error    3.28e-07
Q row-mean P packing error                   4.17e-07
```

The K-side mean version is also algebraically valid if `V'_0 = sum_j v_j`, but that has the same dynamic-range problem as Q/K sum packing. Use `V'_0 = v_mean` with `P'_0 = sum_j P_j` instead. Here `P'_0` remains bounded by the row probability mass, while `v_mean` stays at normal V scale.

The Q-side row-mean packing introduces signed P residuals, so the PV quantization scale should be based on absolute values after centering rather than the current positive-only softmax scale.

The current V quantization already block-scales along the sequence axis after transposition:

```text
packed V    shape: [B, H, D, N / 2]
V scales    shape: [B, H, D, N / 16]
scale block: 16 consecutive sequence positions for each head-dim row
```

This matches the PV accumulation direction. It also means V mean/residual packing should be designed around sequence blocks, not head-dim/channel blocks. The packed `[v_mean, v_i - v_mean]` slots for a 16-wide expanded K block share the same scale-factor direction that the existing PV MMA expects.

## Separate GEMM Components

An alternative is to compute the four score terms as separate compact `b x b` contributions:

```text
q_res             @ k_res.T
q_mean_broadcast  @ k_res.T
q_res             @ k_mean_broadcast.T
q_mean_broadcast  @ k_mean_broadcast.T
```

The cross and mean-mean terms are then scaled by `lambda_q`, `lambda_k`, or both and accumulated into the normal compact score fragment before softmax.

Pros:

- The accumulator fragment is already in original-token coordinates.
- No cross-fragment gather is needed to combine four expanded accumulator entries.
- Existing softmax, score quantization, and PV machinery can remain much closer to the current kernel.
- Causal masking remains natural because score rows and columns still correspond to original tokens.

Cons:

- More QK MMA work per K tile.
- More Q/K operand streams or repeated loads are needed.
- Mean vectors are duplicated across rows/columns in the simple implementation, which wastes math.
- A high-performance version probably needs specialized broadcast/GEMV-style handling for the mean terms.
- It gives up the main layout advantage of the proposal: key means are no longer just normal K slots in the same TMA/MMA stream.

## Current Implementation Direction

Use the specialized `head_dim=128`, BF16, causal build path for development.

1. Add Python-side decomposition and reconstruction helpers for correctness checks.
2. Use `b = 7` first, so each 128-slot tile holds 16 packed 7-token blocks and the mean slots align with P/V sequence scale blocks.
3. Prefer K/V slot-major packing. Decide whether Q should also be slot-major, requiring cross-warp staging of Q-mean score rows, or block-local, allowing warp-local reconstruction while retaining K/P/V scale benefits.
4. Implement the fused prototype as an interleaved expanded GEMM plus score reconstruction before softmax.
5. Treat separate-component GEMMs as a fallback if reconstruction/staging costs more than the saved operand loads and extra GEMM work.
