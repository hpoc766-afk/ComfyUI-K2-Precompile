# Joint projection kernels

Current execution uses [Kohaku backend routing and GQA Flash Attention](backend-routing.md).
The selector may bind the original-backend composition for any operator. The
fixed-joint timings below describe the preceding version, before adaptive routing
and the switch to Kohaku FA.

The existing **Fuse forward operators** switch enables the new paths. The DiT
retains one model-level `fullgraph=True` compilation entry. This change adds
compute operators; it does not introduce a persistent task scheduler or CUDA
Graph replay.

The joint GEMMs use 128-token tiles and `cp.async` double-buffered shared
operands. Each warp owns multiple MMA fragments, reusing a weight tile across
more token rows instead of fetching every MMA operand directly from global
memory. Token tiles are the fast grid axis, so neighboring CTAs reuse the same
weight tile in L2 before moving to the next output-channel/head tile. Eligibility
for carrying normalized inputs across blocks is fixed before graph capture.

## Operators

- **Joint QKV**: one CUDA launch covers Q, K, V and the attention-gate matrices.
  Each Q/K CTA owns a complete 128-channel head and applies RMSNorm and RoPE
  before its first output store. The gate projection stores BF16/FP16 sigmoid
  results. Q/K/V/gate weights remain separate; no concatenation or repacking is
  required. The launch uses signed-INT4 Tensor Core MMA instructions directly.
- **Joint SwiGLU**: a CTA maintains gate/up GEMM accumulators, reuses activation
  fragments for both matrices, rounds both projections to the activation dtype,
  computes and rounds SiLU, then multiplies by up and stores the hidden output.
  The two full projection tensors are not allocated.
- **Residual + normalization**: the `wo/down` INT4 GEMM applies the projection
  cast, time gate, gate-result cast and residual add. Each channel tile publishes
  its residual output with device release/acquire ordering; the last tile of a
  128-token group performs full-row RMSNorm and modulation in the same CUDA
  launch. No block waits for an unscheduled block. Residual and normalized
  outputs are both retained: one feeds the next skip, the other the next GEMM.
- **Gated GQA**: a Triton forward kernel indexes K/V heads directly, runs tiled
  online softmax with `exp2`, rounds the attention output and multiplies the
  precomputed sigmoid gate before storing. It avoids K/V `repeat_interleave`.
  Attention masks, output patches and attention wrappers use the existing path.

The projection kernels currently target **SM89 ConvRot INT4**, with BF16 or FP16
activations and bias-free native projections. Q/K head dimension is 128; the
joint attention path currently accepts batch 1. The SwiGLU projection supports
multiple batches. Hooks, LoRA projection overrides, smoothing and incompatible
weight layouts retain the existing projection path. INT8, FP8 and dense weights
retain their existing matrix multiplication backends.

Input rotation/quantization still runs before the joint GEMMs. On the plain
INT4 main-block path, `wo` produces the MLP's normalized/modulated input, and
`down` produces the next block's normalized/modulated input. Pure time modulation
is prepared one block ahead and passed along without recomputing it. The last
block can produce the final output layer's normalized/modulated input. The first
block keeps its entry RMSNorm/modulation operator. Reference-time splitting,
hooks and customized blocks retain the existing dataflow where needed.

The whole block has **not** been reduced to 2–4 GPU launches. `joint_kernels=False`
on the internal `FullGraphCall` constructor selects the previous projection
operators while keeping the same fullgraph/native execution settings, allowing
speed comparisons under the same model instance.

## Build

No CUTLASS checkout is needed for the new joint-projection library:

```bash
python scripts/build_joint_projection.py --nvcc /path/to/nvcc
```

On Windows, supply `--vcvars` pointing to `vcvars64.bat` if the MSVC environment
has not been initialized. The source-hashed library and build metadata are
written under `.kernel-build/`. Native mode builds its C++ adapter through the
existing PyTorch extension builder. The GQA Triton kernel compiles on first use.

The legacy Python operator adapter and native C++ adapter both expose the new
projection operators. All four weights of a joint QKV launch, or both weights
of a joint SwiGLU launch, remain leased until that launch finishes. If the VBAR residency budget
cannot hold a jointly needed weight range, the native adapter uses a transient
CUDA cast buffer, retains its pinned host staging until the consumer completes,
and releases it through the same completion-event path.

## Local timing

RTX 4060 Laptop 8 GB, real ConvRot INT4 Krea2, BF16, batch 1,
832×1216-equivalent input, 4050 joint tokens, 28 main blocks. VBAR was enabled
with 1536 MiB headroom. One model instance was used for all three modes; five
measurement rounds alternated execution order after each mode's first call and
two warmup calls. Timings include dynamic weight management and synchronization.

| Mode | Warm forward median |
| --- | ---: |
| Original eager | 1607.432 ms |
| Previous fusion, same native execution adapter | 1320.050 ms |
| Joint projections + residual/RMSNorm chain | 1914.131 ms |

The joint version is approximately **45.0% slower than previous fusion** in this
measurement. Its first call, including graph capture/compilation and execution,
took 41.238 seconds; compiled artifacts were not cleared before the run.
Both compiled modes captured one full graph. Across eight calls per compiled
mode (first + two warmups + five measurements), the joint version recorded 224
joint QKV, 224 joint SwiGLU and 448 residual/RMSNorm launches: respectively
28, 28 and 56 per full forward. Both modes transferred 22,525,666,304 H2D bytes
in total during those calls. The measurements do not include VAE, independent
text encoding or a complete sampler run.

Source SHA-256 for the timed CUDA version:
`a0a9d68df1bd1973e937e6aa6fdab133b4c5837d6053f6078625f8d2d6d02bb4`.

## Implementation status

The CUDA projection library and C++ adapter are built from the sources above.
Numerical equivalence has not been measured for these new kernels. Timing
results are recorded separately; the new attention reduction order can differ
from the selected PyTorch attention implementation.
