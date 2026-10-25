# Kohaku backend routing and GQA Flash Attention

The single DiT `fullgraph=True` entry is retained. Backend selection happens in
the compile-preparation phase, using real CUDA tensors with the graph's hinted
shapes. No selector, table access or timing loop is emitted into the model graph.
Hot inference reads the bound route IDs in C++.

## Reused Kohaku components

`KohakuKernelRouter` adapts KohakuMossTrainer's CUDA-event warmup/best-time
measurement, power-of-two token buckets, merging winner tables and fused/split
selection flow. The source revision is
`0536e62b6062039104d8f9575635b5d07bf573d3`. The adaptation removes trainer,
distributed and autograd dependencies; it uses CUDA events rather than CUDA
Graph capture/replay. Attribution and the original KohakuMoss License 1.0 are in
`NOTICE` and `third_party/KohakuMoss/LICENSE`. The project title identifies the
Kohaku backend to follow that license's naming requirement.

The candidates retain the same output contract:

| Operator | Joint candidate | Original-backend candidate |
| --- | --- | --- |
| QKV | joint INT4 projection, Q/K RMSNorm/RoPE, sigmoid gate | shared input quantization, original GEMMs/QK postprocessing, sigmoid gate |
| SwiGLU | dual GEMM + SiLU/multiply | original gate/up GEMMs + fused activation/multiply |
| Residual/RMSNorm | GEMM + gate/residual + full-row RMSNorm/modulation | original output GEMM/gated residual + standalone RMSNorm/modulation |

Choosing the original-backend residual/RMSNorm candidate still returns both
residual and normalized tensors; the following block consumes the prepared
normalized input. Source-level fusion grouping is therefore independent of the
chosen number of GPU launches.

Tables are keyed by device/toolchain/source fingerprint, operator, token bucket,
batch, precision, input/output dimensions and rotation groups. Missing shapes
are timed once; subsequent layers and processes reuse the winner. Incompatible
or failed non-sticky candidates are excluded. Device errors such as illegal
memory access are raised. Synthetic tuning buffers use constants and do not
consume the application's random-number stream. Temporary VBAR pins/cast
buffers are completed and released before compiling the graph.

- `K2_KERNEL_TUNE=table` (default): read tables, measure missing keys and store winners.
- `K2_KERNEL_TUNE=off`: reuse tables, otherwise bind the original-backend candidate.
- `K2_KERNEL_CACHE`: optional cache root. Default:
  `~/.cache/comfyui-k2/KohakuKernelRouter/`.

The joint GEMM candidates currently apply to the supported SM89 ConvRot INT4
path. Other weight formats keep their existing matrix-multiplication backends.
The routing API chooses implementations; it does not convert weight formats.

## Kohaku GQA Flash Attention

The compiled attention path uses a forward adapted directly from KohakuMoss's
Triton `_mma`, `_fwd_tile` and `fwd_kernel`, with its sm89 tile seeds. A program
owns a query tile and one Q head. It reads `kv_head = q_head // (Hq // Hkv)`;
for K2, four Q heads share one of the 12 K/V heads. K/V tensors retain 12 heads
throughout this path. No `repeat_interleave`, head-wise tensor expansion or full
`[B,H,Lq,Lk]` score/probability allocation is performed.

K/V is streamed in tiles. Full tiles have no tail predicates; only the final
partial K/V tile is masked. The online softmax uses `exp2`, FP32 maxima/sums and
FP32 output accumulation, with probabilities rounded to the operand precision
for Tensor Core multiplication. Inference omits training's saved M/L planes.
Boolean and additive masks are read by broadcast strides, without expanding
mask storage. Fully masked rows return zero attention output.

Without output patches, the attention output cast and sigmoid-gate multiplication
are included in the output store. Output patches receive the ungated attention
result before gate multiplication. BF16/FP16 and head dimensions 32/64/128/256
are supported; the actual K2 head dimension is 128.

The frontend is an opaque graph operator. Before Inductor compilation, FA tiles
are selected through KohakuKernelRouter and that operator is lowered to an
explicit wrapped Triton launch. This keeps a single full graph while preventing
SDPA math decomposition into full score matrices. The generated C++ wrapper
launches the Triton kernel in native mode. CUDA Graph replay remains disabled.

ComfyUI attention overrides/wrappers receive the original head counts plus
`enable_gqa=True`; this node does not replicate heads for external backends.
Their implementations determine their own behavior. The earlier `gqa_fusion.py`
and native SDPA interface remain available as development source; the compiled
model's default attention path uses Kohaku FA.


## Local forward timing

RTX 4060 Laptop 8 GB, real ConvRot INT4 Krea2, BF16, batch 1, 4050 tokens
(832×1216-equivalent latent and 98 text tokens), 28 main blocks. VBAR headroom
was 1536 MiB. All modes shared one loaded model instance. Five interleaved
measurement rounds followed each mode's first call and two warmups.

| Path | Warm forward median |
| --- | ---: |
| Original ComfyUI eager | 1617.897 ms |
| Original projection backend + Kohaku FA | 1329.413 ms |
| Kohaku backend routing + Kohaku FA | 1330.806 ms |

The adaptive path reduced warm forward time by approximately 17.7% relative to
the original eager path in this run. The two compiled modes performed similarly.
The timed adaptive graph contains 32 Kohaku FA launches (28 main blocks and four
text-fusion blocks) and remains one full graph. Its main attention uses
`BM=128, BN=64, warps=8, stages=2`, directly addressing 48 Q heads and 12 KV heads.
Projection routing selected the original-backend compositions for all four
measured INT4 projection shapes.

The generated C++ wrapper calls `cuLaunchKernel` for Kohaku `fwd_kernel` entries;
no `_safe_softmax`, SDPA math decomposition or `repeat_interleave` appears in that
wrapper. A profiled hot call entered this project's Python code only once at the
outer `FullGraphCall.__call__`, with no Python FA or tuning calls.

First calls including tuning/compilation/execution took 89.319 s for the fixed
projection path and 59.008 s for the adaptive path. Compiled caches were retained
between modes, so these are not independent empty-cache compilation timings.
These measurements are DiT forward latency, not complete sampling or VAE time.
No numerical/image comparison or new validation scripts were added in this step.

## Build

Rebuild the native C++ adapter after updating the routing code; its existing
source-hashed PyTorch extension builder runs on the first compiled invocation.
The existing CUDA libraries remain applicable. No trainer checkout is required
when using the node. Restart ComfyUI to load changed native operator definitions.

Timing results for the preceding fixed-joint implementation remain historical
in `joint-projection.md`; they do not describe this adaptive/Kohaku-FA version.
