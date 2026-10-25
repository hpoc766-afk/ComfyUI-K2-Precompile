# Weight I/O and normalized INT4 inputs

The DiT retains one model-level `torch.compile(fullgraph=True)` entry and the
Kohaku GQA FA path. These changes do not cache image activations across forwards
and do not add a separate text-processing compiler entry.

## Weight staging and speculative prefetch

The native runtime has bounded reusable pinned-host and CUDA cast buffers.
Host content reuse is keyed by binding, parameter slot, source pointer and source
version; model detach removes that model's entries. A host buffer cannot be
rewritten while its H2D event is pending. CUDA cast buffers cannot be reused until
the consumer event completes. CUDA events themselves are recycled after their
consumers finish.

Speculative prefetch has its own byte budget. It never waits for current compute
to release VBAR pages. When a VBAR range is unavailable, it may stage a temporary
CUDA copy within that budget and an available-memory guard. It defers if the
budget or host staging is unavailable. Abandoned transfers remain charged to the
budget until their transfer events complete. An actual projection may still wait
when necessary. Idle CUDA cache entries can be released under VBAR pressure.
The cache limits cover retained pool capacity; oversized/overlapping active
casts may use ordinary transient allocations.

Default per-device limits:

| Environment variable | Default | Meaning |
| --- | ---: | --- |
| `K2_HOST_STAGING_MIB` | 256 | Retained pinned CPU staging capacity |
| `K2_CAST_CACHE_MIB` | 8 | Retained reusable CUDA cast-buffer capacity |
| `K2_PREFETCH_MIB` | 128 | Bytes held by speculative prefetched weights |

These are private runtime budgets, not a replacement of ComfyUI's global VRAM
reservation setting. No global reservation is overwritten by the node.

## Overlapping input quantization and weight transfer

`wo/down` submit input quantization before preparing/waiting for their weights.
The compute stream only waits for the weight event immediately before GEMM.
This permits input processing to run while CPU staging and transfer-stream H2D
are in progress. Unquantized projections continue to use their original math.

## RMSNorm + modulation + ConvRot + INT4 packing

The new CUDA candidate uses the original 128-thread RMS sum order and explicit
low-precision rounding for normalization and modulation. The normalized row is
kept in shared memory. It then runs the original comfy-kitchen typed ConvRot
stages, row maximum reduction, scale calculation and signed-INT4 packing.
Supported rotation groups are 16/64/256, with BF16/FP16 inputs.

When this candidate wins Kohaku backend selection, no full normalized BF16/FP16
activation is written to global memory. QKV and MLP gate/up projections consume
packed activation bytes and row scales directly. The residual tensor remains
available for the following skip connection. The alternate candidate uses the
existing normalization and quantization kernels with the same packed-output
contract.

On the compatible INT4 main path:

```text
first entry: norm/modulation/rotation/quantization -> packed QKV input
QKV -> Kohaku FA -> wo/gated residual
                   -> norm/modulation/rotation/quantization -> packed MLP input
gate/up + SwiGLU -> down/gated residual
                   -> norm/modulation/rotation/quantization -> next block's packed QKV input
```

The final block keeps the ordinary normalized input for the final output layer.
LoRA, custom projections/hooks, mixed rotation groups within a shared projection
set, reference timestep splitting and attention patches retain the existing
unpacked path. Other weight formats keep their current compute backends.

## Local speed result

RTX 4060 Laptop 8 GB; real ConvRot INT4 Krea2, BF16, 4050 tokens,
28 main blocks, VBAR headroom 1536 MiB. Three fullgraph callables share one
loaded model instance and the Kohaku FA path. Each has one captured graph.
Five interleaved timed calls per mode follow first execution and two warmups.

| Configuration | Warm forward median |
| --- | ---: |
| Previous compiled path (`optimize_io=False`, `fuse_input_quant=False`) | 1359.132 ms |
| Bounded weight buffers + early input quantization | 1206.690 ms |
| Above + normalized packed INT4 input chain | 1175.027 ms |

The combined changes reduced warm forward time by 13.55% in this run. These are
DiT forward latencies, not complete image generation times. The baseline already
includes Kohaku FA and the earlier projection fusions.

Kohaku selection measured normalization/modulation plus separate ConvRot packing
at 3.715 ms and the new 256-thread fused candidate at 2.290 ms. The fused path
was selected and actually executed 56 times per complete forward; the composed
normalized-input path executed zero times in the measured final configuration.
The five calls reused host staging 1570 times and CUDA cast buffers 540 times.
H2D volume remained 2.451 GiB per call in all modes: this run demonstrates better
reuse/overlap and input processing, rather than reduced weight-transfer volume.
A hot-call profile recorded only the outer project Python `__call__` once.

First calls including tuning/compilation/execution were 61.795 s, 41.114 s and
38.851 s respectively. Compiled caches were retained between configurations,
so these are not three independent empty-cache compilation timings. Raw timing
and per-call counter deltas are included in [the benchmark data](benchmarks/io-input-quant-4050.json).
No numerical/image comparison or new validation
scripts were added in this step.

## Build

```bash
python scripts/build_norm_quant.py --kitchen-dir /path/to/comfy-kitchen --nvcc /path/to/nvcc
```

Windows builds may supply `--vcvars /path/to/vcvars64.bat`. The script extracts
comfy-kitchen's device-helper prefix into the ignored build directory, hashes the
backend helpers/toolchain/flags, and writes `.kernel-build/norm_quant.json` plus
the source-matched library. Restart ComfyUI after changing native code.

`FullGraphCall(optimize_io=False, fuse_input_quant=False)` is the internal control
configuration for comparing the previous execution path under the same model
instance. The normal defaults enable these optimizations; input fusion also
requires the existing forward-fusion setting and a compatible native path.
