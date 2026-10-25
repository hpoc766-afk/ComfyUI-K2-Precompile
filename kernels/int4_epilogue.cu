// SPDX-License-Identifier: GPL-3.0-only
// CUTLASS EVT composition follows Comfy-Org/comfy-kitchen's Apache-2.0
// convrot_w4a4.cu (255a43879fe57bbcbecfdb273b46d772b00c5a90).
// Retains the eager low-precision rounding after projection AND gating.
#include <cuda_runtime.h>
#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/default_gemm_universal_with_visitor.h"
#include "cutlass/epilogue/threadblock/fusion/visitors.hpp"

#ifdef _WIN32
#define EXPORT extern "C" __declspec(dllexport)
#else
#define EXPORT extern "C" __attribute__((visibility("default")))
#endif

namespace {
template<class Out, int BM, int BN, int BK, int WM, int WN>
struct ResidualGemm {
    using TB = cutlass::gemm::GemmShape<BM, BN, BK>;
    using Warp = cutlass::gemm::GemmShape<WM, WN, BK>;
    using Inst = cutlass::gemm::GemmShape<16, 8, 64>;
    using Map = cutlass::epilogue::threadblock::OutputTileThreadLayout<TB, Warp, Out, 8, 1>;
    using Acc = cutlass::epilogue::threadblock::VisitorAccFetch;
    using Col = cutlass::epilogue::threadblock::VisitorColBroadcast<Map, float, cute::Stride<cute::_1, cute::_0, int32_t>>;
    using Row = cutlass::epilogue::threadblock::VisitorRowBroadcast<Map, float, cute::Stride<cute::_0, cute::_1, int32_t>>;
    using Gate = cutlass::epilogue::threadblock::VisitorRowBroadcast<Map, Out, cute::Stride<cute::_0, cute::_1, int32_t>>;
    using Residual = cutlass::epilogue::threadblock::VisitorAuxLoad<Map, Out, cute::Stride<int64_t, cute::_1, int64_t>>;
    using MulFloat = cutlass::epilogue::threadblock::VisitorCompute<cutlass::multiplies, float, float, cutlass::FloatRoundStyle::round_to_nearest>;
    using MulOut = cutlass::epilogue::threadblock::VisitorCompute<cutlass::multiplies, Out, float, cutlass::FloatRoundStyle::round_to_nearest>;
    using AddOut = cutlass::epilogue::threadblock::VisitorCompute<cutlass::plus, Out, float, cutlass::FloatRoundStyle::round_to_nearest>;
    using XScaled = cutlass::epilogue::threadblock::Sm80EVT<MulFloat, Acc, Col>;
    using Projected = cutlass::epilogue::threadblock::Sm80EVT<MulOut, XScaled, Row>;
    using Gated = cutlass::epilogue::threadblock::Sm80EVT<MulOut, Projected, Gate>;
    using Added = cutlass::epilogue::threadblock::Sm80EVT<AddOut, Residual, Gated>;
    using Store = cutlass::epilogue::threadblock::VisitorAuxStore<Map, Out, cutlass::FloatRoundStyle::round_to_nearest, cute::Stride<int64_t, cute::_1, int64_t>>;
    using EVT = cutlass::epilogue::threadblock::Sm80EVT<Store, Added>;
    using Kernel = typename cutlass::gemm::kernel::DefaultGemmWithVisitor<
        cutlass::int4b_t, cutlass::layout::RowMajor, cutlass::ComplexTransform::kNone, 32,
        cutlass::int4b_t, cutlass::layout::ColumnMajor, cutlass::ComplexTransform::kNone, 32,
        Out, cutlass::layout::RowMajor, 8, int32_t, float,
        cutlass::arch::OpClassTensorOp, cutlass::arch::Sm89,
        TB, Warp, Inst, EVT, cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
        3, cutlass::arch::OpMultiplyAddSaturate, 1>::GemmKernel;
    using Gemm = cutlass::gemm::device::GemmUniversalAdapter<Kernel>;

    static int run(void* a, void* w, void* xs, void* ws, void* gate,
                   void* residual, void* output, int m, int n, int k, cudaStream_t stream) {
        typename XScaled::Arguments x_args{{}, {static_cast<float*>(xs), 0.f, {cute::_1{}, cute::_0{}, m}}, {}};
        typename Projected::Arguments p_args{x_args, {static_cast<float*>(ws), 0.f, {cute::_0{}, cute::_1{}, n}}, {}};
        typename Gated::Arguments g_args{p_args, {static_cast<Out*>(gate), Out(0), {cute::_0{}, cute::_1{}, n}}, {}};
        typename Added::Arguments r_args{{static_cast<Out*>(residual), Out(0), {n, cute::_1{}, int64_t(m) * n}}, g_args, {}};
        typename EVT::Arguments cb{r_args, {static_cast<Out*>(output), {n, cute::_1{}, int64_t(m) * n}}};
        typename Gemm::Arguments args(
            cutlass::gemm::GemmUniversalMode::kGemm, {m,n,k}, 1, cb,
            static_cast<cutlass::int4b_t*>(a), static_cast<cutlass::int4b_t*>(w), nullptr, nullptr,
            int64_t(m)*k, int64_t(n)*k, 0, 0, k, k, 0, 0);
        Gemm gemm;
        auto status = gemm.can_implement(args);
        if (status != cutlass::Status::kSuccess) return int(status);
        if (Gemm::get_workspace_size(args)) return -1;
        status = gemm.initialize(args, nullptr, stream);
        if (status != cutlass::Status::kSuccess) return int(status);
        return int(gemm(stream));
    }
};

template<class Out>
int launch(void* a, void* w, void* xs, void* ws, void* gate, void* residual,
           void* output, int m, int n, int k, int config, cudaStream_t stream) {
    if (config == 0)
        return ResidualGemm<Out,128,256,128,64,64>::run(a,w,xs,ws,gate,residual,output,m,n,k,stream);
    return ResidualGemm<Out,64,256,128,32,64>::run(a,w,xs,ws,gate,residual,output,m,n,k,stream);
}
}

EXPORT int k2_int4_gated_residual(void* a, void* w, void* xs, void* ws, void* gate,
    void* residual, void* output, int m, int n, int k, int dtype, int config, void* stream) {
    if (m <= 0 || n % 8 || k % 32 || (config != 0 && config != 1)) return -2;
    if (dtype == 0)
        return launch<cutlass::bfloat16_t>(a,w,xs,ws,gate,residual,output,m,n,k,config,static_cast<cudaStream_t>(stream));
    if (dtype == 1)
        return launch<cutlass::half_t>(a,w,xs,ws,gate,residual,output,m,n,k,config,static_cast<cudaStream_t>(stream));
    return -3;
}
