// SPDX-License-Identifier: GPL-3.0-only
// CUTLASS visitor adaptation: Copyright (c) 2017 - 2026 NVIDIA CORPORATION
// & AFFILIATES. BSD-3-Clause; see third_party/CUTLASS-LICENSE.txt.
// One CTA owns one complete head, retaining the rounded projection in shared
// memory before RMSNorm and adjacent-pair RoPE. No projection tensor hits HBM.
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
using namespace cute;
template<class Map, class Out, int BM, int HD, int Threads>
struct NormRopeStore {
    struct Arguments { Out* output; const float* norm_scale; const float* freqs; float eps; };
    using Params = Arguments;
    template<class Shape> static constexpr Params to_underlying_arguments(Shape const&, Arguments const& a, void*) { return a; }
    template<class Shape> static size_t get_workspace_size(Shape const&, Arguments const&) { return 0; }
    struct SharedStorage { alignas(16) Out values[BM * HD]; };
    Params const* p;
    SharedStorage* storage;
    CUTLASS_HOST_DEVICE NormRopeStore() {}
    CUTLASS_HOST_DEVICE NormRopeStore(Params const& params, SharedStorage const& s): p(&params), storage(const_cast<SharedStorage*>(&s)) {}
    using Vec = uint_bit_t<cute::min(128, Map::kElementsPerAccess * sizeof_bits<Out>::value)>;
    static constexpr int VL = sizeof(Vec) / sizeof(Out);
    template<class GT, class RT, class CT, class Shape>
    struct Callbacks: cutlass::epilogue::threadblock::EmptyCallbacks {
        GT global; RT regs; CT coords; Shape shape;
        Params const* p; SharedStorage* storage; int tid, row0, col0;
        CUTLASS_DEVICE void begin_step(int) { clear(regs); }
        template<class A, class I, int F>
        CUTLASS_DEVICE auto visit(int, int, int, int idx,
            cutlass::Array<A,F> const&, cutlass::Array<I,F> const& input) {
            auto r = recast<cutlass::Array<Out,F>>(coalesce(regs));
            r(idx) = cutlass::NumericArrayConverter<Out,I,F>{}(input);
            return input;
        }
        CUTLASS_DEVICE void end_step(int step) {
            auto src = filter(regs);
            auto co = filter(coords(_,_,_,step));
            CUTLASS_PRAGMA_UNROLL
            for (int i=0; i<size(src); ++i) {
                int row = int(get<0>(co(i))) - row0;
                int col = int(get<1>(co(i))) - col0;
                *reinterpret_cast<Vec*>(&storage->values[row * HD + col]) = src(i);
            }
        }
        CUTLASS_DEVICE void end_epilogue() {
            __syncthreads();
            int lane = tid % 32;
            for (int row=tid/32; row<BM; row+=Threads/32) {
                if (row0+row >= int(get<0>(shape))) continue;
                float sq = 0;
                CUTLASS_PRAGMA_UNROLL
                for (int col=lane*4; col<HD; col+=128) {
                    CUTLASS_PRAGMA_UNROLL
                    for (int item=0; item<4; ++item) {
                        float v=float(storage->values[row*HD+col+item]);
                        sq=fmaf(v,v,sq);
                    }
                }
                for (int offset=16; offset; offset/=2) sq += __shfl_down_sync(0xffffffff,sq,offset);
                float inv = rsqrtf(__shfl_sync(0xffffffff,sq,0) / HD + p->eps);
                CUTLASS_PRAGMA_UNROLL
                for (int pair=lane; pair<HD/2; pair+=32) {
                    int c = pair*2;
                    float a = float(storage->values[row*HD+c]);
                    float b = float(storage->values[row*HD+c+1]);
                    // Match K2: float norm, zero-centered weights, low precision
                    // norm result, then float RoPE with an FMA, then output cast.
                    a = float(Out((a*inv)*(p->norm_scale[c]+1.f)));
                    b = float(Out((b*inv)*(p->norm_scale[c+1]+1.f)));
                    const float* f=p->freqs+((row0+row)*(HD/2)+pair)*4;
                    int64_t index=int64_t(row0+row)*int(get<1>(shape))+col0+c;
                    p->output[index]=Out(fmaf(f[0],a,f[1]*b));
                    p->output[index+1]=Out(fmaf(f[2],a,f[3]*b));
                }
            }
        }
    };
    template<class Shape>
    CUTLASS_DEVICE auto get_callbacks(cutlass::gemm::GemmCoord tile,int tid,Shape shape) {
        auto tensor=make_tensor(make_gmem_ptr(p->output),shape,Stride<int64_t,_1,int64_t>{int(get<1>(shape)),_1{},int64_t(get<0>(shape))*int(get<1>(shape))});
        auto g=recast<Vec>(group_modes<3,6>(Map::partition(tensor,tid,tile)));
        auto r=make_tensor_like(take<0,3>(g));
        auto c=outer_partition(group_modes<3,6>(Map::partition(make_identity_tensor(shape),tid,tile)),cute::Shape<Int<VL>>{},(_0{}));
        return Callbacks<decltype(g),decltype(r),decltype(c),Shape>{ {}, g,r,c,shape,p,storage,tid,tile.m()*BM,tile.n()*HD };
    }
};

template<class Out>
struct QKGemm {
    static constexpr int BM=64,HD=128,Threads=128;
    using TB=cutlass::gemm::GemmShape<BM,HD,128>;
    using Warp=cutlass::gemm::GemmShape<32,64,128>;
    using Map=cutlass::epilogue::threadblock::OutputTileThreadLayout<TB,Warp,Out,8,1>;
    using Acc=cutlass::epilogue::threadblock::VisitorAccFetch;
    using Col=cutlass::epilogue::threadblock::VisitorColBroadcast<Map,float,Stride<_1,_0,int32_t>>;
    using Row=cutlass::epilogue::threadblock::VisitorRowBroadcast<Map,float,Stride<_0,_1,int32_t>>;
    using MulF=cutlass::epilogue::threadblock::VisitorCompute<cutlass::multiplies,float,float,cutlass::FloatRoundStyle::round_to_nearest>;
    using MulO=cutlass::epilogue::threadblock::VisitorCompute<cutlass::multiplies,Out,float,cutlass::FloatRoundStyle::round_to_nearest>;
    using XScale=cutlass::epilogue::threadblock::Sm80EVT<MulF,Acc,Col>;
    using Project=cutlass::epilogue::threadblock::Sm80EVT<MulO,XScale,Row>;
    using Store=NormRopeStore<Map,Out,BM,HD,Threads>;
    using EVT=cutlass::epilogue::threadblock::Sm80EVT<Store,Project>;
    using Kernel=typename cutlass::gemm::kernel::DefaultGemmWithVisitor<
        cutlass::int4b_t,cutlass::layout::RowMajor,cutlass::ComplexTransform::kNone,32,
        cutlass::int4b_t,cutlass::layout::ColumnMajor,cutlass::ComplexTransform::kNone,32,
        Out,cutlass::layout::RowMajor,8,int32_t,float,cutlass::arch::OpClassTensorOp,cutlass::arch::Sm89,
        TB,Warp,cutlass::gemm::GemmShape<16,8,64>,EVT,cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
        3,cutlass::arch::OpMultiplyAddSaturate,1>::GemmKernel;
    using Gemm=cutlass::gemm::device::GemmUniversalAdapter<Kernel>;
    static int run(void* a,void* w,void* xs,void* ws,void* norm,void* freqs,void* output,int m,int n,int k,float eps,cudaStream_t stream) {
        typename XScale::Arguments x_args{{},{static_cast<float*>(xs),0.f,{_1{},_0{},m}}, {}};
        typename Project::Arguments p_args{x_args,{static_cast<float*>(ws),0.f,{_0{},_1{},n}}, {}};
        typename EVT::Arguments cb{p_args,{static_cast<Out*>(output),static_cast<float*>(norm),static_cast<float*>(freqs),eps}};
        typename Gemm::Arguments args(cutlass::gemm::GemmUniversalMode::kGemm,{m,n,k},1,cb,
            static_cast<cutlass::int4b_t*>(a),static_cast<cutlass::int4b_t*>(w),nullptr,nullptr,
            int64_t(m)*k,int64_t(n)*k,0,0,k,k,0,0);
        Gemm gemm;auto status=gemm.can_implement(args);
        if(status!=cutlass::Status::kSuccess)return int(status);
        if(Gemm::get_workspace_size(args))return -1;
        status=gemm.initialize(args,nullptr,stream);
        if(status!=cutlass::Status::kSuccess)return int(status);
        return int(gemm(stream));
    }
};
}
EXPORT int k2_int4_qk_norm_rope(void* a,void* w,void* xs,void* ws,void* norm,void* freqs,void* output,int m,int n,int k,int dtype,float eps,void* stream) {
    if(m<=0 || n%128 || k%256)return -2;
    if(dtype==0)return QKGemm<cutlass::bfloat16_t>::run(a,w,xs,ws,norm,freqs,output,m,n,k,eps,static_cast<cudaStream_t>(stream));
    if(dtype==1)return QKGemm<cutlass::half_t>::run(a,w,xs,ws,norm,freqs,output,m,n,k,eps,static_cast<cudaStream_t>(stream));
    return -3;
}
