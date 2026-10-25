// SPDX-License-Identifier: GPL-3.0-only
// SM89 packed INT4 fused projections. 128-row tiles reuse weights across rows;
// cp.async double buffers operands. Each warp owns independent MMA fragments.
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda/atomic>
#include <cstdint>
#ifdef _WIN32
#define EXPORT extern "C" __declspec(dllexport)
#else
#define EXPORT extern "C" __attribute__((visibility("default")))
#endif
namespace {
constexpr int BM=128, PACKED_K=32;
struct AFragment { uint32_t x[4]; };
struct BFragment { uint32_t x[2]; };
__device__ __forceinline__ void copy16(void* dst,const void* src,bool valid){
    unsigned address=unsigned(__cvta_generic_to_shared(dst));
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" ::
                 "r"(address),"l"(src),"r"(valid?16:0):"memory");
}
__device__ __forceinline__ void commit(){asm volatile("cp.async.commit_group;":::"memory");}
__device__ __forceinline__ void await(){asm volatile("cp.async.wait_group 0;":::"memory");}
__device__ __forceinline__ void stage_a(int8_t* dst,const int8_t* src,int row,int kk,int m,int k){
    for(int i=threadIdx.x;i<BM*2;i+=blockDim.x){
        int r=row+i/2;bool valid=r<m;
        const int8_t* p=valid?src+int64_t(r)*(k/2)+kk/2+(i&1)*16:src;
        copy16(dst+i*16,p,valid);
    }
}
template<int BN> __device__ __forceinline__ void stage_b(int8_t* dst,const int8_t* src,int col,int kk,int n,int k){
    for(int i=threadIdx.x;i<BN*2;i+=blockDim.x){
        int r=col+i/2;bool valid=r<n;
        const int8_t* p=valid?src+int64_t(r)*(k/2)+kk/2+(i&1)*16:src;
        copy16(dst+i*16,p,valid);
    }
}
__device__ __forceinline__ AFragment load_a(const int8_t* a,int row,int lane){
    int r=row+(lane>>2),c=(lane&3)*4;AFragment f;
    f.x[0]=*reinterpret_cast<const uint32_t*>(a+r*PACKED_K+c);
    f.x[1]=*reinterpret_cast<const uint32_t*>(a+(r+8)*PACKED_K+c);
    f.x[2]=*reinterpret_cast<const uint32_t*>(a+r*PACKED_K+c+16);
    f.x[3]=*reinterpret_cast<const uint32_t*>(a+(r+8)*PACKED_K+c+16);return f;
}
__device__ __forceinline__ BFragment load_b(const int8_t* b,int col,int lane){
    int r=col+(lane>>2),c=(lane&3)*4;BFragment f;
    f.x[0]=*reinterpret_cast<const uint32_t*>(b+r*PACKED_K+c);
    f.x[1]=*reinterpret_cast<const uint32_t*>(b+r*PACKED_K+c+16);return f;
}
__device__ __forceinline__ void mma(int* d,AFragment const& a,BFragment const& b){
    asm volatile("mma.sync.aligned.m16n8k64.row.col.satfinite.s32.s4.s4.s32 "
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                 : "+r"(d[0]),"+r"(d[1]),"+r"(d[2]),"+r"(d[3])
                 : "r"(a.x[0]),"r"(a.x[1]),"r"(a.x[2]),"r"(a.x[3]),"r"(b.x[0]),"r"(b.x[1]));
}
template<class T> __device__ __forceinline__ T projected(int value,float xs,float ws){
    return T(__fmul_rn(__fmul_rn(float(value),xs),ws));
}
template<class T> __device__ __forceinline__ T visible_load(const T* p);
template<> __device__ __forceinline__ __half visible_load(const __half* p){
    return __ushort_as_half(__ldcg(reinterpret_cast<const unsigned short*>(p)));
}
template<> __device__ __forceinline__ __nv_bfloat16 visible_load(const __nv_bfloat16* p){
    return __ushort_as_bfloat16(__ldcg(reinterpret_cast<const unsigned short*>(p)));
}
// Both GEMMs share A tiles; each warp owns 32x32 outputs from each matrix.
template<class T> __global__ __launch_bounds__(256,2) void swiglu_kernel(
    const int8_t* a,const int8_t* wg,const int8_t* wu,const float* xs,
    const float* sg,const float* su,T* output,int m,int n,int k){
    __shared__ __align__(16) int8_t sa[2][BM*PACKED_K];
    __shared__ __align__(16) int8_t bg[2][64*PACKED_K],bu[2][64*PACKED_K];
    int lane=threadIdx.x&31,warp=threadIdx.x>>5;
    int mr=(warp/2)*32,nc=(warp%2)*32,row0=blockIdx.x*BM,col0=blockIdx.y*64;
    int g[32]={},u[32]={},stage=0;
    stage_a(sa[0],a,row0,0,m,k);stage_b<64>(bg[0],wg,col0,0,n,k);stage_b<64>(bu[0],wu,col0,0,n,k);commit();
    for(int kk=0;kk<k;kk+=64){
        await();__syncthreads();
        if(kk+64<k){stage_a(sa[stage^1],a,row0,kk+64,m,k);stage_b<64>(bg[stage^1],wg,col0,kk+64,n,k);stage_b<64>(bu[stage^1],wu,col0,kk+64,n,k);commit();}
        auto a0=load_a(sa[stage],mr,lane),a1=load_a(sa[stage],mr+16,lane);
        #pragma unroll
        for(int j=0;j<4;++j){
            auto gfrag=load_b(bg[stage],nc+j*8,lane),ufrag=load_b(bu[stage],nc+j*8,lane);
            mma(g+j*4,a0,gfrag);mma(g+16+j*4,a1,gfrag);
            mma(u+j*4,a0,ufrag);mma(u+16+j*4,a1,ufrag);
        }
        __syncthreads();stage^=1;
    }
    #pragma unroll
    for(int i=0;i<32;++i){
        int r=row0+mr+(i>=16?16:0)+(lane>>2)+((i&2)?8:0);
        int c=col0+nc+((i%16)/4)*8+(lane&3)*2+(i&1);
        if(r<m&&c<n){
            T gv=projected<T>(g[i],xs[r],sg[c]),uv=projected<T>(u[i],xs[r],su[c]);
            T activated=T(float(gv)/(1.f+expf(-float(gv))));
            output[int64_t(r)*n+c]=T(__fmul_rn(float(activated),float(uv)));
        }
    }
}
// One CTA owns a complete head. MMA staging storage is reused by Q/K's
// rounded projection values after every outstanding async copy has completed.
template<class T> __global__ __launch_bounds__(256,2) void qkv_kernel(
    const int8_t* a,const int8_t* wq,const int8_t* wk,const int8_t* wv,const int8_t* wg,
    const float* xs,const float* sq,const float* sk,const float* sv,const float* sg,
    const float* nq,const float* nk,const float* freq,
    T* qo,T* ko,T* vo,T* go,int m,int qwidth,int kvwidth,int k,float eq,float ek){
    int tile=blockIdx.y,qtiles=qwidth/128,kvtiles=kvwidth/128,type,ntile;
    if(tile<qtiles){type=0;ntile=tile;}
    else if(tile<qtiles+kvtiles){type=1;ntile=tile-qtiles;}
    else if(tile<qtiles+2*kvtiles){type=2;ntile=tile-qtiles-kvtiles;}
    else{type=3;ntile=tile-qtiles-2*kvtiles;}
    const int8_t* w=type==0?wq:type==1?wk:type==2?wv:wg;
    const float* ws=type==0?sq:type==1?sk:type==2?sv:sg;
    T* dest=type==0?qo:type==1?ko:type==2?vo:go;int n=(type==1||type==2)?kvwidth:qwidth;
    __shared__ __align__(16) unsigned char storage[BM*128*2];
    int8_t* sa=reinterpret_cast<int8_t*>(storage);int8_t* sb=sa+2*BM*PACKED_K;
    T* values=reinterpret_cast<T*>(storage);
    int lane=threadIdx.x&31,warp=threadIdx.x>>5;
    int mr=(warp/2)*32,nc=(warp%2)*64,row0=blockIdx.x*BM,col0=ntile*128;
    int acc[64]={},stage=0;
    stage_a(sa,a,row0,0,m,k);stage_b<128>(sb,w,col0,0,n,k);commit();
    for(int kk=0;kk<k;kk+=64){
        await();__syncthreads();
        if(kk+64<k){stage_a(sa+(stage^1)*BM*PACKED_K,a,row0,kk+64,m,k);stage_b<128>(sb+(stage^1)*128*PACKED_K,w,col0,kk+64,n,k);commit();}
        auto a0=load_a(sa+stage*BM*PACKED_K,mr,lane),a1=load_a(sa+stage*BM*PACKED_K,mr+16,lane);
        #pragma unroll
        for(int j=0;j<8;++j){auto bf=load_b(sb+stage*128*PACKED_K,nc+j*8,lane);mma(acc+j*4,a0,bf);mma(acc+32+j*4,a1,bf);}
        __syncthreads();stage^=1;
    }
    #pragma unroll
    for(int i=0;i<64;++i){
        int r=row0+mr+(i>=32?16:0)+(lane>>2)+((i&2)?8:0);
        int c=col0+nc+((i%32)/4)*8+(lane&3)*2+(i&1);
        if(r<m){T value=projected<T>(acc[i],xs[r],ws[c]);
            if(type<2)values[(r-row0)*128+c-col0]=value;
            else dest[int64_t(r)*n+c]=type==3?T(1.f/(1.f+expf(-float(value)))):value;
        }
    }
    if(type>=2)return;__syncthreads();
    const float* norm=type==0?nq:nk;float eps=type==0?eq:ek;
    for(int r=warp;r<BM;r+=8){
        int token=row0+r;if(token>=m)continue;float sum=0.f;
        #pragma unroll
        for(int j=0;j<4;++j){float v=float(values[r*128+lane*4+j]);sum=fmaf(v,v,sum);}
        for(int d=16;d;d>>=1)sum+=__shfl_down_sync(0xffffffff,sum,d);
        float inv=rsqrtf(__shfl_sync(0xffffffff,sum,0)/128.f+eps);
        for(int pair=lane;pair<64;pair+=32){
            int c=pair*2;float av=float(T((float(values[r*128+c])*inv)*(norm[c]+1.f)));
            float bv=float(T((float(values[r*128+c+1])*inv)*(norm[c+1]+1.f)));
            const float* f=freq+(int64_t(token)*64+pair)*4;int64_t out=int64_t(token)*n+col0+c;
            dest[out]=T(fmaf(f[0],av,f[1]*bv));dest[out+1]=T(fmaf(f[2],av,f[3]*bv));
        }
    }
}
// Channel tiles publish residual results; the final tile continues full-row
// RMSNorm/modulation. No block spins waiting for unscheduled work.
template<class T> __global__ __launch_bounds__(256,2) void residual_norm_kernel(
    const int8_t* a,const int8_t* w,const float* xs,const float* ws,
    const T* gate,const T* residual,const float* norm,const T* scale,const T* shift,
    T* output,T* normalized,int* counters,int m,int n,int k,float eps){
    __shared__ __align__(16) int8_t sa[2][BM*PACKED_K],sb[2][128*PACKED_K];
    int lane=threadIdx.x&31,warp=threadIdx.x>>5;
    int mr=(warp/2)*32,nc=(warp%2)*64,row0=blockIdx.x*BM,col0=blockIdx.y*128;
    int acc[64]={},stage=0;
    stage_a(sa[0],a,row0,0,m,k);stage_b<128>(sb[0],w,col0,0,n,k);commit();
    for(int kk=0;kk<k;kk+=64){
        await();__syncthreads();
        if(kk+64<k){stage_a(sa[stage^1],a,row0,kk+64,m,k);stage_b<128>(sb[stage^1],w,col0,kk+64,n,k);commit();}
        auto a0=load_a(sa[stage],mr,lane),a1=load_a(sa[stage],mr+16,lane);
        #pragma unroll
        for(int j=0;j<8;++j){auto bf=load_b(sb[stage],nc+j*8,lane);mma(acc+j*4,a0,bf);mma(acc+32+j*4,a1,bf);}
        __syncthreads();stage^=1;
    }
    #pragma unroll
    for(int i=0;i<64;++i){
        int r=row0+mr+(i>=32?16:0)+(lane>>2)+((i&2)?8:0);
        int c=col0+nc+((i%32)/4)*8+(lane&3)*2+(i&1);
        if(r<m){T p=projected<T>(acc[i],xs[r],ws[c]);T g=T(__fmul_rn(float(p),float(gate[c])));
            output[int64_t(r)*n+c]=T(__fadd_rn(float(residual[int64_t(r)*n+c]),float(g)));}
    }
    __threadfence();__syncthreads();__shared__ int last;
    if(threadIdx.x==0){cuda::atomic_ref<int,cuda::thread_scope_device> count(counters[blockIdx.x]);last=count.fetch_add(1,cuda::memory_order_acq_rel)==int(gridDim.y)-1;}
    __syncthreads();if(!last)return;
    __shared__ float sums[8];int group=threadIdx.x/128,tid=threadIdx.x%128;
    for(int local=group;local<BM;local+=2){
        int r=row0+local;float sum=0;
        for(int c=tid*4;c<n;c+=512){
            #pragma unroll
            for(int j=0;j<4;++j){float v=r<m?float(visible_load(output+int64_t(r)*n+c+j)):0.f;sum=fmaf(v,v,sum);}
        }
        for(int d=16;d;d>>=1)sum+=__shfl_down_sync(0xffffffff,sum,d);
        if((tid&31)==0)sums[group*4+(tid>>5)]=sum;__syncthreads();
        if(tid==0){int base=group*4;sums[base]=((sums[base+2]+sums[base])+(sums[base+3]+sums[base+1]))/float(n);}
        __syncthreads();float inv=rsqrtf(sums[group*4]+eps);
        for(int c=tid;c<n;c+=128){if(r<m){
            T z=T(__fmul_rn(__fmul_rn(float(visible_load(output+int64_t(r)*n+c)),inv),__fadd_rn(norm[c],1.f)));
            T factor=T(__fadd_rn(float(scale[c]),1.f));z=T(__fmul_rn(float(z),float(factor)));
            normalized[int64_t(r)*n+c]=T(__fadd_rn(float(z),float(shift[c])));
        }}
        __syncthreads();
    }
}
}
EXPORT int k2_int4_swiglu(const void* a,const void* wg,const void* wu,const void* xs,
    const void* sg,const void* su,void* out,int m,int n,int k,int dtype,void* stream){
    if(m<=0||n<=0||k<=0||k%256||dtype<0||dtype>1)return int(cudaErrorInvalidValue);
    dim3 grid((m+BM-1)/BM,(n+63)/64);
    if(dtype==0)swiglu_kernel<<<grid,256,0,(cudaStream_t)stream>>>((const int8_t*)a,(const int8_t*)wg,(const int8_t*)wu,(const float*)xs,(const float*)sg,(const float*)su,(__nv_bfloat16*)out,m,n,k);
    else swiglu_kernel<<<grid,256,0,(cudaStream_t)stream>>>((const int8_t*)a,(const int8_t*)wg,(const int8_t*)wu,(const float*)xs,(const float*)sg,(const float*)su,(__half*)out,m,n,k);
    return int(cudaGetLastError());
}
EXPORT int k2_int4_residual_norm(const void* a,const void* w,const void* xs,const void* ws,
    const void* gate,const void* residual,const void* norm,const void* scale,const void* shift,
    void* output,void* normalized,void* counters,int m,int n,int k,int dtype,float eps,void* stream){
    if(m<=0||n<=0||n%128||k<=0||k%256||dtype<0||dtype>1)return int(cudaErrorInvalidValue);
    dim3 grid((m+BM-1)/BM,n/128);cudaError_t status=cudaMemsetAsync(counters,0,grid.x*sizeof(int),(cudaStream_t)stream);
    if(status!=cudaSuccess)return int(status);
    if(dtype==0)residual_norm_kernel<<<grid,256,0,(cudaStream_t)stream>>>((const int8_t*)a,(const int8_t*)w,(const float*)xs,(const float*)ws,(const __nv_bfloat16*)gate,(const __nv_bfloat16*)residual,(const float*)norm,(const __nv_bfloat16*)scale,(const __nv_bfloat16*)shift,(__nv_bfloat16*)output,(__nv_bfloat16*)normalized,(int*)counters,m,n,k,eps);
    else residual_norm_kernel<<<grid,256,0,(cudaStream_t)stream>>>((const int8_t*)a,(const int8_t*)w,(const float*)xs,(const float*)ws,(const __half*)gate,(const __half*)residual,(const float*)norm,(const __half*)scale,(const __half*)shift,(__half*)output,(__half*)normalized,(int*)counters,m,n,k,eps);
    return int(cudaGetLastError());
}
EXPORT int k2_int4_qkv(const void* a,const void* wq,const void* wk,const void* wv,const void* wg,
    const void* xs,const void* sq,const void* sk,const void* sv,const void* sg,
    const void* nq,const void* nk,const void* freq,void* qo,void* ko,void* vo,void* go,
    int m,int qwidth,int kvwidth,int k,int dtype,float eq,float ek,void* stream){
    if(m<=0||qwidth<=0||kvwidth<=0||qwidth%128||kvwidth%128||k<=0||k%256||dtype<0||dtype>1)return int(cudaErrorInvalidValue);
    dim3 grid((m+BM-1)/BM,(2*qwidth+2*kvwidth)/128);
    if(dtype==0)qkv_kernel<<<grid,256,0,(cudaStream_t)stream>>>((const int8_t*)a,(const int8_t*)wq,(const int8_t*)wk,(const int8_t*)wv,(const int8_t*)wg,(const float*)xs,(const float*)sq,(const float*)sk,(const float*)sv,(const float*)sg,(const float*)nq,(const float*)nk,(const float*)freq,(__nv_bfloat16*)qo,(__nv_bfloat16*)ko,(__nv_bfloat16*)vo,(__nv_bfloat16*)go,m,qwidth,kvwidth,k,eq,ek);
    else qkv_kernel<<<grid,256,0,(cudaStream_t)stream>>>((const int8_t*)a,(const int8_t*)wq,(const int8_t*)wk,(const int8_t*)wv,(const int8_t*)wg,(const float*)xs,(const float*)sq,(const float*)sk,(const float*)sv,(const float*)sg,(const float*)nq,(const float*)nk,(const float*)freq,(__half*)qo,(__half*)ko,(__half*)vo,(__half*)go,m,qwidth,kvwidth,k,eq,ek);
    return int(cudaGetLastError());
}
