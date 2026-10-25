// SPDX-License-Identifier: GPL-3.0-only
// ConvRot stages/packing reuse comfy-kitchen Apache-2.0 helper functions;
// build_norm_quant.py extracts the unmodified helper prefix from its source.
#include "norm_quant_helpers.cuh"
#ifdef _WIN32
#define EXPORT extern "C" __declspec(dllexport)
#else
#define EXPORT extern "C" __attribute__((visibility("default")))
#endif
namespace {
template<class T,int THREADS,int GROUP>
__global__ void norm_mod_quant(const T* x,const float* weight,const T* scale,const T* shift,
                               int8_t* q,float* scales,int K,int length,int cut,int batch,float eps){
    constexpr int GROUP_THREADS=GROUP==256?64:GROUP/4;
    constexpr int GROUPS=THREADS/GROUP_THREADS,WARPS=THREADS/32;
    extern __shared__ __align__(16) unsigned char raw[];
    T* values=reinterpret_cast<T*>(raw);T* tmp=values+K;
    __shared__ float norm_sums[4],warp_smem[WARPS],block_smem;
    int row=blockIdx.x,tid=threadIdx.x;int64_t row_offset=int64_t(row)*K;
    if(tid<128){
        float sum=0;
        for(int col=tid*4;col<K;col+=512){
            #pragma unroll
            for(int j=0;j<4;++j){float v=col+j<K?to_float(x[row_offset+col+j]):0.f;sum=fmaf(v,v,sum);}
        }
        for(int off=16;off;off>>=1)sum+=__shfl_down_sync(0xffffffff,sum,off);
        if((tid&31)==0)norm_sums[tid>>5]=sum;
    }
    __syncthreads();
    if(tid==0){float sum=__fadd_rn(__fadd_rn(norm_sums[2],norm_sums[0]),__fadd_rn(norm_sums[3],norm_sums[1]));
        norm_sums[0]=rsqrtf(__fadd_rn(__fdiv_rn(sum,float(K)),eps));}
    __syncthreads();float inv=norm_sums[0];int b=row/length;if(cut>=0&&row%length>=cut)b+=batch;
    for(int col=tid;col<K;col+=THREADS){
        T y=from_float<T>(__fmul_rn(__fmul_rn(to_float(x[row_offset+col]),inv),__fadd_rn(weight[col],1.f)));
        T factor=from_float<T>(__fadd_rn(to_float(scale[int64_t(b)*K+col]),1.f));
        y=from_float<T>(__fmul_rn(to_float(y),to_float(factor)));
        values[col]=from_float<T>(__fadd_rn(to_float(y),to_float(shift[int64_t(b)*K+col])));
    }
    __syncthreads();
    int sub=tid/GROUP_THREADS,lane=tid%GROUP_THREADS;
    T* buf0=tmp+sub*(2*GROUP);T* buf1=buf0+GROUP;float abs_max=0.f;
    int groups=K/GROUP,iters=(groups+GROUPS-1)/GROUPS;
    for(int it=0;it<iters;++it){
        int group=it*GROUPS+sub;bool active=group<groups;int base=lane*4,col=group*GROUP;
        T zero=from_float<T>(0.f);
        T x0=active?values[col+base]:zero,x1=active?values[col+base+1]:zero;
        T x2=active?values[col+base+2]:zero,x3=active?values[col+base+3]:zero;
        convrot_h4_store_typed(x0,x1,x2,x3,buf1,base);
        if constexpr(GROUP==256){
            __syncwarp();convrot_fht_stage64_vec2_typed(buf1,buf0,lane,4);
            __syncwarp();convrot_fht_stage64_vec2_typed(buf0,buf1,lane,16);__syncthreads();
            if(active)abs_max=fmaxf(abs_max,convrot_fht_stage64_store_absmax_typed(buf1,values+col,lane,64));
        }else{
            __syncthreads();
            if constexpr(GROUP==16){if(active)abs_max=fmaxf(abs_max,convrot_fht_stage64_store_absmax_typed(buf1,values+col,lane,4));}
            else {convrot_fht_stage64_typed(buf1,buf0,lane,4);__syncthreads();
                if(active)abs_max=fmaxf(abs_max,convrot_fht_stage64_store_absmax_typed(buf0,values+col,lane,16));}
        }
        __syncthreads();
    }
    abs_max=block_reduce_max<WARPS>(abs_max,warp_smem,&block_smem);
    float row_scale=fmaxf(finite_absmax_for_int4_scale<T>(abs_max)*(1.f/float(kInt4Max)),1.e-10f);
    if(tid==0)scales[row]=row_scale;float inverse=1.f/row_scale;
    auto words=reinterpret_cast<uint32_t*>(q+int64_t(row)*(K/2));
    for(int word=tid;word<K/8;word+=THREADS)
        words[word]=quantize_int4_pack4_word<T,false>(values,word*8,inverse,0,row_offset);
}
template<class T,int GROUP>
int launch(const void* x,const void* norm,const void* scale,const void* shift,void* q,void* scales,
           int rows,int K,int length,int cut,int batch,float eps,cudaStream_t stream){
    constexpr int THREADS=256;
    constexpr int GROUP_THREADS=GROUP==256?64:GROUP/4;
    size_t smem=(size_t(K)+(THREADS/GROUP_THREADS)*2*GROUP)*sizeof(T);
    auto kernel=norm_mod_quant<T,THREADS,GROUP>;
    auto status=cudaFuncSetAttribute(kernel,cudaFuncAttributeMaxDynamicSharedMemorySize,int(smem));
    if(status!=cudaSuccess)return int(status);
    kernel<<<rows,THREADS,smem,stream>>>((const T*)x,(const float*)norm,(const T*)scale,(const T*)shift,
                                       (int8_t*)q,(float*)scales,K,length,cut,batch,eps);
    return int(cudaGetLastError());
}
template<class T>
int dispatch(const void* x,const void* norm,const void* scale,const void* shift,void* q,void* scales,
             int rows,int K,int length,int cut,int batch,float eps,int group,cudaStream_t stream){
    if(group==16)return launch<T,16>(x,norm,scale,shift,q,scales,rows,K,length,cut,batch,eps,stream);
    if(group==64)return launch<T,64>(x,norm,scale,shift,q,scales,rows,K,length,cut,batch,eps,stream);
    return launch<T,256>(x,norm,scale,shift,q,scales,rows,K,length,cut,batch,eps,stream);
}
}
EXPORT int k2_norm_mod_quant4(const void* x,const void* norm,const void* scale,const void* shift,
                             void* q,void* scales,int rows,int K,int length,int cut,int batch,
                             float eps,int group,int dtype,void* stream){
    if(rows<=0||K<=0||length<=0||(group!=16&&group!=64&&group!=256)||K%group||K%8)return int(cudaErrorInvalidValue);
    if(dtype==2)return dispatch<__nv_bfloat16>(x,norm,scale,shift,q,scales,rows,K,length,cut,batch,eps,group,(cudaStream_t)stream);
    if(dtype==1)return dispatch<__half>(x,norm,scale,shift,q,scales,rows,K,length,cut,batch,eps,group,(cudaStream_t)stream);
    return int(cudaErrorInvalidValue);
}
