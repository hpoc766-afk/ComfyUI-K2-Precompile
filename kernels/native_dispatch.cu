// SPDX-License-Identifier: GPL-3.0-only
// Native launch boundary. Quantizers/GEMMs are linked from comfy-kitchen
// (Apache-2.0) at build time; this file does not call its Python bindings.
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <stdexcept>
#ifdef _WIN32
#define EXPORT extern "C" __declspec(dllexport)
#else
#define EXPORT extern "C" __attribute__((visibility("default")))
#endif
extern "C" void launch_quantize_int4_rowwise_convrot64_kernel(const void*,void*,void*,int64_t,int64_t,int,int,bool,uint64_t,cudaStream_t);
extern "C" bool launch_cutlass_int4_dequant(const void*,const void*,const void*,const void*,const void*,void*,int64_t,int64_t,int64_t,int,cudaStream_t);
extern "C" void launch_int4_linear_kernel(const void*,const void*,const void*,const void*,const void*,void*,int64_t,int64_t,int64_t,bool,int,int,cudaStream_t);
extern "C" void launch_quantize_int8_rowwise_kernel(const void*,void*,void*,int64_t,int64_t,int,bool,uint64_t,cudaStream_t);
extern "C" void launch_quantize_int8_rowwise_convrot64_kernel(const void*,void*,void*,int64_t,int64_t,int,int,bool,int,uint64_t,cudaStream_t);
extern "C" void launch_dequantize_int8_linear_kernel(const void*,const void*,const void*,const void*,void*,int64_t,int64_t,int64_t,bool,int,int,cudaStream_t);
extern "C" void launch_quantize_fp8_kernel(const void*,void*,const void*,int64_t,int,int,cudaStream_t);
extern "C" void launch_dequantize_fp8_kernel(const void*,void*,const void*,int64_t,int,int,cudaStream_t);

EXPORT void k2_quant4(const void* x,void* q,void* s,int64_t m,int64_t k,int group,int dtype,void* stream){
    launch_quantize_int4_rowwise_convrot64_kernel(x,q,s,m,k,group,dtype,false,0,(cudaStream_t)stream);
}
EXPORT void k2_gemm4(const void* a,const void* w,const void* xs,const void* ws,const void* bias,void* out,int64_t m,int64_t n,int64_t k,int dtype,void* stream){
    if(!launch_cutlass_int4_dequant(a,w,xs,ws,bias,out,m,n,k,dtype,(cudaStream_t)stream))
        launch_int4_linear_kernel(a,w,xs,ws,bias,out,m,n,k,bias!=nullptr,dtype,0,(cudaStream_t)stream);
}
EXPORT void k2_quant8(const void* x,void* q,void* s,int64_t m,int64_t k,int group,int dtype,void* stream){
    if(group)launch_quantize_int8_rowwise_convrot64_kernel(x,q,s,m,k,group,dtype,false,0,0,(cudaStream_t)stream);
    else launch_quantize_int8_rowwise_kernel(x,q,s,m,k,dtype,false,0,(cudaStream_t)stream);
}
EXPORT void k2_dequant8(const void* a,const void* xs,const void* ws,const void* bias,void* out,int64_t m,int64_t n,int64_t ns,int dtype,void* stream){
    launch_dequantize_int8_linear_kernel(a,xs,ws,bias,out,m,n,ns,bias!=nullptr,dtype,dtype,(cudaStream_t)stream);
}
EXPORT void k2_fp8(const void* x,void* y,const void* s,int64_t n,int src,int dst,bool quant,void* stream){
    if(quant)launch_quantize_fp8_kernel(x,y,s,n,src,dst,(cudaStream_t)stream);
    else launch_dequantize_fp8_kernel(x,y,s,n,src,dst,(cudaStream_t)stream);
}
namespace {
template<class T> __device__ float fp(T x){return float(x);}
template<class T> __device__ T low(float x){return T(x);}
template<class T> __global__ void norm_mod(const T* x,const float* w,const T* scale,const T* shift,T* out,int d,int length,int cut,int batch,float eps){
    int row=blockIdx.x,tid=threadIdx.x;float sum=0;
    for(int i=tid*4;i<d;i+=512){
        #pragma unroll
        for(int j=0;j<4;++j){float v=i+j<d?fp(x[int64_t(row)*d+i+j]):0.f;sum=fmaf(v,v,sum);}
    }
    for(int off=16;off;off>>=1)sum+=__shfl_down_sync(0xffffffff,sum,off);
    __shared__ float sums[4];
    if((tid&31)==0)sums[tid>>5]=sum;
    __syncthreads();
    if(tid==0)sums[0]=((sums[2]+sums[0])+(sums[3]+sums[1]))/float(d);
    __syncthreads();float inv=rsqrtf(sums[0]+eps);
    int b=row/length;if(cut>=0&&row%length>=cut)b+=batch;
    for(int col=tid;col<d;col+=128){
        T y=low<T>((fp(x[int64_t(row)*d+col])*inv)*(w[col]+1.f));
        if(scale){T factor=low<T>(fp(scale[int64_t(b)*d+col])+1.f);y=low<T>(fp(y)*fp(factor));y=low<T>(fp(y)+fp(shift[int64_t(b)*d+col]));}
        out[int64_t(row)*d+col]=y;
    }
}
template<class T> __global__ void elementwise(const T* a,const T* b,const T* r,T* y,int64_t n,int mode,int width){
    int64_t i=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;if(i>=n)return;
    if(mode==2){T z=low<T>(fp(b[i])*fp(a[i%width]));y[i]=low<T>(fp(r[i])+fp(z));return;}
    float v=fp(a[i]);T z;
    if(mode==0)z=low<T>(v/(1.f+expf(-v)));
    else z=low<T>(1.f/(1.f+expf(-v)));
    y[i]=b?low<T>(fp(z)*fp(b[i])):z;
}
template<class T> __global__ void rope(const T* q,const T* k,const float* f,T* qo,T* ko,int length,int qh,int kh,int dim,int fb){
    int row=blockIdx.x;int h=row%(qh+kh),token=(row/(qh+kh))%length,b=row/((qh+kh)*length);
    auto src=h<qh?q:k;auto dst=h<qh?qo:ko;int heads=h<qh?qh:kh,head=h<qh?h:h-qh;
    for(int p=threadIdx.x;p<dim/2;p+=blockDim.x){int64_t pos=((int64_t(b)*heads+head)*length+token)*dim+p*2;
        float a=fp(src[pos]),v=fp(src[pos+1]);const float* rot=f+((int64_t(fb==1?0:b)*length+token)*(dim/2)+p)*4;
        dst[pos]=low<T>(fmaf(rot[0],a,rot[1]*v));dst[pos+1]=low<T>(fmaf(rot[2],a,rot[3]*v));}
}
}
EXPORT void k2_norm(const void* x,const void* w,const void* scale,const void* shift,void* y,int rows,int d,int length,int cut,int batch,float eps,int dtype,void* stream){
    if(dtype==2)norm_mod<<<rows,128,0,(cudaStream_t)stream>>>((const __nv_bfloat16*)x,(const float*)w,(const __nv_bfloat16*)scale,(const __nv_bfloat16*)shift,(__nv_bfloat16*)y,d,length,cut,batch,eps);
    else if(dtype==1)norm_mod<<<rows,128,0,(cudaStream_t)stream>>>((const __half*)x,(const float*)w,(const __half*)scale,(const __half*)shift,(__half*)y,d,length,cut,batch,eps);
    else norm_mod<<<rows,128,0,(cudaStream_t)stream>>>((const float*)x,(const float*)w,(const float*)scale,(const float*)shift,(float*)y,d,length,cut,batch,eps);
}
EXPORT void k2_elementwise(const void* a,const void* b,const void* r,void* y,int64_t n,int mode,int width,int dtype,void* stream){
    int grid=int((n+255)/256);
    if(dtype==2)elementwise<<<grid,256,0,(cudaStream_t)stream>>>((const __nv_bfloat16*)a,(const __nv_bfloat16*)b,(const __nv_bfloat16*)r,(__nv_bfloat16*)y,n,mode,width);
    else if(dtype==1)elementwise<<<grid,256,0,(cudaStream_t)stream>>>((const __half*)a,(const __half*)b,(const __half*)r,(__half*)y,n,mode,width);
    else elementwise<<<grid,256,0,(cudaStream_t)stream>>>((const float*)a,(const float*)b,(const float*)r,(float*)y,n,mode,width);
}
EXPORT void k2_rope(const void* q,const void* k,const void* f,void* qo,void* ko,int batch,int length,int qh,int kh,int dim,int fb,int dtype,void* stream){
    int rows=batch*length*(qh+kh);
    if(dtype==2)rope<<<rows,32,0,(cudaStream_t)stream>>>((const __nv_bfloat16*)q,(const __nv_bfloat16*)k,(const float*)f,(__nv_bfloat16*)qo,(__nv_bfloat16*)ko,length,qh,kh,dim,fb);
    else if(dtype==1)rope<<<rows,32,0,(cudaStream_t)stream>>>((const __half*)q,(const __half*)k,(const float*)f,(__half*)qo,(__half*)ko,length,qh,kh,dim,fb);
    else rope<<<rows,32,0,(cudaStream_t)stream>>>((const float*)q,(const float*)k,(const float*)f,(float*)qo,(float*)ko,length,qh,kh,dim,fb);
}
