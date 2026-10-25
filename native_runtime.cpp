// SPDX-License-Identifier: GPL-3.0-only
// Native inference scheduling. Python registers immutable model metadata once;
// operator execution, VBAR leases, copies, events and GEMM submission stay here.
#include <ATen/ATen.h>
#include <ATen/ops/_scaled_mm.h>
#include <ATen/ops/_int_mm.h>
#include <ATen/ops/scaled_dot_product_attention.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/library.h>
#include <torch/csrc/utils/pybind.h>
#include <pybind11/stl.h>
#include <cuda_runtime_api.h>
#include <mutex>
#include <deque>
#include <unordered_map>
#include <memory>
#include <array>
#include <filesystem>
#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <limits>
#ifdef _WIN32
#include <windows.h>
#else
#include <dlfcn.h>
#endif
using at::Tensor;
using Tensors=std::vector<Tensor>;
using Keys=std::vector<int64_t>;
namespace {
void check(cudaError_t e){TORCH_CHECK(e==cudaSuccess,cudaGetErrorString(e));}
void* library(std::string const& path){
#ifdef _WIN32
 auto h=LoadLibraryW(std::filesystem::u8path(path).c_str());TORCH_CHECK(h,"Cannot load native library: ",path);return (void*)h;
#else
 auto h=dlopen(path.c_str(),RTLD_NOW|RTLD_LOCAL);TORCH_CHECK(h,"Cannot load native library: ",path);return h;
#endif
}
void* symbol(void* h,const char* name){
#ifdef _WIN32
 auto p=(void*)GetProcAddress((HMODULE)h,name);
#else
 auto p=dlsym(h,name);
#endif
 TORCH_CHECK(p,"Missing native symbol: ",name);return p;
}
struct API {
 using Quant=void(*)(const void*,void*,void*,int64_t,int64_t,int,int,void*);
 using Gemm=void(*)(const void*,const void*,const void*,const void*,const void*,void*,int64_t,int64_t,int64_t,int,void*);
 using Dequant=void(*)(const void*,const void*,const void*,const void*,void*,int64_t,int64_t,int64_t,int,void*);
 using FP8=void(*)(const void*,void*,const void*,int64_t,int,int,bool,void*);
 using Norm=void(*)(const void*,const void*,const void*,const void*,void*,int,int,int,int,int,float,int,void*);
 using NormQuant=int(*)(const void*,const void*,const void*,const void*,void*,void*,int,int,int,int,int,float,int,int,void*);
 using Element=void(*)(const void*,const void*,const void*,void*,int64_t,int,int,int,void*);
 using Rope=void(*)(const void*,const void*,const void*,void*,void*,int,int,int,int,int,int,int,void*);
 using Gated=int(*)(void*,void*,void*,void*,void*,void*,void*,int,int,int,int,int,void*);
 using QK=int(*)(void*,void*,void*,void*,void*,void*,void*,int,int,int,int,float,void*);
 using JointPair=int(*)(const void*,const void*,const void*,const void*,const void*,const void*,void*,int,int,int,int,void*);
 using JointNorm=int(*)(const void*,const void*,const void*,const void*,const void*,const void*,
                       const void*,const void*,const void*,void*,void*,void*,int,int,int,int,float,void*);
 using JointQkv=int(*)(const void*,const void*,const void*,const void*,const void*,const void*,
                      const void*,const void*,const void*,const void*,const void*,const void*,const void*,
                      void*,void*,void*,void*,int,int,int,int,int,float,float,void*);
 using Fault=int(*)(void*,void*,uint64_t,uint64_t,uint32_t*);
 using Unpin=void(*)(void*,void*,uint64_t,uint64_t);
 Quant quant4=nullptr,quant8=nullptr;Gemm gemm4=nullptr;Dequant dequant8=nullptr;FP8 fp8=nullptr;
 Norm norm=nullptr;Element element=nullptr;Rope rope=nullptr;Gated gated=nullptr;QK qk=nullptr;
 Fault fault=nullptr;Unpin unpin=nullptr;JointPair joint_pair=nullptr;JointQkv joint_qkv=nullptr;
 JointNorm joint_norm=nullptr;NormQuant norm_quant=nullptr;
};
API& api(){static auto* p=new API;return *p;}
struct Binding {
 int64_t key=0;int kind=0,format=0,group=256;bool fuse=false,optimize_io=true;double eps=1e-5,theta=1000;
 Tensors cpu,gpu;Keys vbar,axes,lookahead;Tensor input_scale;
 std::vector<uint32_t> signature;std::unordered_map<std::string,int64_t> stats;
 std::unordered_map<int64_t,int> routes;
};
std::unordered_map<int64_t,std::shared_ptr<Binding>>& records(){static auto* p=new std::unordered_map<int64_t,std::shared_ptr<Binding>>;return *p;}
std::recursive_mutex& registry_mutex(){static auto* p=new std::recursive_mutex;return *p;}
std::shared_ptr<Binding> get(int64_t key){std::lock_guard<std::recursive_mutex> l(registry_mutex());auto i=records().find(key);TORCH_CHECK(i!=records().end(),"Native K2 binding released: ",key);return i->second;}
int dtype_code(Tensor const& x){return x.scalar_type()==at::kBFloat16?2:x.scalar_type()==at::kHalf?1:0;}
struct DeviceBuffer {Tensor bytes;bool busy=false;uint64_t stamp=0;};
struct HostBuffer {
 Tensor bytes;cudaEvent_t copied=nullptr;bool recorded=false,valid=false;
 int64_t owner=0,slot=0;const void* source=nullptr;uint32_t version=0;size_t payload=0;uint64_t stamp=0;
 ~HostBuffer(){if(copied)cudaEventDestroy(copied);}
};
size_t setting_bytes(const char* name,size_t mib){auto value=std::getenv(name);return (value?std::stoull(value):mib)*(1ULL<<20);}
size_t buffer_capacity(size_t bytes){return ((bytes+65535)/65536)*65536;}
struct Lease {
 std::shared_ptr<Binding> record;Tensors tensors,staging;std::vector<std::shared_ptr<DeviceBuffer>> buffers;
 cudaEvent_t ready=nullptr,done=nullptr;bool pinned=false,abandoned_prefetch=false;size_t bytes=0;
 ~Lease(){
  if(pinned||!buffers.empty())cudaDeviceSynchronize();
  if(pinned){auto& v=record->vbar;api().unpin((void*)v[0],(void*)v[1],uint64_t(v[2]-v[3]),uint64_t(v[4]));}
  for(auto& block:buffers)block->busy=false;
  if(ready)cudaEventDestroy(ready);if(done)cudaEventDestroy(done);
 }
};
struct Runtime {
 int device;c10::cuda::CUDAStream transfer,side;
 std::unordered_map<int64_t,std::unique_ptr<Lease>> prefetched;
 std::deque<std::unique_ptr<Lease>> pending;
 std::vector<std::unique_ptr<HostBuffer>> hosts;
 std::vector<std::shared_ptr<DeviceBuffer>> devices;
 std::vector<cudaEvent_t> events;
 size_t host_bytes=0,device_bytes=0,prefetched_bytes=0,pending_prefetch_bytes=0;
 size_t host_limit=setting_bytes("K2_HOST_STAGING_MIB",256),device_limit=setting_bytes("K2_CAST_CACHE_MIB",8);
 size_t prefetch_limit=setting_bytes("K2_PREFETCH_MIB",128);uint64_t clock=0;
 Runtime(int d):device(d),transfer(c10::cuda::getStreamFromPool(false,d)),side(c10::cuda::getStreamFromPool(false,d)){}
 cudaEvent_t event(bool pooled){
  if(pooled&&!events.empty()){auto out=events.back();events.pop_back();return out;}
  cudaEvent_t out;check(cudaEventCreateWithFlags(&out,cudaEventDisableTiming));return out;
 }
 void recycle(cudaEvent_t& e,bool pooled){if(!e)return;if(pooled&&events.size()<512)events.push_back(e);else check(cudaEventDestroy(e));e=nullptr;}
 bool host_ready(HostBuffer const& h){if(!h.recorded)return true;auto s=cudaEventQuery(h.copied);if(s==cudaErrorNotReady)return false;check(s);return true;}
 HostBuffer* host(std::shared_ptr<Binding> const& b,int slot,Tensor const& src,bool may_wait){
  size_t n=src.nbytes(),capacity=buffer_capacity(n);if(!host_limit||capacity>host_limit)return nullptr;
  auto const& identity=b->cpu.at(slot);auto const& vc=identity.unsafeGetTensorImpl()->version_counter();uint32_t version=vc.enabled()?vc.current_version():0;
  for(auto& h:hosts)if(h->valid&&h->owner==b->key&&h->slot==slot&&h->source==identity.data_ptr()&&h->version==version&&h->payload==n){h->stamp=++clock;b->stats["host_content_hits"]++;return h.get();}
  while(true){
   HostBuffer* best=nullptr;
   for(auto& h:hosts)if(host_ready(*h)&&size_t(h->bytes.numel())>=capacity&&(!best||h->bytes.numel()<best->bytes.numel()))best=h.get();
   if(!best&&host_bytes+capacity<=host_limit){
    auto h=std::make_unique<HostBuffer>();h->bytes=at::empty({int64_t(capacity)},at::TensorOptions().dtype(at::kByte).device(at::kCPU).pinned_memory(true));
    check(cudaEventCreateWithFlags(&h->copied,cudaEventDisableTiming));host_bytes+=capacity;best=h.get();hosts.push_back(std::move(h));b->stats["host_buffer_allocations"]++;
   }else if(best)b->stats["host_buffer_reuses"]++;
   if(best){std::memcpy(best->bytes.data_ptr(),src.data_ptr(),n);best->owner=b->key;best->slot=slot;best->source=identity.data_ptr();best->version=version;best->payload=n;best->valid=true;best->stamp=++clock;b->stats["host_copy_bytes"]+=n;return best;}
   auto free=hosts.end();
   for(auto it=hosts.begin();it!=hosts.end();++it)if(host_ready(**it)&&(free==hosts.end()||(*it)->stamp<(*free)->stamp))free=it;
   if(free!=hosts.end()){host_bytes-=(*free)->bytes.numel();hosts.erase(free);continue;}
   if(!may_wait||hosts.empty())return nullptr;
   auto oldest=std::min_element(hosts.begin(),hosts.end(),[](auto const& a,auto const& b){return a->stamp<b->stamp;});
   b->stats["host_pool_waits"]++;check(cudaEventSynchronize((*oldest)->copied));
  }
 }
 Tensor destination(Tensor const& source,Lease& lease){
  auto options=source.options().device(at::Device(at::kCUDA,device)).pinned_memory(false);
  size_t capacity=buffer_capacity(source.nbytes());auto b=lease.record;
  if(!b->optimize_io||!device_limit||capacity>device_limit)return at::empty(source.sizes(),options);
  std::shared_ptr<DeviceBuffer> found;
  for(auto& p:devices)if(!p->busy&&size_t(p->bytes.numel())>=capacity&&(!found||p->bytes.numel()<found->bytes.numel()))found=p;
  if(found)b->stats["device_buffer_reuses"]++;
  if(!found){
   for(auto it=devices.begin();it!=devices.end()&&device_bytes+capacity>device_limit;){
    if(!(*it)->busy){device_bytes-=(*it)->bytes.numel();it=devices.erase(it);}else ++it;
   }
   if(device_bytes+capacity>device_limit){b->stats["device_pool_bypasses"]++;return at::empty(source.sizes(),options);}
   found=std::make_shared<DeviceBuffer>();found->bytes=at::empty({int64_t(capacity)},options.dtype(at::kByte));
   device_bytes+=capacity;devices.push_back(found);b->stats["device_buffer_allocations"]++;
  }
  found->busy=true;found->stamp=++clock;lease.buffers.push_back(found);
  return at::from_blob(found->bytes.data_ptr(),source.sizes(),options);
 }
 void trim_devices(){for(auto it=devices.begin();it!=devices.end();){if(!(*it)->busy){device_bytes-=(*it)->bytes.numel();it=devices.erase(it);}else ++it;}}
 void dispose(std::unique_ptr<Lease> item){
  if(item->abandoned_prefetch)pending_prefetch_bytes-=item->bytes;
  if(item->pinned){auto& v=item->record->vbar;api().unpin((void*)v[0],(void*)v[1],uint64_t(v[2]-v[3]),uint64_t(v[4]));item->pinned=false;}
  item->tensors.clear();for(auto& p:item->buffers)p->busy=false;item->buffers.clear();
  recycle(item->ready,item->record->optimize_io);recycle(item->done,item->record->optimize_io);
 }
 void drain(bool wait=false){
  for(auto it=pending.begin();it!=pending.end();){auto status=cudaEventQuery((*it)->done);
   if(status==cudaSuccess){auto item=std::move(*it);it=pending.erase(it);dispose(std::move(item));}
   else if(status==cudaErrorNotReady&&!wait)++it;
   else {if(status!=cudaErrorNotReady)check(status);check(cudaEventSynchronize((*it)->done));auto item=std::move(*it);it=pending.erase(it);dispose(std::move(item));}
  }
 }
 void wait_oldest(){TORCH_CHECK(!pending.empty(),"Native K2 has no completed weight lease to reclaim");check(cudaEventSynchronize(pending.front()->done));auto item=std::move(pending.front());pending.pop_front();dispose(std::move(item));drain();}
 void release(std::unique_ptr<Lease> item,cudaStream_t stream){item->done=event(item->record->optimize_io);check(cudaEventRecord(item->done,stream));pending.push_back(std::move(item));}
 void defer(std::unique_ptr<Lease> item){item->abandoned_prefetch=true;pending_prefetch_bytes+=item->bytes;release(std::move(item),transfer.stream());}
 bool copy(Lease& lease,int slot,Tensor source,Tensor const& destination,bool speculative){
  auto b=lease.record;
  // VBAR geometry can use a different floating dtype than the CPU checkpoint.
  // Convert the payload before copying into its typed destination view.
  if(source.scalar_type()!=destination.scalar_type())source=source.to(destination.scalar_type());
  source=source.contiguous();HostBuffer* cached=nullptr;
  if(!source.is_cuda()&&!source.is_pinned()){
   if(b->optimize_io)cached=host(b,slot,source,!speculative);
   if(!cached){
    if(speculative&&b->optimize_io)return false;
    auto pinned=at::empty(source.sizes(),source.options().pinned_memory(true));pinned.copy_(source);source=pinned;lease.staging.push_back(pinned);
   }
  }
  const void* ptr=cached?cached->bytes.data_ptr():source.data_ptr();
  check(cudaMemcpyAsync(destination.data_ptr(),ptr,source.nbytes(),source.is_cuda()?cudaMemcpyDeviceToDevice:cudaMemcpyHostToDevice,transfer.stream()));
  if(cached){check(cudaEventRecord(cached->copied,transfer.stream()));cached->recorded=true;}
  if(!source.is_cuda())b->stats["h2d_bytes"]+=int64_t(source.nbytes());return true;
 }
};
std::unordered_map<int,std::unique_ptr<Runtime>>& runtimes(){static auto* r=new std::unordered_map<int,std::unique_ptr<Runtime>>;return *r;}
Runtime& runtime(int device){auto& p=runtimes()[device];if(!p)p=std::make_unique<Runtime>(device);return *p;}
cudaStream_t stream(Tensor const& x){return c10::cuda::getCurrentCUDAStream(x.get_device()).stream();}
std::unique_ptr<Lease> prepare(Runtime& rt,int64_t key,bool speculative=false){
 auto cached=rt.prefetched.find(key);
 if(cached!=rt.prefetched.end()){auto out=std::move(cached->second);rt.prefetched_bytes-=out->bytes;rt.prefetched.erase(cached);out->record->stats["prefetch_hits"]++;return out;}
 rt.drain();auto b=get(key);auto out=std::make_unique<Lease>();out->record=b;
 for(auto const& t:b->cpu)if(t.defined())out->bytes+=t.nbytes();
 out->ready=rt.event(b->optimize_io);
 if(b->vbar.empty()){
  out->tensors=b->gpu;
  if(out->tensors.empty()){
   c10::cuda::CUDAStreamGuard guard(rt.transfer);
   for(size_t i=0;i<b->cpu.size();++i){auto src=b->cpu[i];if(!src.defined()){out->tensors.emplace_back();continue;}
    auto dst=rt.destination(src,*out);out->tensors.push_back(dst);
    if(!rt.copy(*out,int(i),src,dst,speculative)){rt.defer(std::move(out));return nullptr;}
   }
  }
  check(cudaEventRecord(out->ready,rt.transfer.stream()));return out;
 }
 auto& v=b->vbar;std::vector<uint32_t> sig(size_t(v[4]/(32<<20)+2));
 int status=api().fault((void*)v[0],(void*)v[1],uint64_t(v[2]-v[3]),uint64_t(v[4]),sig.data());
 if(status==1&&speculative){
  size_t available=0,total=0;check(cudaMemGetInfo(&available,&total));
  if(available<out->bytes+(64ULL<<20)){b->stats["prefetch_deferred"]++;rt.recycle(out->ready,b->optimize_io);return nullptr;}
 }
 if(status==1&&b->optimize_io&&!speculative)rt.trim_devices();
 while(status==1&&!speculative&&!rt.pending.empty()){
  b->stats["memory_waits"]++;rt.wait_oldest();status=api().fault((void*)v[0],(void*)v[1],uint64_t(v[2]-v[3]),uint64_t(v[4]),sig.data());
 }
 if(status==1){
  try{
   c10::cuda::CUDAStreamGuard guard(rt.transfer);
   for(size_t i=0;i<b->cpu.size();++i){auto src=b->cpu[i];if(!src.defined()){out->tensors.emplace_back();continue;}
    auto dst=rt.destination(src,*out);out->tensors.push_back(dst);
    if(!rt.copy(*out,int(i),src,dst,speculative)){b->stats["prefetch_deferred"]++;rt.defer(std::move(out));return nullptr;}
   }
   b->stats["cast_buffer_fallbacks"]++;
   check(cudaEventRecord(out->ready,rt.transfer.stream()));return out;
  }catch(c10::OutOfMemoryError const&){
   if(speculative){b->stats["prefetch_deferred"]++;rt.defer(std::move(out));return nullptr;}
   cudaStreamSynchronize(rt.transfer.stream());throw;
  }catch(...){cudaStreamSynchronize(rt.transfer.stream());throw;}
 }
 TORCH_CHECK(status==0,"Native K2 VBAR preparation failed: ",status);
 out->pinned=true;out->tensors=b->gpu;
 if(sig!=b->signature){
  try{
   c10::cuda::CUDAStreamGuard guard(rt.transfer);
   for(size_t i=0;i<b->cpu.size();++i){auto src=b->cpu[i];if(!src.defined())continue;
    if(!rt.copy(*out,int(i),src,out->tensors[i],speculative)){
     b->stats["prefetch_deferred"]++;rt.defer(std::move(out));return nullptr;
    }
   }
   b->signature=sig;b->stats["weight_copies"]++;
  }catch(...){cudaStreamSynchronize(rt.transfer.stream());api().unpin((void*)v[0],(void*)v[1],uint64_t(v[2]-v[3]),uint64_t(v[4]));out->pinned=false;throw;}
 }
 check(cudaEventRecord(out->ready,rt.transfer.stream()));return out;
}
void prefetch(Runtime& rt,Keys const& keys){
 rt.drain();
 for(auto key:keys)if(!rt.prefetched.count(key)){
  auto b=get(key);size_t bytes=0;for(auto const& t:b->cpu)if(t.defined())bytes+=t.nbytes();
  if(b->optimize_io&&rt.prefetched_bytes+rt.pending_prefetch_bytes+bytes>rt.prefetch_limit){b->stats["prefetch_budget_skips"]++;continue;}
  auto item=prepare(rt,key,b->optimize_io);if(!item)continue;
  item->record->stats["prefetch_submissions"]++;rt.prefetched_bytes+=item->bytes;rt.prefetched.emplace(key,std::move(item));
 }
}
void wait(std::unique_ptr<Lease> const& w,cudaStream_t s){check(cudaStreamWaitEvent(s,w->ready,0));}
Tensor norm_impl(Tensor x,std::shared_ptr<Binding> const& b,Tensor scale={},Tensor shift={},int64_t cut=-1){
 x=x.contiguous();if(scale.defined()){scale=scale.contiguous();shift=shift.contiguous();}auto y=at::empty_like(x);auto weight=b->gpu.at(0);int d=int(x.size(-1));
 TORCH_CHECK(weight.scalar_type()==at::kFloat&&weight.numel()==d,"Native RMSNorm weight mismatch");
 api().norm(x.data_ptr(),weight.data_ptr(),scale.defined()?scale.data_ptr():nullptr,shift.defined()?shift.data_ptr():nullptr,y.data_ptr(),int(x.numel()/d),d,int(x.dim()==3?x.size(1):1),int(cut),int(x.size(0)),float(b->eps),dtype_code(x),stream(x));check(cudaGetLastError());return y;
}
struct Quantized {Tensor data,scale;};
Quantized quantize(Tensor const& x,std::shared_ptr<Binding> const& b){
 auto flat=x.reshape({-1,x.size(-1)}).contiguous();int64_t m=flat.size(0),k=flat.size(1);Quantized q;
 if(b->format==1){q={at::empty({m,k/2},x.options().dtype(at::kChar)),at::empty({m,1},x.options().dtype(at::kFloat))};api().quant4(flat.data_ptr(),q.data.data_ptr(),q.scale.data_ptr(),m,k,b->group,dtype_code(x),stream(x));}
 else if(b->format==2){q={at::empty({m,k},x.options().dtype(at::kChar)),at::empty({m,1},x.options().dtype(at::kFloat))};api().quant8(flat.data_ptr(),q.data.data_ptr(),q.scale.data_ptr(),m,k,b->group,dtype_code(x),stream(x));}
 else if(b->format==3||b->format==4){
  auto dtype=b->format==3?at::kFloat8_e4m3fn:at::kFloat8_e5m2;int code=b->format==3?5:6;
  Tensor scale=b->input_scale;
  if(!scale.defined()){
   // FP16 division can underflow before a later FP32 cast, especially E5M2.
   // Keep the existing BF16 route; form FP16's dynamic scale in FP32.
   auto magnitude=at::amax(at::abs(flat));
   if(x.scalar_type()==at::kHalf)magnitude=magnitude.to(at::kFloat);
   scale=magnitude.div(b->format==3?448.:57344.).to(at::kFloat).clamp_min((std::numeric_limits<float>::min)());
  }
  q={at::empty({m,k},x.options().dtype(dtype)),scale};api().fp8(flat.data_ptr(),q.data.data_ptr(),scale.data_ptr(),flat.numel(),dtype_code(x),code,true,stream(x));
 }
 check(cudaGetLastError());return q;
}
Tensor project(Tensor const& x,std::unique_ptr<Lease> const& w,Quantized const& q,std::shared_ptr<Binding> const& norm={},Tensor freqs={},Tensor gate={},Tensor residual={}){
 auto b=w->record;auto weight=w->tensors.at(0);Tensor bias=w->tensors.size()>2?w->tensors[2]:Tensor();
 if(b->format==0){bias=w->tensors.size()>1?w->tensors[1]:Tensor();return at::linear(x,weight.to(x.scalar_type()),bias.defined()?c10::optional<Tensor>(bias.to(x.scalar_type())):c10::nullopt);}
 auto ws=w->tensors.at(1);int64_t k=x.size(-1),m=x.numel()/k,n=weight.size(0);auto shape=x.sizes().vec();shape.back()=n;auto out=at::empty(shape,x.options());
 if(b->format==1){
  if(norm&&freqs.defined()){
   int code=api().qk(q.data.data_ptr(),weight.data_ptr(),q.scale.data_ptr(),ws.data_ptr(),norm->gpu[0].data_ptr(),freqs.data_ptr(),out.data_ptr(),int(m),int(n),int(k),x.scalar_type()==at::kHalf?1:0,float(norm->eps),stream(x));TORCH_CHECK(code==0,"Native Q/K kernel failed: ",code);b->stats["qk_gemm_norm_rope"]++;
  }else if(gate.defined()&&api().gated&&b->fuse&&x.size(0)==1&&gate.numel()==n&&!bias.defined()){
   auto g=gate.contiguous(),r=residual.contiguous();int code=api().gated(q.data.data_ptr(),weight.data_ptr(),q.scale.data_ptr(),ws.data_ptr(),g.data_ptr(),r.data_ptr(),out.data_ptr(),int(m),int(n),int(k),x.scalar_type()==at::kHalf?1:0,0,stream(x));TORCH_CHECK(code==0,"Native gated kernel failed: ",code);b->stats["gated_gemm"]++;
  }else{
   auto bf=bias.defined()?bias.to(at::kFloat).contiguous():Tensor();api().gemm4(q.data.data_ptr(),weight.data_ptr(),q.scale.data_ptr(),ws.data_ptr(),bf.defined()?bf.data_ptr():nullptr,out.data_ptr(),m,n,k,dtype_code(x),stream(x));
  }
 }else if(b->format==2){
  int64_t mp=(m+31)/32*32,kp=(k+15)/16*16,np=(n+15)/16*16;
  auto a=q.data,wt=weight;
  if(mp!=m||kp!=k)a=at::constant_pad_nd(a,{0,kp-k,0,mp-m},0);
  if(np!=n||kp!=k)wt=at::constant_pad_nd(wt,{0,kp-k,0,np-n},0);
  auto values=at::_int_mm(a,wt.t()).slice(0,0,m).slice(1,0,n).contiguous();auto bi=bias.defined()?bias.to(x.scalar_type()).contiguous():Tensor();
  api().dequant8(values.data_ptr(),q.scale.data_ptr(),ws.data_ptr(),bi.defined()?bi.data_ptr():nullptr,out.data_ptr(),m,n,ws.numel(),dtype_code(x),stream(x));
 }else if(b->format==3&&k%16==0&&n%16==0){out=at::_scaled_mm(q.data,weight.t(),q.scale,ws,bias.defined()?c10::optional<Tensor>(bias.to(x.scalar_type())):c10::nullopt,c10::nullopt,x.scalar_type(),false).reshape(shape);}
 else{
  auto a=at::empty(q.data.sizes(),x.options()),wfloat=at::empty(weight.sizes(),x.options());
  int code=b->format==3?5:6;
  api().fp8(q.data.data_ptr(),a.data_ptr(),q.scale.data_ptr(),a.numel(),code,dtype_code(x),false,stream(x));
  api().fp8(weight.data_ptr(),wfloat.data_ptr(),ws.data_ptr(),wfloat.numel(),code,dtype_code(x),false,stream(x));
  out=at::linear(a,wfloat,bias.defined()?c10::optional<Tensor>(bias.to(x.scalar_type())):c10::nullopt).reshape(shape);
 }
 check(cudaGetLastError());return out;
}
std::tuple<Tensor,Tensor> rope_impl(Tensor q,Tensor k,Tensor f){
 q=q.contiguous();k=k.contiguous();f=f.contiguous();auto qo=at::empty_like(q),ko=at::empty_like(k);
 TORCH_CHECK(f.scalar_type()==at::kFloat&&f.size(-3)*2==q.size(-1),"Unsupported native RoPE layout");
 api().rope(q.data_ptr(),k.data_ptr(),f.data_ptr(),qo.data_ptr(),ko.data_ptr(),int(q.size(0)),int(q.size(2)),int(q.size(1)),int(k.size(1)),int(q.size(-1)),int(f.size(0)),dtype_code(q),stream(q));check(cudaGetLastError());return {qo,ko};
}
Tensor element(Tensor a,Tensor b,Tensor r,int mode,int width){
 if(mode==2&&a.dim()>0&&a.numel()!=a.size(-1))return r+a*b;
 a=a.contiguous();if(b.defined())b=b.contiguous();if(r.defined())r=r.contiguous();auto y=at::empty_like(r.defined()?r:a);
 api().element(a.data_ptr(),b.defined()?b.data_ptr():nullptr,r.defined()?r.data_ptr():nullptr,y.data_ptr(),y.numel(),mode,width,dtype_code(a),stream(a));check(cudaGetLastError());return y;
}
Tensor linear(Tensor const& x,int64_t key,Tensor gate={},Tensor residual={}){
 std::lock_guard<std::recursive_mutex> lock(registry_mutex());c10::cuda::CUDAGuard device(x.device());auto& rt=runtime(x.get_device());auto b=get(key);Quantized q;
 if(b->optimize_io)q=quantize(x,b);
 auto w=prepare(rt,key);wait(w,stream(x));if(!b->optimize_io)q=quantize(x,b);auto out=project(x,w,q,{}, {},gate,residual);
 bool fused=gate.defined()&&api().gated&&w->record->format==1&&w->record->fuse&&x.size(0)==1&&gate.numel()==out.size(-1)&&(w->tensors.size()<3||!w->tensors[2].defined());
 auto next=w->record->lookahead;rt.release(std::move(w),stream(x));prefetch(rt,next);
 return gate.defined()&&!fused?(get(key)->fuse?element(gate,out,residual,2,int(gate.numel())):residual+gate*out):out;
}
Tensors group(Tensor const& x,Keys const& keys,std::vector<int64_t> const& widths,Keys const& norms={},Tensor freqs={},Quantized const& prepared={}){
 std::lock_guard<std::recursive_mutex> lock(registry_mutex());c10::cuda::CUDAGuard device(x.device());auto& rt=runtime(x.get_device());auto main=stream(x);Tensors outputs;std::unordered_map<int64_t,Quantized> cache;cudaEvent_t shared;check(cudaEventCreateWithFlags(&shared,cudaEventDisableTiming));
 bool qkfused=api().qk&&norms.size()==2&&get(keys[0])->fuse&&get(keys[1])->fuse&&freqs.defined()&&x.size(0)==1&&x.size(-1)%256==0&&get(norms[0])->gpu[0].numel()==128&&freqs.is_contiguous()&&freqs.scalar_type()==at::kFloat&&get(keys[0])->format==1&&get(keys[1])->format==1;
 if(qkfused)for(int i=0;i<2;++i)qkfused=qkfused&&(get(keys[i])->cpu.size()<3||!get(keys[i])->cpu[2].defined());
 bool concurrent=keys.size()>1&&(x.numel()/x.size(-1)<=1024||(keys.size()==4&&widths[1]*4<=widths[0]));
 for(size_t i=0;i<keys.size();++i){
  auto w=prepare(rt,keys[i]);auto b=w->record;int64_t cache_key=int64_t(b->format)*100000+b->group;
  if(b->input_scale.defined())cache_key^=int64_t(b->input_scale.data_ptr());
  if(!b->fuse)cache_key=int64_t(i)+1000000000;
  if(!cache.count(cache_key)){cache[cache_key]=prepared.data.defined()?prepared:quantize(x,b);check(cudaEventRecord(shared,main));}
  bool side=concurrent&&i>0&&(keys.size()==2||i+1<keys.size());
  auto compute=side?rt.side:c10::cuda::getCurrentCUDAStream(x.get_device());
  if(side)b->stats["side_stream_gemms"]++;
  {c10::cuda::CUDAStreamGuard guard(compute);check(cudaStreamWaitEvent(compute.stream(),shared,0));wait(w,compute.stream());
   auto norm=qkfused&&i<2?get(norms[i]):std::shared_ptr<Binding>();outputs.push_back(project(x,w,cache[cache_key],norm,freqs));rt.release(std::move(w),compute.stream());}
  if(i==0){Keys ahead(keys.begin()+1,keys.end());prefetch(rt,ahead);}
 }
 if(concurrent){cudaEvent_t joined;check(cudaEventCreateWithFlags(&joined,cudaEventDisableTiming));check(cudaEventRecord(joined,rt.side.stream()));check(cudaStreamWaitEvent(main,joined,0));check(cudaEventDestroy(joined));get(keys[0])->stats["parallel_groups"]++;}
 check(cudaEventDestroy(shared));
 if(norms.size()==2){int h=int(widths[0]/get(norms[0])->gpu[0].numel()),kh=int(widths[1]/get(norms[1])->gpu[0].numel());int d=int(get(norms[0])->gpu[0].numel());
  if(!qkfused){outputs[0]=norm_impl(outputs[0].reshape({x.size(0),x.size(1),h,d}),get(norms[0]));outputs[1]=norm_impl(outputs[1].reshape({x.size(0),x.size(1),kh,d}),get(norms[1]));}
  outputs[0]=outputs[0].reshape({x.size(0),x.size(1),h,d}).transpose(1,2);outputs[1]=outputs[1].reshape({x.size(0),x.size(1),kh,d}).transpose(1,2);
  if(!qkfused&&freqs.defined()){auto pair=rope_impl(outputs[0],outputs[1],freqs);outputs[0]=std::get<0>(pair).transpose(1,2).contiguous().transpose(1,2);outputs[1]=std::get<1>(pair).transpose(1,2).contiguous().transpose(1,2);}
 }
 return outputs;
}
std::tuple<Tensor,Tensor> residual_norm_fused(Tensor x,Tensor gate,Tensor residual,Tensor scale,Tensor shift,
                                       int64_t key,int64_t norm_key){
 std::lock_guard<std::recursive_mutex> lock(registry_mutex());c10::cuda::CUDAGuard device(x.device());
 TORCH_CHECK(api().joint_norm&&x.dim()==3&&x.size(0)==1&&
             (x.scalar_type()==at::kHalf||x.scalar_type()==at::kBFloat16),"Unsupported fused residual/norm input");
 auto& rt=runtime(x.get_device());auto current=stream(x);auto binding=get(key);Quantized q;
 if(binding->optimize_io)q=quantize(x,binding);
 auto w=prepare(rt,key);wait(w,current);auto norm=get(norm_key);
 int m=int(x.size(1)),k=int(x.size(-1)),n=int(w->tensors[0].size(0));
 TORCH_CHECK(w->record->format==1&&w->tensors.size()==2&&norm->gpu[0].numel()==n&&
             gate.numel()==n&&scale.numel()==n&&shift.numel()==n&&residual.numel()==int64_t(m)*n,
             "Fused residual/norm storage contract changed");
 if(!binding->optimize_io)q=quantize(x,binding);gate=gate.contiguous();residual=residual.contiguous();
 scale=scale.contiguous();shift=shift.contiguous();auto out=at::empty_like(residual),normalized=at::empty_like(residual);
 auto counts=at::empty({(m+127)/128},x.options().dtype(at::kInt));
 int code=api().joint_norm(q.data.data_ptr(),w->tensors[0].data_ptr(),q.scale.data_ptr(),w->tensors[1].data_ptr(),
                           gate.data_ptr(),residual.data_ptr(),norm->gpu[0].data_ptr(),scale.data_ptr(),shift.data_ptr(),
                           out.data_ptr(),normalized.data_ptr(),counts.data_ptr(),m,n,k,
                           x.scalar_type()==at::kHalf?1:0,float(norm->eps),current);
 TORCH_CHECK(code==0,"Fused residual/RMSNorm launch failed: ",code);w->record->stats["residual_norm"]++;
 auto next=w->record->lookahead;rt.release(std::move(w),current);prefetch(rt,next);
 return {out,normalized};
}
Tensor joint_swiglu_fused(Tensor const& x,Keys const& keys,int64_t width,Quantized const& prepared={}){
 std::lock_guard<std::recursive_mutex> lock(registry_mutex());c10::cuda::CUDAGuard device(x.device());
 TORCH_CHECK(api().joint_pair&&keys.size()==2&&x.dim()==3&&
             (x.scalar_type()==at::kHalf||x.scalar_type()==at::kBFloat16),"Unsupported joint SwiGLU call");
 auto& rt=runtime(x.get_device());auto current=stream(x);
 auto gate=prepare(rt,keys[0]),up=prepare(rt,keys[1]);wait(gate,current);wait(up,current);
 TORCH_CHECK(gate->record->format==1&&up->record->format==1&&gate->record->group==up->record->group&&
             gate->tensors.size()==2&&up->tensors.size()==2&&gate->tensors[0].size(0)==width&&
             up->tensors[0].size(0)==width,"Joint SwiGLU weight contract changed");
 auto q=prepared.data.defined()?prepared:quantize(x,gate->record);auto shape=x.sizes().vec();shape.back()=width;auto out=at::empty(shape,x.options());
 int code=api().joint_pair(q.data.data_ptr(),gate->tensors[0].data_ptr(),up->tensors[0].data_ptr(),
                           q.scale.data_ptr(),gate->tensors[1].data_ptr(),up->tensors[1].data_ptr(),
                           out.data_ptr(),int(x.numel()/x.size(-1)),int(width),int(x.size(-1)),
                           x.scalar_type()==at::kHalf?1:0,current);
 TORCH_CHECK(code==0,"Joint SwiGLU launch failed: ",code);gate->record->stats["joint_swiglu"]++;
 rt.release(std::move(gate),current);rt.release(std::move(up),current);return out;
}
std::tuple<Tensor,Tensor,Tensor,Tensor> joint_qkv_fused(Tensor const& x,Tensor const& freqs,Keys const& keys,
                                               Keys const& norms,int64_t heads,int64_t kvheads,Quantized const& prepared={}){
 std::lock_guard<std::recursive_mutex> lock(registry_mutex());c10::cuda::CUDAGuard device(x.device());
 TORCH_CHECK(api().joint_qkv&&keys.size()==4&&norms.size()==2&&x.dim()==3&&x.size(0)==1&&
             (x.scalar_type()==at::kHalf||x.scalar_type()==at::kBFloat16)&&freqs.is_contiguous()&&
             freqs.scalar_type()==at::kFloat&&freqs.numel()==x.size(1)*64*4,"Unsupported joint QKV call");
 auto& rt=runtime(x.get_device());auto current=stream(x);std::array<std::unique_ptr<Lease>,4> w;
 for(int i=0;i<4;++i){w[i]=prepare(rt,keys[i]);wait(w[i],current);
  TORCH_CHECK(w[i]->record->format==1&&w[i]->record->group==w[0]->record->group&&w[i]->tensors.size()==2,
              "Joint QKV weight contract changed");}
 int m=int(x.size(1)),k=int(x.size(-1)),qn=int(heads*128),kn=int(kvheads*128);
 TORCH_CHECK(w[0]->tensors[0].size(0)==qn&&w[1]->tensors[0].size(0)==kn&&
             w[2]->tensors[0].size(0)==kn&&w[3]->tensors[0].size(0)==qn,"Joint QKV head dimensions changed");
 auto q=prepared.data.defined()?prepared:quantize(x,w[0]->record);auto nq=get(norms[0]),nk=get(norms[1]);
 auto qo=at::empty({1,m,qn},x.options()),ko=at::empty({1,m,kn},x.options());
 auto vo=at::empty_like(ko),go=at::empty_like(qo);
 int code=api().joint_qkv(q.data.data_ptr(),w[0]->tensors[0].data_ptr(),w[1]->tensors[0].data_ptr(),
      w[2]->tensors[0].data_ptr(),w[3]->tensors[0].data_ptr(),q.scale.data_ptr(),
      w[0]->tensors[1].data_ptr(),w[1]->tensors[1].data_ptr(),w[2]->tensors[1].data_ptr(),w[3]->tensors[1].data_ptr(),
      nq->gpu[0].data_ptr(),nk->gpu[0].data_ptr(),freqs.data_ptr(),qo.data_ptr(),ko.data_ptr(),vo.data_ptr(),go.data_ptr(),
      m,qn,kn,k,x.scalar_type()==at::kHalf?1:0,float(nq->eps),float(nk->eps),current);
 TORCH_CHECK(code==0,"Joint QKV launch failed: ",code);w[0]->record->stats["joint_qkv"]++;
 for(auto& weight:w)rt.release(std::move(weight),current);
 return {qo.view({1,m,heads,128}).transpose(1,2),ko.view({1,m,kvheads,128}).transpose(1,2),vo,go};
}
int route(int op,int64_t key,int64_t rows){
 std::lock_guard<std::recursive_mutex> lock(registry_mutex());auto b=get(key);
 auto it=b->routes.find(int64_t(op)*1000000000000LL+rows);return it==b->routes.end()?0:it->second;
}
void set_route(int op,int64_t key,int64_t rows,int path){
 TORCH_CHECK(op>=0&&op<=3&&rows>0&&(path==0||path==1),"Invalid kernel route");
 std::lock_guard<std::recursive_mutex> lock(registry_mutex());get(key)->routes[int64_t(op)*1000000000000LL+rows]=path;
}
std::tuple<Tensor,Tensor> norm_quant(Tensor x,Tensor scale,Tensor shift,int64_t norm_key,int64_t target_key,int64_t cut){
 std::lock_guard<std::recursive_mutex> lock(registry_mutex());c10::cuda::CUDAGuard device(x.device());
 auto norm=get(norm_key),target=get(target_key);int64_t k=x.size(-1),m=x.numel()/k;
 TORCH_CHECK(target->format==1&&norm->gpu[0].numel()==k,"Normalized quantization requires compatible INT4 input");
 Quantized out;
 if(!api().norm_quant||route(3,norm_key,m)==0){
  out=quantize(norm_impl(x,norm,scale,shift,cut),target);norm->stats["norm_quant_composed"]++;
 }else{
  x=x.contiguous();scale=scale.contiguous();shift=shift.contiguous();
  out={at::empty({m,k/2},x.options().dtype(at::kChar)),at::empty({m,1},x.options().dtype(at::kFloat))};
  int code=api().norm_quant(x.data_ptr(),norm->gpu[0].data_ptr(),scale.data_ptr(),shift.data_ptr(),out.data.data_ptr(),out.scale.data_ptr(),
      int(m),int(k),int(x.size(1)),int(cut),int(x.size(0)),float(norm->eps),target->group,dtype_code(x),stream(x));
  TORCH_CHECK(code==0,"Fused normalized INT4 quantization failed: ",code);norm->stats["norm_quant_fused"]++;
 }
 prefetch(runtime(x.get_device()),norm->lookahead);return {out.data,out.scale};
}
void prepared_input(Tensor const& x,Tensor const& qa,Tensor const& qs,Keys const& keys){
 TORCH_CHECK(qa.scalar_type()==at::kChar&&qs.scalar_type()==at::kFloat&&qa.is_contiguous()&&qs.is_contiguous()&&
             qa.device()==x.device()&&qs.device()==x.device()&&qa.numel()*2==x.numel()&&qs.numel()==x.numel()/x.size(-1),
             "Invalid packed normalized activation");
 auto group=get(keys.at(0))->group;
 for(auto key:keys)TORCH_CHECK(get(key)->format==1&&get(key)->group==group,"Packed input cannot be shared across incompatible quantization layouts");
}
std::tuple<Tensor,Tensor,Tensor,Tensor> qkv_prepared(Tensor const& x,Tensor const& qa,Tensor const& qs,Tensor const& freqs,
                                                 Keys const& keys,Keys const& norms,int64_t heads,int64_t kvheads){
 std::lock_guard<std::recursive_mutex> lock(registry_mutex());c10::cuda::CUDAGuard device(x.device());prepared_input(x,qa,qs,keys);
 Quantized q{qa,qs};get(keys[0])->stats["prepared_qkv"]++;
 if(route(0,keys[0],x.numel()/x.size(-1))==1)return joint_qkv_fused(x,freqs,keys,norms,heads,kvheads,q);
 auto values=group(x,keys,{heads*128,kvheads*128,kvheads*128,heads*128},norms,freqs,q);
 return {values[0],values[1],values[2],at::sigmoid(values[3])};
}
Tensor swiglu_prepared(Tensor const& x,Tensor const& qa,Tensor const& qs,Keys const& keys,int64_t width){
 std::lock_guard<std::recursive_mutex> lock(registry_mutex());c10::cuda::CUDAGuard device(x.device());prepared_input(x,qa,qs,keys);
 Quantized q{qa,qs};get(keys[0])->stats["prepared_swiglu"]++;
 if(route(1,keys[0],x.numel()/x.size(-1))==1)return joint_swiglu_fused(x,keys,width,q);
 auto values=group(x,keys,{width,width},{},{},q);return element(values[0],values[1],{},0,0);
}
std::tuple<Tensor,Tensor,Tensor> gated_norm_quant(Tensor x,Tensor gate,Tensor residual,Tensor scale,Tensor shift,
                                               int64_t key,int64_t norm_key,int64_t target_key,int64_t cut){
 std::lock_guard<std::recursive_mutex> lock(registry_mutex());c10::cuda::CUDAGuard device(x.device());
 auto out=linear(x,key,gate,residual);auto packed=norm_quant(out,scale,shift,norm_key,target_key,cut);
 return {out,std::get<0>(packed),std::get<1>(packed)};
}
std::tuple<Tensor,Tensor,Tensor,Tensor> joint_qkv(Tensor const& x,Tensor const& freqs,Keys const& keys,
                                               Keys const& norms,int64_t heads,int64_t kvheads){
 std::lock_guard<std::recursive_mutex> lock(registry_mutex());c10::cuda::CUDAGuard device(x.device());
 if(route(0,keys.at(0),x.numel()/x.size(-1))==1)return joint_qkv_fused(x,freqs,keys,norms,heads,kvheads);
 auto values=group(x,keys,{heads*128,kvheads*128,kvheads*128,heads*128},norms,freqs);
 get(keys[0])->stats["routed_qkv_split"]++;
 return {values[0],values[1],values[2],at::sigmoid(values[3])};
}
Tensor joint_swiglu(Tensor const& x,Keys const& keys,int64_t width){
 std::lock_guard<std::recursive_mutex> lock(registry_mutex());c10::cuda::CUDAGuard device(x.device());
 if(route(1,keys.at(0),x.numel()/x.size(-1))==1)return joint_swiglu_fused(x,keys,width);
 auto values=group(x,keys,{width,width});get(keys[0])->stats["routed_swiglu_split"]++;
 return element(values[0],values[1],{},0,0);
}
std::tuple<Tensor,Tensor> residual_norm(Tensor x,Tensor gate,Tensor residual,Tensor scale,Tensor shift,
                                       int64_t key,int64_t norm_key){
 std::lock_guard<std::recursive_mutex> lock(registry_mutex());c10::cuda::CUDAGuard device(x.device());
 if(route(2,key,x.numel()/x.size(-1))==1)return residual_norm_fused(x,gate,residual,scale,shift,key,norm_key);
 auto out=linear(x,key,gate,residual);auto normalized=norm_impl(out,get(norm_key),scale,shift);
 get(key)->stats["routed_residual_norm_split"]++;return {out,normalized};
}
Tensor sdpa(Tensor const& q,Tensor const& k,Tensor const& v,bool enable_gqa){
 c10::cuda::CUDAGuard device(q.device());
 auto out=at::scaled_dot_product_attention(q,k,v,c10::nullopt,0.,false,c10::nullopt,enable_gqa);
 return out.transpose(1,2).contiguous().view({q.size(0),q.size(2),q.size(1)*q.size(3)});
}
Tensor managed(Tensor x,int64_t key,int64_t kind,int64_t width){
 auto b=get(key);
 if(kind==0)return linear(x,key);
 if(kind==1)return norm_impl(x,b);
 if(kind==3)return x+b->gpu[0].to(x.scalar_type()).unsqueeze(0);
 if(kind==4)return x+b->gpu[0].to(x.scalar_type());
 if(kind==5)return at::gelu(x,"tanh");
 Tensors pieces;
 for(size_t i=0;i<b->axes.size();++i){int64_t d=b->axes[i];auto pos=x.select(-1,int64_t(i));auto scale=at::linspace(0.,double(d-2)/d,d/2,x.options().dtype(at::kDouble));auto omega=at::reciprocal(at::pow(b->theta,scale));auto angle=pos.to(at::kFloat).unsqueeze(-1)*omega;auto cosine=angle.cos(),sine=angle.sin();auto shape=angle.sizes().vec();shape.insert(shape.end(),{2,2});pieces.push_back(at::stack({cosine,-sine,sine,cosine},-1).reshape(shape).to(at::kFloat));}
 return at::cat(pieces,-3).unsqueeze(1);
}
Tensor norm_mod(Tensor x,Tensor scale,Tensor shift,int64_t key,int64_t cut){auto b=get(key);auto y=norm_impl(x,b,scale,shift,cut);std::lock_guard<std::recursive_mutex> lock(registry_mutex());prefetch(runtime(x.get_device()),b->lookahead);return y;}
Tensor timestep(Tensor t,int64_t dim){t=t*1000.;auto freq=at::exp(-std::log(10000.)*at::arange(dim/2,t.options().dtype(at::kFloat))/double(dim/2));auto args=t.to(at::kFloat).unsqueeze(-1)*freq.unsqueeze(0);auto out=at::cat({args.cos(),args.sin()},-1);if(dim%2)out=at::cat({out,at::zeros_like(out.slice(-1,0,1))},-1);return t.is_floating_point()?out.to(t.scalar_type()):out;}
void configure(std::string kernels,std::string epilogue,std::string qk,std::string aimdo){auto h=library(kernels);auto& a=api();
#define LOAD(field,name) a.field=(decltype(a.field))symbol(h,name)
 LOAD(quant4,"k2_quant4");LOAD(quant8,"k2_quant8");LOAD(gemm4,"k2_gemm4");LOAD(dequant8,"k2_dequant8");LOAD(fp8,"k2_fp8");LOAD(norm,"k2_norm");LOAD(element,"k2_elementwise");LOAD(rope,"k2_rope");
#undef LOAD
 if(!epilogue.empty())a.gated=(API::Gated)symbol(library(epilogue),"k2_int4_gated_residual");
 if(!qk.empty())a.qk=(API::QK)symbol(library(qk),"k2_int4_qk_norm_rope");
 if(!aimdo.empty()){auto v=library(aimdo);a.fault=(API::Fault)symbol(v,"vbar_fault");a.unpin=(API::Unpin)symbol(v,"vbar_unpin");}
}
void configure_joint(std::string const& path){
 if(path.empty())return;auto h=library(path);api().joint_pair=(API::JointPair)symbol(h,"k2_int4_swiglu");
 api().joint_qkv=(API::JointQkv)symbol(h,"k2_int4_qkv");
 api().joint_norm=(API::JointNorm)symbol(h,"k2_int4_residual_norm");
}
void configure_norm_quant(std::string const& path){if(!path.empty())api().norm_quant=(API::NormQuant)symbol(library(path),"k2_norm_mod_quant4");}
void add_binding(int64_t key,int kind,int format,Tensors cpu,Tensors gpu,Keys vbar,int group,bool fuse,double eps,Keys axes,double theta,Tensor input_scale,bool optimize_io){auto b=std::make_shared<Binding>();b->key=key;b->kind=kind;b->format=format;b->cpu=std::move(cpu);b->gpu=std::move(gpu);b->vbar=std::move(vbar);b->group=group;b->fuse=fuse;b->optimize_io=optimize_io;b->eps=eps;b->axes=std::move(axes);b->theta=theta;b->input_scale=input_scale.numel()?input_scale:Tensor();std::lock_guard<std::recursive_mutex> lock(registry_mutex());records()[key]=b;}
using Paths=std::vector<std::vector<std::string>>;
PyObject* resolve_path(PyObject* args,PyObject* kwargs,std::vector<std::string> const& path){
 PyObject* value=nullptr;
 for(size_t i=0;i<path.size();++i){auto const& part=path[i];char kind=part[0];auto name=part.substr(1);
  if(i==0&&kind=='A')value=args;
  else if(i==0&&kind=='K')value=kwargs;
  else if(i==0&&kind=='a'){auto index=std::stoll(name);TORCH_CHECK(index<PyTuple_GET_SIZE(args),"K2_NATIVE_INPUT_SIGNATURE_CHANGED");value=PyTuple_GET_ITEM(args,index);}
  else if(i==0&&kind=='k')value=PyDict_GetItemString(kwargs,name.c_str());
  else if(kind=='i'){auto index=std::stoll(name);TORCH_CHECK(value&&(PyList_Check(value)||PyTuple_Check(value)),"K2_NATIVE_INPUT_SIGNATURE_CHANGED");TORCH_CHECK(index<PySequence_Size(value),"K2_NATIVE_INPUT_SIGNATURE_CHANGED");value=PyList_Check(value)?PyList_GET_ITEM(value,index):PyTuple_GET_ITEM(value,index);}
  else {TORCH_CHECK(value&&PyDict_Check(value),"K2_NATIVE_INPUT_SIGNATURE_CHANGED");value=PyDict_GetItemString(value,name.c_str());}
  TORCH_CHECK(value,"K2_NATIVE_INPUT_SIGNATURE_CHANGED");
 }
 return value;
}
struct NativeGraph {
 using Entry=void(*)(void**,void**);
 Entry entry;Tensors constants;Keys sources;Paths paths,guard_paths,container_paths;Keys lengths;
 struct Signature{Keys sizes,strides;at::ScalarType dtype;at::Device device;bool requires_grad;};
 std::vector<Signature> signatures;
 std::vector<pybind11::object> guards;int outputs;
 NativeGraph(std::string path,Tensors inputs,Keys mapping,Tensors examples,Paths locations,
             Paths check_paths,std::vector<pybind11::object> values,Paths containers,Keys sizes,int nout):
  entry((Entry)symbol(library(path),"k2_compiled_entry")),constants(std::move(inputs)),sources(std::move(mapping)),paths(std::move(locations)),guard_paths(std::move(check_paths)),container_paths(std::move(containers)),lengths(std::move(sizes)),guards(std::move(values)),outputs(nout){
  TORCH_CHECK(constants.size()==sources.size(),"Invalid native input map");
  for(auto const& tensor:examples)signatures.push_back({tensor.sizes().vec(),tensor.strides().vec(),tensor.scalar_type(),tensor.device(),tensor.requires_grad()});
  for(size_t i=0;i<constants.size();++i)if(sources[i]>=0)constants[i]=Tensor();
 }
 Tensor run(pybind11::args args,pybind11::kwargs kwargs){
  for(size_t i=0;i<container_paths.size();++i)TORCH_CHECK(PyObject_Length(resolve_path(args.ptr(),kwargs.ptr(),container_paths[i]))==lengths[i],"K2_NATIVE_INPUT_SIGNATURE_CHANGED");
  for(size_t i=0;i<guard_paths.size();++i){auto obj=resolve_path(args.ptr(),kwargs.ptr(),guard_paths[i]);auto expected=guards[i].ptr();bool eq=obj==expected;
   if(!eq&&Py_TYPE(obj)==Py_TYPE(expected)&&(PyLong_CheckExact(obj)||PyFloat_CheckExact(obj)||PyUnicode_CheckExact(obj)))eq=PyObject_RichCompareBool(obj,expected,Py_EQ)==1;
   TORCH_CHECK(eq,"K2_NATIVE_INPUT_SIGNATURE_CHANGED");}
  Tensors dynamic;dynamic.reserve(paths.size());
  for(size_t i=0;i<paths.size();++i){auto obj=resolve_path(args.ptr(),kwargs.ptr(),paths[i]);auto a=pybind11::cast<Tensor>(pybind11::handle(obj));auto& b=signatures[i];TORCH_CHECK(a.sizes().equals(b.sizes)&&a.strides().equals(b.strides)&&a.scalar_type()==b.dtype&&a.device()==b.device&&a.requires_grad()==b.requires_grad,"K2_NATIVE_INPUT_SIGNATURE_CHANGED");dynamic.push_back(a);}
  std::vector<void*> in(constants.size()),out(size_t(outputs),nullptr);
  for(size_t i=0;i<in.size();++i)in[i]=new Tensor(sources[i]<0?constants[i]:dynamic[size_t(sources[i])]);
  // The generated ABI steals all input Tensor handles and owns their deletion.
  // Its C++ entry releases the GIL; CUDA/ATen/custom operator dispatch is native.
  try{entry(in.data(),out.data());}catch(...){for(auto ptr:out)delete (Tensor*)ptr;throw;}
  TORCH_CHECK(!out.empty()&&out[0],"Native graph returned no model output");
  Tensor result=*(Tensor*)out[0];for(auto ptr:out)delete (Tensor*)ptr;return result;
 }
};
void finish_tuning(Keys keys){std::lock_guard<std::recursive_mutex> lock(registry_mutex());for(auto& kv:runtimes()){auto& rt=*kv.second;for(auto key:keys){auto it=rt.prefetched.find(key);if(it!=rt.prefetched.end()){check(cudaEventSynchronize(it->second->ready));rt.prefetched_bytes-=it->second->bytes;rt.dispose(std::move(it->second));rt.prefetched.erase(it);}}rt.drain(true);}}
void release_bindings(Keys keys){std::lock_guard<std::recursive_mutex> lock(registry_mutex());finish_tuning(keys);for(auto& entry:runtimes()){auto& rt=*entry.second;for(auto it=rt.hosts.begin();it!=rt.hosts.end();){if(std::find(keys.begin(),keys.end(),(*it)->owner)!=keys.end()){if((*it)->recorded)check(cudaEventSynchronize((*it)->copied));rt.host_bytes-=(*it)->bytes.numel();it=rt.hosts.erase(it);}else ++it;}rt.trim_devices();}for(auto key:keys)records().erase(key);}
}
TORCH_LIBRARY(k2_native,m){
 m.def("norm_quant(Tensor x, Tensor scale, Tensor shift, int norm_key, int target_key, int cut) -> (Tensor, Tensor)");
 m.def("gated_norm_quant(Tensor x, Tensor gate, Tensor residual, Tensor scale, Tensor shift, int key, int norm_key, int target_key, int cut) -> (Tensor, Tensor, Tensor)");
 m.def("qkv_prepared(Tensor x, Tensor qa, Tensor qs, Tensor freqs, int[] keys, int[] norms, int heads, int kvheads) -> (Tensor, Tensor, Tensor, Tensor)");
 m.def("swiglu_prepared(Tensor x, Tensor qa, Tensor qs, int[] keys, int width) -> Tensor");
 m.def("sdpa(Tensor q, Tensor k, Tensor v, bool enable_gqa) -> Tensor");
 m.def("managed(Tensor x, int key, int kind, int width) -> Tensor");
 m.def("gated(Tensor x, Tensor gate, Tensor residual, int key) -> Tensor");
 m.def("norm_mod(Tensor x, Tensor scale, Tensor shift, int key, int cut) -> Tensor");
 m.def("project_many(Tensor x, int[] keys, int[] widths) -> Tensor[]");
 m.def("attention_inputs(Tensor x, Tensor? freqs, int[] keys, int[] widths, int[] norms, int heads, int kvheads) -> (Tensor, Tensor, Tensor, Tensor)");
 m.def("residual_norm(Tensor x, Tensor gate, Tensor residual, Tensor scale, Tensor shift, int key, int norm_key) -> (Tensor, Tensor)");
 m.def("joint_swiglu(Tensor x, int[] keys, int width) -> Tensor");
 m.def("joint_qkv(Tensor x, Tensor freqs, int[] keys, int[] norms, int heads, int kvheads) -> (Tensor, Tensor, Tensor, Tensor)");
 m.def("activation(Tensor x, str kind) -> Tensor");m.def("activate_mul(Tensor a, Tensor b, str kind) -> Tensor");
 m.def("modulation(Tensor x, Tensor scale, Tensor shift) -> Tensor");m.def("residual(Tensor x, Tensor gate, Tensor value) -> Tensor");
 m.def("rope(Tensor q, Tensor k, Tensor freqs) -> (Tensor, Tensor)");m.def("timestep(Tensor t, int dim) -> Tensor");
}
TORCH_LIBRARY_IMPL(k2_native,CUDA,m){
 m.impl("residual_norm",residual_norm);
 m.impl("joint_swiglu",joint_swiglu);m.impl("joint_qkv",joint_qkv);
 m.impl("norm_quant",norm_quant);m.impl("gated_norm_quant",gated_norm_quant);
 m.impl("qkv_prepared",qkv_prepared);m.impl("swiglu_prepared",swiglu_prepared);
 m.impl("sdpa",sdpa);
 m.impl("managed",managed);m.impl("gated",[](Tensor x,Tensor g,Tensor r,int64_t key){return linear(x,key,g,r);});m.impl("norm_mod",norm_mod);
 m.impl("project_many",[](Tensor x,Keys keys,Keys widths){return group(x,keys,widths);});
 m.impl("attention_inputs",[](Tensor x,c10::optional<Tensor> f,Keys keys,Keys widths,Keys norms,int64_t,int64_t){auto o=group(x,keys,widths,norms,f.has_value()?*f:Tensor());return std::make_tuple(o[0],o[1],o[2],o[3]);});
 m.impl("activation",[](Tensor x,std::string kind){return element(x,{}, {},kind=="silu"?0:1,0);});
 m.impl("activate_mul",[](Tensor a,Tensor b,std::string kind){return element(a,b,{},kind=="silu"?0:1,0);});
 m.impl("modulation",[](Tensor x,Tensor s,Tensor h){return (1+s)*x+h;});m.impl("residual",[](Tensor x,Tensor g,Tensor v){return element(g,v,x,2,int(g.numel()));});
 m.impl("rope",rope_impl);m.impl("timestep",timestep);
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){
 pybind11::class_<NativeGraph,std::shared_ptr<NativeGraph>>(m,"NativeGraph")
  .def(pybind11::init<std::string,Tensors,Keys,Tensors,Paths,Paths,std::vector<pybind11::object>,Paths,Keys,int>())
  .def("run",&NativeGraph::run);
 m.def("configure_norm_quant",configure_norm_quant);
 m.def("configure_joint",configure_joint);
 m.def("set_route",set_route);m.def("finish_tuning",finish_tuning);
 m.def("reset_stats",[](Keys keys){std::lock_guard<std::recursive_mutex> lock(registry_mutex());for(auto key:keys)get(key)->stats.clear();});
 m.def("configure",configure);m.def("add_binding",add_binding);m.def("release_bindings",release_bindings);
 m.def("set_lookahead",[](int64_t key,Keys next){get(key)->lookahead=std::move(next);});
 m.def("stats",[](Keys keys){std::unordered_map<std::string,int64_t> values;for(auto key:keys)for(auto const& kv:get(key)->stats)values[kv.first]+=kv.second;return values;});
}
