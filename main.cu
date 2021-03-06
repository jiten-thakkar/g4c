#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <errno.h>
#include "g4c.h"
#include "mm.h"
#include "internal.hh"

static const char *G4C_SEMCPY = "GPU memory copy failed";
static const char *G4C_SEKERNEL = "GPU kernel launch or execution failed";
static const char *G4C_SENOTEXIST = "no such error";

// G4C configurations:
struct g4c_context {
    int nr_streams;
    cudaStream_t *streams;
    int *stream_uses;

    int hostmem_handle;
    size_t hostmem_sz;
    void *hostmem_start;

    int wcmem_handle;
    size_t wcmem_sz;
    void *wcmem_start;

    int devmem_handle;
    size_t devmem_sz;
    void *devmem_start;
} __cur_ctx;

extern "C" const char *g4c_strerror(int err) {
    switch(err) {
    case G4C_EMCPY:
	return G4C_SEMCPY;
    case G4C_EKERNEL:
	return G4C_SEKERNEL;
    default:
	return G4C_SENOTEXIST;
    }
}

#define csc(...) _cuda_safe_call(__VA_ARGS__, __FILE__, __LINE__)
static cudaError_t
_cuda_safe_call(cudaError_t e, const char *file, int line) {
    if (e!=cudaSuccess) {
	fprintf(stderr, "g4c Error: %s %d %s\n",
		file, line, cudaGetErrorString(e));
	cudaDeviceReset();
	abort();
    }
    return e;
}


extern "C" int
g4c_init(int nr_ss, size_t hm_sz, size_t wcm_sz, size_t dm_sz) {
    int i;

    // Enable memory map, and spin CPU thread when waiting for sync to
    // decrease latency.
    csc( cudaSetDeviceFlags(
	     cudaDeviceScheduleSpin|cudaDeviceMapHost ) );

    // Most algorithms we wrote don't use shared memory, but access
    // global memory frequently, so much cache less shared memory.
    csc( cudaDeviceSetCacheConfig(cudaFuncCachePreferL1) );

    // Set up stream management:
    __cur_ctx.nr_streams = nr_ss;
    __cur_ctx.streams = (cudaStream_t*)malloc(sizeof(cudaStream_t)
					      *(__cur_ctx.nr_streams+1));
    __cur_ctx.stream_uses = (int*)malloc(sizeof(int)
					 *(__cur_ctx.nr_streams+1));
    if (!__cur_ctx.streams || !__cur_ctx.stream_uses)
	return -ENOMEM;
	
    // Create streams
    for (i=1; i<__cur_ctx.nr_streams+1; i++) {
	csc( cudaStreamCreate(&__cur_ctx.streams[i]) );
	__cur_ctx.stream_uses[i] = 0;
    }

    // Set up MM:
    __cur_ctx.hostmem_sz = hm_sz;
    __cur_ctx.devmem_sz = dm_sz;
    __cur_ctx.wcmem_sz = wcm_sz;

    csc( cudaHostAlloc(&__cur_ctx.hostmem_start,
		       hm_sz, cudaHostAllocPortable) );
    csc( cudaMalloc(&__cur_ctx.devmem_start,
		    dm_sz) );
    csc( cudaHostAlloc(&__cur_ctx.wcmem_start,
		       wcm_sz,
		       cudaHostAllocPortable|
		       cudaHostAllocWriteCombined) );

    __cur_ctx.hostmem_handle =
	g4c_new_mm_handle(
	    __cur_ctx.hostmem_start,
	    hm_sz,
	    G4C_PAGE_SHIFT);
    __cur_ctx.devmem_handle =
	g4c_new_mm_handle(
	    __cur_ctx.devmem_start,
	    dm_sz,
	    G4C_PAGE_SHIFT);
    __cur_ctx.wcmem_handle =
	g4c_new_mm_handle(
	    __cur_ctx.wcmem_start,
	    wcm_sz,
	    G4C_PAGE_SHIFT);
    
    csc( cudaGetLastError() );

    return 0;
}

extern "C" void
g4c_exit(void) {
    int i;

    for (i=1; i<__cur_ctx.nr_streams+1; i++) {
	csc( cudaStreamDestroy(__cur_ctx.streams[i]) );
    }

    g4c_release_mm_handle(__cur_ctx.hostmem_handle);
    g4c_release_mm_handle(__cur_ctx.devmem_handle);
    g4c_release_mm_handle(__cur_ctx.wcmem_handle);

    csc( cudaFree(__cur_ctx.devmem_start) );
    csc( cudaFreeHost(__cur_ctx.hostmem_start) );
    csc( cudaFreeHost(__cur_ctx.wcmem_start) );

    csc( cudaDeviceReset() );
}

extern "C" void
g4c_abort(void) {
    csc( cudaDeviceReset() );
}

cudaStream_t
g4c_get_stream(int s) {
    return __cur_ctx.streams[s];
}	

extern "C" int
g4c_stream_done(int s) {
    cudaError_t e = cudaStreamQuery(__cur_ctx.streams[s]);
    if (e == cudaSuccess) {
	return 1;
    } else if (e != cudaErrorNotReady) {
	csc(e);
    }

    return 0;
}

extern "C" int
g4c_stream_sync(int s) {
    csc( cudaStreamSynchronize(__cur_ctx.streams[s]) );
    return 0;
}

extern "C" int
g4c_h2d_async(void *h, void *d, size_t sz, int s)
{
    csc( cudaMemcpyAsync(d, h, sz, cudaMemcpyHostToDevice,
			 __cur_ctx.streams[s]) );
    return 0;
}

extern "C" int
g4c_d2h_async(void *d, void *h, size_t sz, int s)
{
    csc( cudaMemcpyAsync(h, d, sz, cudaMemcpyDeviceToHost,
			 __cur_ctx.streams[s]) );
    return 0;
}

extern "C" int
g4c_dev_memset(void *d, int val, size_t sz, int s)
{
    csc( cudaMemsetAsync(d, val, sz, __cur_ctx.streams[s]) );
    return 0;
}

extern "C" int
g4c_alloc_stream()
{
    for (int i=1; i<__cur_ctx.nr_streams+1; i++) {
	if (__cur_ctx.stream_uses[i] == 0) {
	    __cur_ctx.stream_uses[i] = 1;
	    return i;
	}
    }

    return 0;
}

extern "C" void
g4c_free_stream(int s)
{
    __cur_ctx.stream_uses[s] = 0;
}



// Memory management functions.
extern "C" void *
g4c_alloc_page_lock_mem(size_t sz)
{
    return g4c_alloc_mem(__cur_ctx.hostmem_handle, sz);
}

extern "C" void
g4c_free_host_mem(void *p)
{
    if (g4c_ptr_within(__cur_ctx.hostmem_start,
		       __cur_ctx.hostmem_sz, p))
	g4c_free_mem(__cur_ctx.hostmem_handle, p);
    else
	g4c_free_mem(__cur_ctx.wcmem_handle, p);
}

extern "C" void *
g4c_alloc_dev_mem(size_t sz)
{
    return g4c_alloc_mem(__cur_ctx.devmem_handle, sz);
}

extern "C" void
g4c_free_dev_mem(void *p)
{
    g4c_free_mem(__cur_ctx.devmem_handle, p);
}

extern "C" void *
g4c_alloc_wc_mem(size_t sz)
{
    return g4c_alloc_mem(__cur_ctx.wcmem_handle, sz);
}

// End of file.

