#include <cuda.h>
#include "g4c_ac.h"
#include "internal.hh"
#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#define __mytid (blockDim.x * blockIdx.x + threadIdx.x)
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}
__global__ void
gpu_ac_match_general(char *strs, int stride, int *lens, unsigned int *ress,
		     ac_dev_machine_t *acm)
{
    int id = blockDim.x * blockIdx.x + threadIdx.x;
    char *mystr = strs + (id*stride);
    unsigned int *res = ress + id*AC_ALPHABET_SIZE;

    char c;
    ac_state_t *st = acm_state(acm, 0);
    __syncthreads();
    for (int i=0; i<stride; i++) {
	c = mystr[i];
	if (c>=0) {
	    int nid = acm_state_transitions(acm, st->id)[c];
	    st = acm_state(acm, nid);			
	    if (st->noutput > 0) {
		for (int j=0; j<st->noutput; j++) {
		    int ot = acm_state_output(
			acm,
			st->output)[j];
		    res[ot]++;
		}
	    }
	}
    }
}

typedef union __align__(8) {
    uint64_t u64;
    uint32_t u32[2];
    uint8_t  u8[8];
} u64b_t;

__global__ void
gpu_acm_t1(char *strs, int stride, int *lens, unsigned int *ress,
	   ac_dev_machine_t *acm)
{
    int id = __mytid;
    int len = lens[id];

    unsigned int *res = ress + id*AC_ALPHABET_SIZE;

    uint64_t *p = (uint64_t*)(strs+id*stride);
    u64b_t d;
    int nid;
    ac_state_t *st = acm_state(acm, 0);
    __syncthreads();
    
    for (int i=0; i<len; i+= 8, ++p) {
	d.u64 = *p;	
	for (int j=0; j<8; j++) {
	    nid = acm_state_transitions(acm, st->id)[d.u8[j]];
	    st = acm_state(acm, nid);
	    if (st->noutput > 0) {
		for (int k=0; k<st->noutput; k++)
		    ++res[acm_state_output(acm, st->output)[k]];
	    }
	}
    }    
}

__global__ void
gpu_acm_t2(char *strs, int stride, int *lens, unsigned int *ress,
	   ac_dev_machine_t *acm)
{
    int id = __mytid;
    int len = lens[id];

    uint64_t *p = (uint64_t*)(strs+id*stride);
    u64b_t d;
    int nid;
    ac_state_t *st = acm_state(acm, 0);
    __syncthreads();
    
    for (int i=0; i<len; i+= 8, ++p) {
	d.u64 = *p;	
	for (int j=0; j<8; j++) {
	    nid = acm_state_transitions(acm, st->id)[d.u8[j]];
	    st = acm_state(acm, nid);
	    ress[id] += st->noutput;
	}
    }    
}

__global__ void
gpu_acm_t3(char *strs, int stride, int *lens, unsigned int *ress,
	   ac_dev_machine_t *acm)
{
    int id = __mytid;
    int len = lens[id];

    uint64_t *p = (uint64_t*)(strs+id*stride);
    u64b_t d;
    int nid;
    ac_state_t *st = acm_state(acm, 0);
    __syncthreads();
    
    for (int i=0; i<len; i+= 8, ++p) {
	d.u64 = *p;	
	for (int j=0; j<8; j++) {
	    nid = acm_state_transitions(acm, st->id)[d.u8[j]];
	    st = acm_state(acm, nid);
	    ress[id] = st->noutput;
	    return;
	}
    }    
}

__global__ void
gpu_acm_t4(char *strs, int stride, int *lens, unsigned int *ress,
	   ac_dev_machine_t *acm)
{
    int id = __mytid;
    int len = lens[id];

    uint64_t *p = (uint64_t*)(strs+id*stride);
    u64b_t d;
    unsigned int r=0;
    int nid;
    ac_state_t *st = acm_state(acm, 0);
    __syncthreads();
    
    for (int i=0; i<len; i+= 8, ++p) {
	d.u64 = *p;	
	for (int j=0; j<8; j++) {
	    nid = acm_state_transitions(acm, st->id)[d.u8[j]];
	    st = acm_state(acm, nid);
	    r += st->noutput;
	}
    }
   __syncthreads();
    ress[id] = r;
}


extern "C" size_t
ac_dev_acm_size(ac_machine_t *hacm)
{
    return g4c_ptr_offset(hacm->patterns, hacm->states);
}

extern "C" void
ac_free_dev_acm(ac_dev_machine_t **pdacm)
{
    ac_dev_machine_t *dacm = *pdacm;
    if (dacm) {
	if (dacm->dev_self)
	    g4c_free_dev_mem(dacm->dev_self);
	if (dacm->mem)
	    g4c_free_dev_mem(dacm->mem);
	g4c_free_host_mem(dacm);
	*pdacm = 0;
    }	
}

extern "C" int
ac_prepare_gmatch(ac_machine_t *hacm, ac_dev_machine_t **pdacm, int s)
{
    ac_dev_machine_t *dacm = *pdacm;
    if (!dacm) {
	*pdacm = (ac_dev_machine_t*)
	    g4c_alloc_page_lock_mem(sizeof(ac_dev_machine_t));
	if (!*pdacm) {
	    return -ENOMEM;
	}
	dacm = *pdacm;
	memset(dacm, 0, sizeof(ac_dev_machine_t));		
    }
	
    if (!dacm->dev_self) {
	dacm->dev_self = (ac_dev_machine_t*)
	    g4c_alloc_dev_mem(sizeof(ac_dev_machine_t));
	if (!dacm->dev_self) {
	    return -ENOMEM;
	}
    }

    if (!dacm->mem) {
	dacm->memsz = ac_dev_acm_size(hacm);
	dacm->mem = g4c_alloc_dev_mem(dacm->memsz);
	if (!dacm->mem)
	    return -ENOMEM;
    }

    dacm->memflags = hacm->memflags;
    dacm->nstates = hacm->nstates;
    dacm->noutputs = hacm->noutputs;

    dacm->states = (ac_state_t*)dacm->mem;
    dacm->transitions = (int*)g4c_ptr_add(
	dacm->states,
	g4c_ptr_offset(hacm->transitions,
		       hacm->states));
    dacm->outputs = (int*)g4c_ptr_add(
	dacm->states,
	g4c_ptr_offset(hacm->outputs,
		       hacm->states));

    dacm->hacm = hacm;
	
    int rt = g4c_h2d_async(
	dacm, dacm->dev_self, sizeof(ac_dev_machine_t), s);
    rt |= g4c_h2d_async(hacm->mem, dacm->mem, dacm->memsz, s);
		
    return rt;
}

extern "C" int
ac_gmatch(char *dstrs, int nstrs, int stride, int *dlens,
	  unsigned int *dress, ac_dev_machine_t *dacm, int s)
{
    cudaStream_t st = g4c_get_stream(s);
    int nblocks = g4c_round_up(nstrs, 32)/32;

    gpu_ac_match_general<<<nblocks, 32, 0, st>>>(
	dstrs, stride, dlens, dress, dacm);
    return 0;
}

extern "C" int
ac_gmatch2(char *dstrs, int nstrs, int stride, int *dlens,
	   unsigned int *dress, ac_dev_machine_t *dacm, int s,
	   unsigned int mtype)
{
    cudaStream_t st = g4c_get_stream(s);
    int nblocks = g4c_round_up(nstrs, 32)/32;

    switch(mtype) {
    case 1:
	gpu_acm_t1<<<nblocks, 32, 0, st>>>(dstrs, stride, dlens, dress, dacm);
	break;
    case 2:
	gpu_acm_t2<<<nblocks, 32, 0, st>>>(dstrs, stride, dlens, dress, dacm);
	break;
    case 3:
	gpu_acm_t3<<<nblocks, 32, 0, st>>>(dstrs, stride, dlens, dress, dacm);
	break;
    case 4:
	gpu_acm_t4<<<nblocks, 32, 0, st>>>(dstrs, stride, dlens, dress, dacm);
	break;
    case 0:
    default:
	gpu_ac_match_general<<<nblocks, 32, 0, st>>>(dstrs, stride, dlens, dress, dacm);
	break;
    }
    return 0;
}

/*
 * May not need this.
 */
extern "C" int
ac_gmatch_finish(int nstrs, unsigned int *dress, unsigned int *hress,
		 int s)
{
    return g4c_d2h_async(dress, hress,
			 nstrs*AC_ALPHABET_SIZE*sizeof(unsigned int),
			 s);
}


extern "C" int
ac_gmatch2_ofs(char *dstrs, int n, int stride, int *dlens, int *dress,
	       ac_dev_machine_t *dacm, int s, unsigned int mtype,
	       uint32_t pkt_ofs, uint32_t res_stride, uint32_t res_ofs)
{
    return 0;
}


__global__ void
gacm_match_l0(g4c_kmp_t *dacm,
	      uint8_t *data, uint32_t data_stride, uint32_t data_ofs,
	      int *lens,
	      int *ress, uint32_t res_stride, uint32_t res_ofs)
{
//    int tid = threadIdx.x + blockIdx.x*blockDim.x;
//
//    uint8_t *payload = data + data_stride*tid + data_ofs;
//    int mylen = lens[tid]-data_ofs;
//
//    int outidx = 0x1fffffff;
//    int nid, cid = 0, tres;
//    for (int i=0; i<mylen; i++) {
//	nid = g4c_acm_dtransitions(dacm, cid)[payload[i]];
//        tres = *g4c_acm_doutput(dacm, cid);
//	if (tres && tres < outidx) {
//	    outidx = tres;
//	}
//	cid = nid;
//    }
//    if (outidx == 0x1fffffff)
//	outidx = 0;
//    *(ress + tid*res_stride+res_ofs) = outidx;
}

__global__ void
gacm_match_nl0(g4c_kmp_t *dacm,
	      uint8_t *data, uint32_t data_stride, uint32_t data_ofs,
	      int *ress, uint32_t res_stride, uint32_t res_ofs, int maxPatternLen, int totalPatterns)
{
    //const unsigned long long int blockId = blockIdx.x //1D
      //                                     + blockIdx.y * gridDim.x //2D
        //                                   + gridDim.x * gridDim.y * blockIdx.z; //3D

// global unique thread index, block dimension uses only x-coordinate
//    const unsigned long long int tid = blockId * blockDim.x + threadIdx.x;
    //printf("in kernel\n");
    //printf("blockdimex: %d, blockdimy: %d blockdimz: %d \n", blockDim.x, blockDim.y, blockDim.z);
    //printf("blockx: %d, blcky: %d\n", blockIdx.x, blockIdx.y);
    //int tid = threadIdx.y + threadIdx.x * blockDim.y;
    //printf("threadId: %d\n", tid);
    //int tid0 = threadIdx.x + blockIdx.x*blockDim.x;
    //printf("threadId2: %d\n", tid0);
//	int zdim = threadIdx.z;
    //printf("threadidx: %d \n", threadIdx.x);
    //int tid = threadIdx.x+(blockIdx.y*blockDim.x);
    int tid = blockIdx.x;
    //int tid = 0;
    int patternId = threadIdx.x;
    //int patternId = 0;
    //if(patternId == 15)
     //printf("tid: %d\n", patternId);
    //printf("patternid: %d\n", patternId);
    //printf("patternid: %d\n", patternId);
    //__syncthreads();
    extern __shared__ uint8_t sm[];
    uint8_t *payload_temp = data + data_stride*tid + data_ofs;
    //char* pattern_temp = g4c_kmp_dpatterns(dacm, patternId);
    //char* pattern = (char*)&sm[(data_stride-data_ofs)];
    //int *lps_temp = g4c_kmp_dlpss(dacm, patternId);
    //int *lps = (int*)&pattern[totalPatterns*maxPatternLen];
    //char payloadt[] = "ABABDABACDABABCABAB";
    //char* payload = payloadt + threadIdx.y%19;
    uint8_t *payload = sm;
    if(threadIdx.x==0 && threadIdx.y==0) {
    for (int i=0; i<(data_stride-data_ofs); i++) {
      payload[i] = payload_temp[i];
    }
    /*for(int j=0; j<totalPatterns*maxPatternLen; j++) {
      //int patternLengthTemp = dacm->dPatternLengths[j];
      pattern[j] = pattern_temp[j]; 
    }
    for(int j=0; j<totalPatterns*maxPatternLen; j++) {
      //int patternLengthTemp = dacm->dPatternLengths[j];
      lps[j] = lps_temp[j];
    }*/
    }
__syncthreads();
    //memcpy(payload, payload_temp, data_stride);
   //printf("in kernel0\n");
    int outidx = 0x1fffffff;
    //printf("in kernel1\n");
//    int nid, cid = 0, tres;
//    for (int i=0; i<(data_stride-data_ofs); i++) {
//	nid = g4c_acm_dtransitions(dacm, cid)[payload[i]];
//        tres = *g4c_acm_doutput(dacm, cid);
//	if (tres && tres < outidx) {
//	    outidx = tres;
//	}
//	cid = nid;
//    }
    //printf("reading lps\n");
    //printf("read lps, reading pattern\n");
    int patternLength = dacm->dPatternLengths[patternId];
    //int patternLength = 9;
    char* pattern = g4c_kmp_dpatterns(dacm, patternId);
    //char* pattern = (char*)&sm;
    //char patternt[] = "ABABCABAB";
    //char* pattern = patternt;
    int *lps = g4c_kmp_dlpss(dacm, patternId);
    //int *lps = (int*)&pattern[(maxPatternLen)];
    //int lpst[] = {0,0,1,2,0,1,2,3,4};
    //int *lps = lpst;
    /*if (threadIdx.x == 0 && threadIdx.y == 0) {
    for(int i=0; i<patternLength; i++){
      pattern[i] = pattern_temp[i];
    }


    for(int i=0; i<patternLength; i++) {
      lps[i] = lps_temp[i];
    }
    }*/
    //__syncthreads();
    //pattern[patternLength] = '\0';
    //printf("pid: %d plen: %d pattern: %s\n", patternId, patternLength, pattern);
   //__syncthreads();
    //printf("read lps and pattern\n");
    int i = 0, j = 0;// index for txt[]
    while (i < data_stride) {
        //printf("in while, i: %d\n", i);
//        if (pattern[j] == payload[i]) {
//            j++;
//            i++;
//        }
        int tmp = pattern[j] == payload[i];
        j += tmp;
        i += tmp;
        if (j == patternLength) {
//            printf("Found pattern at index %d \n", i-j);
            outidx = i-j;
            break;
//            j = lps[j-1];
        } else if (i < (data_stride-data_ofs) && pattern[j] != payload[i]) { // mismatch after j matches
            // Do not match lps[0..lps[j-1]] characters,
            // they will match anyway
            if (j != 0) {
                //printf("before j\n");
                j = lps[j-1]; 
                //printf("after j\n");    
            }
            else
                i = i+1;
        }
    }

    if (outidx == 0x1fffffff)
	outidx = 0;
    *(ress + tid*res_stride + patternId + res_ofs) = outidx;
    //*/
}

__global__ void
gacm_match_l1(g4c_kmp_t *dacm,
	      uint8_t *data, uint32_t data_stride, uint32_t data_ofs,
	      int *lens,
	      int *ress, uint32_t res_stride, uint32_t res_ofs)
{
//    int tid = threadIdx.x + blockIdx.x*blockDim.x;
//
//    uint8_t *payload = data + data_stride*tid + data_ofs;
//    int mylen = lens[tid]-data_ofs;
//
//    int nid, cid = 0, tres;
//    for (int i=0; i<mylen; i++) {
//	nid = g4c_acm_dtransitions(dacm, cid)[payload[i]];
//        tres = *g4c_acm_doutput(dacm, cid);
//	if (tres) {
//	    *(ress + tid*res_stride+res_ofs) = tres;
//	    return;
//	}
//	cid = nid;
//    }
//    *(ress + tid*res_stride+res_ofs) = 0;
}

__global__ void
gacm_match_nl1(g4c_kmp_t *dacm,
	      uint8_t *data, uint32_t data_stride, uint32_t data_ofs,
	      int *ress, uint32_t res_stride, uint32_t res_ofs)
{
//    int tid = threadIdx.x + blockIdx.x*blockDim.x;
//
//    uint8_t *payload = data + data_stride*tid + data_ofs;
//
//    int nid, cid = 0, tres;
//    for (int i=0; i<(data_stride-data_ofs); i++) {
//	nid = g4c_acm_dtransitions(dacm, cid)[payload[i]];
//        tres = *g4c_acm_doutput(dacm, cid);
//	if (tres) {
//	    *(ress + tid*res_stride + res_ofs) = tres;
//	}
//	cid = nid;
//    }
//    *(ress + tid*res_stride + res_ofs) = 0;
}

extern "C" int
g4c_gpu_acm_match(
    g4c_kmp_t *dacm, int nr,
    uint8_t *ddata, uint32_t data_stride, uint32_t data_ofs,
    int *dlens,
    int *dress, uint32_t res_stride, uint32_t res_ofs,
    int s, int mtype)
{
//    if (s <= 0)
//	return -1;
    int threadsPerBlock = 32;
    cudaStream_t stream = g4c_get_stream(s);
    int nblocks = g4c_round_up(nr, threadsPerBlock)/threadsPerBlock;
    dim3 dimGrid(nr);
    int nthreads = nr > threadsPerBlock? threadsPerBlock:nr;
    dim3 dimBlock(TOTAL_PATTERNS);
    if (dlens) {
	switch(mtype) {
	case 1:
	    gacm_match_l1<<<nblocks, nthreads, 0, stream>>>(
		dacm, ddata, data_stride, data_ofs, dlens,
		dress, res_stride, res_ofs);
	    break;
	case 0:
	default:
	    gacm_match_l0<<<nblocks, nthreads, 0, stream>>>(
		dacm, ddata, data_stride, data_ofs, dlens,
		dress, res_stride, res_ofs);
	    break;
	}  	    
    } else {
        switch(mtype) {
        case 1:
            gacm_match_nl1<<<nblocks, nthreads, 0, stream>>>(
            dacm, ddata, data_stride, data_ofs,
            dress, res_stride, res_ofs);
            break;
        case 0:
        default:
            //printf("calling kernel\n");
            //printf("blocks %d, threads: %d\n", nblocks, nthreads);
            gacm_match_nl0<<<dimGrid, dimBlock, (sizeof(uint8_t)*(data_stride-data_ofs)), stream>>>(
            //gacm_match_nl0<<<dimGrid, dimBlock, (sizeof(uint8_t)*(data_stride-data_ofs))+(sizeof(int)*PATTERN_LENGTH*TOTAL_PATTERNS), stream>>>(
            dacm, ddata, data_stride, data_ofs,
            dress, res_stride, res_ofs, PATTERN_LENGTH, TOTAL_PATTERNS);
            //printf("kernel done\n");
            gpuErrchk( cudaPeekAtLastError() );
            //gpuErrchk( cudaDeviceSynchronize() );
            break;
        }
    }
    
    return 0;
}
