#ifndef CUDA_REDUCTION
#define CUDA_REDUCTION

#include <cuda.h>
#include <cuda_runtime_api.h>

#include "cudaphotomosaicdata.h"

//Performs sum reduction in a single warp
template <size_t blockSize>
__device__
void warpReduceAdd(volatile double *sdata, const size_t tid);

//Performs sum reduction
template <size_t blockSize>
__global__
void reduceAdd(double *g_idata, double *g_odata, const size_t N, const size_t noLibIm);

void reduceAddData(CUDAPhotomosaicData &photomosaicData, cudaStream_t stream[8],
                   const size_t noOfStreams);
#endif
