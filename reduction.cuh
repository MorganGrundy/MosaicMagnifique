#ifndef CUDA_REDUCTION
#define CUDA_REDUCTION

#include <cuda.h>

//Performs sum reduction in a single warp
template <size_t blockSize>
__device__
void warpReduceAdd(volatile double *sdata, const size_t tid);

//Performs sum reduction
template <size_t blockSize>
__global__
void reduceAdd(double *g_idata, double *g_odata, const size_t N, const size_t noLibIm);

void reduceAddData(double *data, double *output, const size_t N, const size_t maxBlockSize,
				   const size_t noLibIm);
#endif
