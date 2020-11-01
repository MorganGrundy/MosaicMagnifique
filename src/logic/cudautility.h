#ifndef CUDAUTILITY_H
#define CUDAUTILITY_H

#include <cuda_runtime_api.h>
#include <cstdio>
#include <cstdlib>

#define gpuErrchk(ans) { CUDAUtility::gpuAssert((ans), __FILE__, __LINE__); }

namespace CUDAUtility
{

inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort)
            exit(code);
    }
}

}

#endif // CUDAUTILITY_H
