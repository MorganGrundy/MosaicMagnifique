#include "reduction.cuh"

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>

#include "cudaphotomosaicdata.h"

//Performs sum reduction in a single warp
template <size_t blockSize>
__device__
void warpReduceAdd(volatile double *sdata, const size_t tid)
{
    if (blockSize >= 64)
        sdata[tid] += sdata[tid + 32];
    if (blockSize >= 32)
        sdata[tid] += sdata[tid + 16];
    if (blockSize >= 16)
        sdata[tid] += sdata[tid + 8];
    if (blockSize >= 8)
        sdata[tid] += sdata[tid + 4];
    if (blockSize >= 4)
        sdata[tid] += sdata[tid + 2];
    if (blockSize >= 2)
        sdata[tid] += sdata[tid + 1];

}

//Performs sum reduction
template <size_t blockSize>
__global__
void reduceAdd(double *g_idata, double *g_odata, const size_t N, const size_t noLibIm)
{
    extern __shared__ double sdata[];

    for (size_t libI = 0; libI < noLibIm; ++libI)
    {
        size_t offset = libI * N;
        //Each thread loads atleast one element from global to shared memory
        size_t tid = threadIdx.x;
        size_t i = blockIdx.x * blockSize * 2 + threadIdx.x;
        size_t gridSize = blockSize * 2 * gridDim.x;
        sdata[tid] = 0;

        while (i < N)
        {
            sdata[tid] += (i + blockSize < N) ?
                        g_idata[i + offset] + g_idata[i + blockSize + offset] : g_idata[i + offset];
            i += gridSize;
        }
        __syncthreads();

        //Do reduction in shared memory
        if (blockSize >= 2048)
        {
            if (tid < 1024)
                sdata[tid] += sdata[tid + 1024];
            __syncthreads();
        }
        if (blockSize >= 1024)
        {
            if (tid < 512)
                sdata[tid] += sdata[tid + 512];
            __syncthreads();
        }
        if (blockSize >= 512)
        {
            if (tid < 256)
                sdata[tid] += sdata[tid + 256];
            __syncthreads();
        }
        if (blockSize >= 256)
        {
            if (tid < 128)
                sdata[tid] += sdata[tid + 128];
            __syncthreads();
        }
        if (blockSize >= 128)
        {
            if (tid < 64)
                sdata[tid] += sdata[tid + 64];
            __syncthreads();
        }

        if (tid < 32)
            warpReduceAdd<blockSize>(sdata, tid);

        //Write result for this block to global memory
        if (tid == 0)
            g_odata[blockIdx.x + libI * gridDim.x] = sdata[0];
    }
}

void reduceAddData(double *data, double *output, const size_t N, const size_t maxBlockSize,
                   const size_t noLibIm)
{
    size_t reduceDataSize = N;

    //Number of blocks needed assuming max block size
    size_t numBlocks = ((reduceDataSize + maxBlockSize - 1) / maxBlockSize + 1) / 2;

    //Minimum number of threads per block
    size_t reduceBlockSize;

    //Stores number of threads to use per block (power of 2)
    size_t threads = maxBlockSize;

    do
    {
        //Calculate new number of blocks and threads
        numBlocks = ((reduceDataSize + maxBlockSize - 1) / maxBlockSize + 1) / 2;
        reduceBlockSize = (reduceDataSize + numBlocks - 1) / numBlocks;
        while (threads > reduceBlockSize * 2)
            threads >>= 1;

        //Reduce
        switch (threads)
        {
        case 2048:
            reduceAdd<2048><<<static_cast<unsigned int>(numBlocks),
                              static_cast<unsigned int>(threads),
                              static_cast<unsigned int>(threads * sizeof(double))
                           >>>(data, output, reduceDataSize, noLibIm);
            break;
        case 1024:
            reduceAdd<1024><<<static_cast<unsigned int>(numBlocks),
                              static_cast<unsigned int>(threads),
                              static_cast<unsigned int>(threads * sizeof(double))
                           >>>(data, output, reduceDataSize, noLibIm);
            break;
        case 512:
            reduceAdd<512><<<static_cast<unsigned int>(numBlocks),
                             static_cast<unsigned int>(threads),
                             static_cast<unsigned int>(threads * sizeof(double))
                          >>>(data, output, reduceDataSize, noLibIm);
            break;
        case 256:
            reduceAdd<256><<<static_cast<unsigned int>(numBlocks),
                             static_cast<unsigned int>(threads),
                             static_cast<unsigned int>(threads * sizeof(double))
                          >>>(data, output, reduceDataSize, noLibIm);
            break;
        case 128:
            reduceAdd<128><<<static_cast<unsigned int>(numBlocks),
                             static_cast<unsigned int>(threads),
                             static_cast<unsigned int>(threads * sizeof(double))
                          >>>(data, output, reduceDataSize, noLibIm);
            break;
        case 64:
            reduceAdd<64><<<static_cast<unsigned int>(numBlocks),
                            static_cast<unsigned int>(threads),
                            static_cast<unsigned int>(threads * sizeof(double))
                         >>>(data, output, reduceDataSize, noLibIm);
            break;
        case 32:
            reduceAdd<32><<<static_cast<unsigned int>(numBlocks),
                            static_cast<unsigned int>(threads),
                            static_cast<unsigned int>(threads * sizeof(double))
                         >>>(data, output, reduceDataSize, noLibIm);
            break;
        case 16:
            reduceAdd<16><<<static_cast<unsigned int>(numBlocks),
                            static_cast<unsigned int>(threads),
                            static_cast<unsigned int>(threads * sizeof(double))
                         >>>(data, output, reduceDataSize, noLibIm);
            break;
        case 8:
            reduceAdd<8><<<static_cast<unsigned int>(numBlocks),
                           static_cast<unsigned int>(threads),
                           static_cast<unsigned int>(threads * sizeof(double))
                        >>>(data, output, reduceDataSize, noLibIm);
            break;
        case 4:
            reduceAdd<4><<<static_cast<unsigned int>(numBlocks),
                           static_cast<unsigned int>(threads),
                           static_cast<unsigned int>(threads * sizeof(double))
                        >>>(data, output, reduceDataSize, noLibIm);
            break;
        case 2:
            reduceAdd<2><<<static_cast<unsigned int>(numBlocks),
                           static_cast<unsigned int>(threads),
                           static_cast<unsigned int>(threads * sizeof(double))
                        >>>(data, output, reduceDataSize, noLibIm);
            break;
        case 1:
            reduceAdd<1><<<static_cast<unsigned int>(numBlocks),
                           static_cast<unsigned int>(threads),
                           static_cast<unsigned int>(threads * sizeof(double))
                        >>>(data, output, reduceDataSize, noLibIm);
            break;
        }

        gpuErrchk(cudaPeekAtLastError());
        gpuErrchk(cudaDeviceSynchronize());
        //Copy results back to data
        gpuErrchk(cudaMemcpy(data, output, numBlocks * noLibIm * sizeof(double),
                             cudaMemcpyDeviceToDevice));

        //New data length is equal to number of blocks
        reduceDataSize = numBlocks;
    }
    while (numBlocks > 1); //Keep reducing until only 1 block was used
}
