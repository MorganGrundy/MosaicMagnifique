/*
	Copyright © 2018-2020, Morgan Grundy

	This file is part of Mosaic Magnifique.

    Mosaic Magnifique is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    Mosaic Magnifique is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with Mosaic Magnifique.  If not, see <https://www.gnu.org/licenses/>.
*/

#include "reduction.cuh"
#include "cudautility.h"

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
void reduceAdd(double *g_idata, double *g_odata, const size_t N)
{
    extern __shared__ double sdata[];

    //Each thread loads atleast one element from global to shared memory
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockSize * 2 + threadIdx.x;
    unsigned int gridSize = blockSize * 2 * gridDim.x;
    sdata[tid] = 0;

    while (i < N)
    {
        sdata[tid] += (i + blockSize < N) ? g_idata[i] + g_idata[i + blockSize] : g_idata[i];
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
        g_odata[blockIdx.x] = sdata[0];
}

//Wrapper for reduce add kernel
void reduceAddKernelWrapper(size_t blockSize, size_t size, double *variants, double *reductionMem)
{
    //Size of data to be reduced
    size_t reduceDataSize = size;

    //Number of blocks needed assuming max block size
    size_t numBlocks = (((reduceDataSize + 1) / 2 + blockSize - 1) / blockSize + 1) / 2;

    //Minimum number of threads per block
    size_t reduceBlockSize;

    //Stores number of threads to use per block (power of 2)
    size_t threads = blockSize;

    do
    {
        //Calculate new number of blocks and threads
        numBlocks = (((reduceDataSize + 1) / 2 + blockSize - 1) / blockSize + 1) / 2;
        reduceBlockSize = ((reduceDataSize + 1) / 2 + numBlocks - 1) / numBlocks;
        while ((threads >> 1) >= reduceBlockSize)
            threads >>= 1;

        //Reduce
        switch (threads)
        {
        case 2048:
            reduceAdd<2048><<<static_cast<unsigned int>(numBlocks),
                              static_cast<unsigned int>(threads),
                              static_cast<unsigned int>(threads * sizeof(double))>>>(
                variants, reductionMem, reduceDataSize);
            break;
        case 1024:
            reduceAdd<1024><<<static_cast<unsigned int>(numBlocks),
                              static_cast<unsigned int>(threads),
                              static_cast<unsigned int>(threads * sizeof(double))>>>(
                variants, reductionMem, reduceDataSize);
            break;
        case 512:
            reduceAdd<512><<<static_cast<unsigned int>(numBlocks),
                             static_cast<unsigned int>(threads),
                             static_cast<unsigned int>(threads * sizeof(double))>>>(
                variants, reductionMem, reduceDataSize);
            break;
        case 256:
            reduceAdd<256><<<static_cast<unsigned int>(numBlocks),
                             static_cast<unsigned int>(threads),
                             static_cast<unsigned int>(threads * sizeof(double))>>>(
                variants, reductionMem, reduceDataSize);
            break;
        case 128:
            reduceAdd<128><<<static_cast<unsigned int>(numBlocks),
                             static_cast<unsigned int>(threads),
                             static_cast<unsigned int>(threads * sizeof(double))>>>(
                variants, reductionMem, reduceDataSize);
            break;
        case 64:
            reduceAdd<64><<<static_cast<unsigned int>(numBlocks),
                            static_cast<unsigned int>(threads),
                            static_cast<unsigned int>(threads * sizeof(double))>>>(
                variants, reductionMem, reduceDataSize);
            break;
        case 32:
            reduceAdd<32><<<static_cast<unsigned int>(numBlocks),
                            static_cast<unsigned int>(threads),
                            static_cast<unsigned int>(threads * sizeof(double))>>>(
                variants, reductionMem, reduceDataSize);
            break;
        case 16:
            reduceAdd<16><<<static_cast<unsigned int>(numBlocks),
                            static_cast<unsigned int>(threads),
                            static_cast<unsigned int>(threads * sizeof(double))>>>(
                variants, reductionMem, reduceDataSize);
            break;
        case 8:
            reduceAdd<8><<<static_cast<unsigned int>(numBlocks),
                           static_cast<unsigned int>(threads),
                           static_cast<unsigned int>(threads * sizeof(double))>>>(
                variants, reductionMem, reduceDataSize);
            break;
        case 4:
            reduceAdd<4><<<static_cast<unsigned int>(numBlocks),
                           static_cast<unsigned int>(threads),
                           static_cast<unsigned int>(threads * sizeof(double))>>>(
                variants, reductionMem, reduceDataSize);
            break;
        case 2:
            reduceAdd<2><<<static_cast<unsigned int>(numBlocks),
                           static_cast<unsigned int>(threads),
                           static_cast<unsigned int>(threads * sizeof(double))>>>(
                variants, reductionMem, reduceDataSize);
            break;
        case 1:
            reduceAdd<1><<<static_cast<unsigned int>(numBlocks),
                           static_cast<unsigned int>(threads),
                           static_cast<unsigned int>(threads * sizeof(double))>>>(
                variants, reductionMem, reduceDataSize);
            break;
        }

        //Copy results back to data
        gpuErrchk(cudaMemcpy(variants, reductionMem, numBlocks * sizeof(double),
                             cudaMemcpyDeviceToDevice));

        //New data length is equal to number of blocks
        reduceDataSize = numBlocks;
    }
    while (numBlocks > 1); //Keep reducing until only 1 block was used
}
