#ifndef TST_CUDAKERNEL_H
#define TST_CUDAKERNEL_H

#include <gtest/gtest.h>
#include <gmock/gmock-matchers.h>

#include "cudaphotomosaicdata.h"

using namespace testing;

//Wrapper for calculate repeats kernel
void calculateRepeatsKernelWrapper(bool *states, size_t *bestFit, size_t *repeats,
                                   const int noXCell,
                                   const int leftRange, const int rightRange,
                                   const int upRange,
                                   const size_t repeatAddition);

TEST(CUDAKernel, CalculateRepeats)
{
    srand(time(NULL));

    const int size = 5;
    const size_t noLibraryImages = 10;

    const size_t repeatRange = 2;
    const size_t repeatAddition = 500;

    //Create cell states, all active
    bool HOST_states[size * size];
    for (size_t i = 0; i < size * size; ++i)
        HOST_states[i] = true;
    bool *cellStates;
    gpuErrchk(cudaMalloc((void **)&cellStates, size * size * sizeof(bool)));
    gpuErrchk(cudaMemcpy(cellStates, HOST_states, size * size * sizeof(bool),
                         cudaMemcpyHostToDevice));

    //Create best fits
    size_t HOST_bestFit[size * size];
    for (size_t i = 0; i < size * size; ++i)
        HOST_bestFit[i] = rand() % noLibraryImages;
    size_t *bestFit;
    gpuErrchk(cudaMalloc((void **)&bestFit, size * size * sizeof(size_t)));
    gpuErrchk(cudaMemcpy(bestFit, HOST_bestFit, size * size * sizeof(size_t),
                         cudaMemcpyHostToDevice));

    //Create repeats, all 0
    size_t HOST_repeats[noLibraryImages];
    for (size_t i = 0; i < noLibraryImages; ++i)
        HOST_repeats[i] = 0;
    size_t *repeats;
    gpuErrchk(cudaMalloc((void **)&repeats, noLibraryImages * sizeof(size_t)));
    gpuErrchk(cudaMemcpy(repeats, HOST_repeats, noLibraryImages * sizeof(size_t),
                         cudaMemcpyHostToDevice));

    size_t cellY = 2, cellX = 2;
    //Calculate repeats at cell position (2, 2) with CUDA kernel
    size_t cellPosition = cellY * size + cellX;
    calculateRepeatsKernelWrapper(cellStates + cellPosition, bestFit + cellPosition,
                                  repeats, size, repeatRange, repeatRange, repeatRange,
                                  repeatAddition);
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());
    gpuErrchk(cudaMemcpy(HOST_repeats, repeats, noLibraryImages * sizeof(size_t),
                         cudaMemcpyDeviceToHost));

    //Calculate repeats at cell position (2, 2) without CUDA
    std::vector<size_t> expectedRepeats(noLibraryImages, 0);
    for (size_t y = cellY - repeatRange; y <= cellY + repeatRange; ++y)
    {
        for (size_t x = cellX - repeatRange; x <= cellX + repeatRange; ++x)
        {
            size_t cellIndex = y * size + x;
            if (cellIndex < cellPosition)
                expectedRepeats.at(HOST_bestFit[cellIndex]) += repeatAddition;
            else
                break;
        }
        if (y * size >= cellPosition)
            break;
    }

    //Compare results
    for (size_t i = 0; i < noLibraryImages; ++i)
        ASSERT_TRUE(HOST_repeats[i] == expectedRepeats.at(i));
}

#endif // TST_CUDAKERNEL_H
