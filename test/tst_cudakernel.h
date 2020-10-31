#ifndef TST_CUDAKERNEL_H
#define TST_CUDAKERNEL_H

#include <gtest/gtest.h>
#include <gmock/gmock-matchers.h>

#include "cudaphotomosaicdata.h"
#include "testutility.h"
#include "photomosaicgenerator.cuh"

using namespace testing;

TEST(CUDAKernel, CalculateRepeats)
{
    srand(static_cast<unsigned int>(time(NULL)));

    const int size = 5;
    const size_t noLibraryImages = 10;

    const int repeatRange = 2;
    const size_t repeatAddition = 500;

    //Create cell states, all active
    bool states[size * size];
    for (size_t i = 0; i < size * size; ++i)
        states[i] = true;
    bool *d_cellStates;
    gpuErrchk(cudaMalloc((void **)&d_cellStates, size * size * sizeof(bool)));
    gpuErrchk(cudaMemcpy(d_cellStates, states, size * size * sizeof(bool), cudaMemcpyHostToDevice));

    //Create best fits
    size_t bestFit[size * size];
    for (size_t i = 0; i < size * size; ++i)
        bestFit[i] = rand() % noLibraryImages;
    size_t *d_bestFit;
    gpuErrchk(cudaMalloc((void **)&d_bestFit, size * size * sizeof(size_t)));
    gpuErrchk(cudaMemcpy(d_bestFit, bestFit, size * size * sizeof(size_t), cudaMemcpyHostToDevice));

    //Create repeats, all 0
    size_t CUDArepeats[noLibraryImages];
    for (size_t i = 0; i < noLibraryImages; ++i)
        CUDArepeats[i] = 0;
    size_t *d_repeats;
    gpuErrchk(cudaMalloc((void **)&d_repeats, noLibraryImages * sizeof(size_t)));
    gpuErrchk(cudaMemcpy(d_repeats, CUDArepeats, noLibraryImages * sizeof(size_t),
                         cudaMemcpyHostToDevice));

    int cellY = 2, cellX = 2;
    //Calculate repeats at cell position (2, 2) with CUDA kernel
    size_t cellPosition = cellY * size + cellX;
    const int leftRange = std::min(repeatRange, cellX);
    const int rightRange = std::min(repeatRange, size - cellX - 1);
    const int upRange = std::min(repeatRange, cellY);
    calculateRepeatsKernelWrapper(d_cellStates + cellPosition, d_bestFit + cellPosition,
                                  d_repeats, size, leftRange, rightRange, upRange,
                                  repeatAddition);
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());
    gpuErrchk(cudaMemcpy(CUDArepeats, d_repeats, noLibraryImages * sizeof(size_t),
                         cudaMemcpyDeviceToHost));

    //Calculate repeats at cell position (2, 2) without CUDA
    std::vector<size_t> expectedRepeats(noLibraryImages, 0);
    for (int y = std::max(0, cellY - repeatRange);
         y <= std::min(size - 1, cellY + repeatRange); ++y)
    {
        for (int x = std::max(0, cellX - repeatRange);
             x <= std::min(size - 1, cellX + repeatRange); ++x)
        {
            size_t cellIndex = y * size + x;
            if (cellIndex < cellPosition)
                expectedRepeats.at(bestFit[cellIndex]) += repeatAddition;
            else
                break;
        }
        if (static_cast<size_t>(y * size) >= cellPosition)
            break;
    }

    //Compare results
    for (size_t i = 0; i < noLibraryImages; ++i)
        ASSERT_TRUE(CUDArepeats[i] == expectedRepeats.at(i));
}

TEST(CUDAKernel, AddRepeats)
{
    srand(static_cast<unsigned int>(time(NULL)));

    const size_t noLibraryImages = 1 << 14;

    //Create variants
    double variants[noLibraryImages];
    for (size_t i = 0; i < noLibraryImages; ++i)
        variants[i] = TestUtility::randFloat(0, 100);
    double *d_variants;
    gpuErrchk(cudaMalloc((void **)&d_variants, noLibraryImages * sizeof(double)));
    gpuErrchk(cudaMemcpy(d_variants, variants, noLibraryImages * sizeof(double),
                         cudaMemcpyHostToDevice));

    //Create repeats
    size_t repeats[noLibraryImages];
    for (size_t i = 0; i < noLibraryImages; ++i)
        repeats[i] = static_cast<size_t>(TestUtility::randFloat(0, 100));
    size_t *d_repeats;
    gpuErrchk(cudaMalloc((void **)&d_repeats, noLibraryImages * sizeof(size_t)));
    gpuErrchk(cudaMemcpy(d_repeats, repeats, noLibraryImages * sizeof(size_t),
                         cudaMemcpyHostToDevice));

    //Run add repeats kernel
    cudaDeviceProp deviceProp;
    gpuErrchk(cudaGetDeviceProperties(&deviceProp, 0));
    const size_t blockSize = deviceProp.maxThreadsPerBlock;
    addRepeatsKernelWrapper(d_variants, d_repeats, noLibraryImages, blockSize);
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());

    //Create results
    double CUDAResults[noLibraryImages];
    gpuErrchk(cudaMemcpy(CUDAResults, d_variants, noLibraryImages * sizeof(double),
                         cudaMemcpyDeviceToHost));

    //Calculate add repeats on host
    for (size_t i = 0; i < noLibraryImages; ++i)
        variants[i] += repeats[i];

    //Compare results
    for (size_t i = 0; i < noLibraryImages; ++i)
        ASSERT_EQ(variants[i], CUDAResults[i]);
}

TEST(CUDAKernel, FindLowest)
{
    srand(static_cast<unsigned int>(time(NULL)));

    const size_t noLibraryImages = 1 << 14;

    //Create lowest variant
    double lowestVariant = std::numeric_limits<double>::max();
    double *d_lowestVariant;
    gpuErrchk(cudaMalloc((void **)&d_lowestVariant, sizeof(double)));
    gpuErrchk(cudaMemcpy(d_lowestVariant, &lowestVariant, sizeof(double), cudaMemcpyHostToDevice));

    //Create best fit
    size_t bestFit = 0;
    size_t *d_bestFit;
    gpuErrchk(cudaMalloc((void **)&d_bestFit, sizeof(size_t)));
    gpuErrchk(cudaMemcpy(d_bestFit, &bestFit, sizeof(size_t), cudaMemcpyHostToDevice));

    //Create variants
    double variants[noLibraryImages];
    for (size_t i = 0; i < noLibraryImages; ++i)
        variants[i] = TestUtility::randFloat(0, 1000);
    double *d_variants;
    gpuErrchk(cudaMalloc((void **)&d_variants, noLibraryImages * sizeof(double)));
    gpuErrchk(cudaMemcpy(d_variants, variants, noLibraryImages * sizeof(double),
                         cudaMemcpyHostToDevice));

    //Run find lowest kernel
    findLowestKernelWrapper(d_lowestVariant, d_bestFit, d_variants, noLibraryImages);
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());

    //Create results
    size_t CUDABestFit = 0;
    double CUDALowestVariant = 0;
    gpuErrchk(cudaMemcpy(&CUDABestFit, d_bestFit, sizeof(size_t), cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(&CUDALowestVariant, d_lowestVariant, sizeof(double),
                         cudaMemcpyDeviceToHost));

    //Calculate find lowest on host
    for (size_t i = 0; i < noLibraryImages; ++i)
    {
        if (variants[i] < lowestVariant)
        {
            lowestVariant = variants[i];
            bestFit = i;
        }
    }

    //Compare results
    std::cout << "Best fit: " << bestFit << ", Variant: " << lowestVariant << "\n";
    ASSERT_EQ(lowestVariant, CUDALowestVariant);
    ASSERT_EQ(bestFit, CUDABestFit);
}

#endif // TST_CUDAKERNEL_H
