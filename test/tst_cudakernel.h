#pragma once

#ifdef CUDA

#include <gtest/gtest.h>
#include <gmock/gmock-matchers.h>
#include <algorithm>
#include <vector>
#include <execution>

#include "cudautility.h"
#include "testutility.h"
#include "photomosaicgenerator.cuh"
#include "reduction.cuh"

TEST(CUDAKernel, CalculateRepeats)
{
    srand(static_cast<unsigned int>(time(NULL)));

    const int size = 5;
    const size_t noLibraryImages = 10;

    const int repeatRange = 2;
    const size_t repeatAddition = 500;

    //Create best fits
    size_t bestFit[size * size];
    for (size_t i = 0; i < size * size; ++i)
        bestFit[i] = rand() % noLibraryImages;
    size_t *d_bestFit;
    gpuErrchk(cudaMalloc((void **)&d_bestFit, size * size * sizeof(size_t)));
    gpuErrchk(cudaMemcpy(d_bestFit, bestFit, size * size * sizeof(size_t), cudaMemcpyHostToDevice));

    //Create variants
    double variants[noLibraryImages];
    for (size_t i = 0; i < noLibraryImages; ++i)
        variants[i] = TestUtility::randFloat(0, 100);
    double *d_variants;
    gpuErrchk(cudaMalloc((void **)&d_variants, noLibraryImages * sizeof(double)));
    gpuErrchk(cudaMemcpy(d_variants, variants, noLibraryImages * sizeof(double),
                         cudaMemcpyHostToDevice));

    int cellY = 2, cellX = 2;
    //Calculate repeats at cell position (2, 2) with CUDA kernel
    calculateRepeatsKernelWrapper(d_variants,
                                  d_bestFit, noLibraryImages,
                                  size, cellX, cellY,
                                  0,
                                  repeatRange, repeatAddition);
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());

    //Create results
    double CUDAResults[noLibraryImages];
    gpuErrchk(cudaMemcpy(CUDAResults, d_variants, noLibraryImages * sizeof(double),
                         cudaMemcpyDeviceToHost));

    gpuErrchk(cudaFree(d_bestFit));
    gpuErrchk(cudaFree(d_variants));

    //Calculate repeats at cell position without CUDA
    size_t cellPosition = cellY * size + cellX;
    for (int y = std::max(0, cellY - repeatRange);
         y <= std::min(size - 1, cellY + repeatRange); ++y)
    {
        for (int x = std::max(0, cellX - repeatRange);
             x <= std::min(size - 1, cellX + repeatRange); ++x)
        {
            size_t cellIndex = y * size + x;
            if (cellIndex < cellPosition)
                variants[bestFit[cellIndex]] += repeatAddition;
            else
                break;
        }
        if (static_cast<size_t>(y * size) >= cellPosition)
            break;
    }

    //Compare results
    for (size_t i = 0; i < noLibraryImages; ++i)
        ASSERT_TRUE(CUDAResults[i] == variants[i]);
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

    gpuErrchk(cudaFree(d_lowestVariant));
    gpuErrchk(cudaFree(d_bestFit));
    gpuErrchk(cudaFree(d_variants));

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
    ASSERT_EQ(lowestVariant, CUDALowestVariant);
    ASSERT_EQ(bestFit, CUDABestFit);
}

TEST(CUDAKernel, AddReduction)
{
    srand(static_cast<unsigned int>(time(NULL)));

    const size_t imageSize = 128;

    //Create variants
    std::vector<double> variants;
    for (size_t i = 0; i < imageSize * imageSize; ++i)
        variants.push_back(TestUtility::randFloat(0, 1000));
    double *d_variants;
    gpuErrchk(cudaMalloc((void **)&d_variants, imageSize * imageSize * sizeof(double)));
    gpuErrchk(cudaMemcpy(d_variants, variants.data(), imageSize * imageSize * sizeof(double),
                         cudaMemcpyHostToDevice));

    //Get CUDA block size
    cudaDeviceProp deviceProp;
    gpuErrchk(cudaGetDeviceProperties(&deviceProp, 0));
    const size_t blockSize = deviceProp.maxThreadsPerBlock;

    //Create reduction memory
    size_t reductionMemSize = ((imageSize * imageSize + blockSize - 1) / blockSize + 1) / 2;
    double *d_reductionMem;
    gpuErrchk(cudaMalloc((void **)&d_reductionMem, reductionMemSize * sizeof(double)));
    gpuErrchk(cudaMemset(d_reductionMem, 0, reductionMemSize * sizeof(double)));

    //Run reduction kernel
    reduceAddKernelWrapper(blockSize, imageSize * imageSize, d_variants, d_reductionMem);
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());

    //Create results
    double CUDAResult = 0;
    gpuErrchk(cudaMemcpy(&CUDAResult, d_variants, sizeof(double), cudaMemcpyDeviceToHost));

    gpuErrchk(cudaFree(d_variants));
    gpuErrchk(cudaFree(d_reductionMem));

    //Calculate sum on host
    const double result = std::reduce(std::execution::par, variants.cbegin(), variants.cend());

    //Compare results
    EXPECT_EQ(result, CUDAResult);
}

#endif