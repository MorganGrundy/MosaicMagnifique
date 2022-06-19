#pragma once

#ifdef CUDA

#include <gtest/gtest.h>
#include <gmock/gmock-matchers.h>
#include <algorithm>
#include <vector>
#include <execution>

#include "..\src\Photomosaic\CUDA\CUDAUtility.h"
#include "testutility.h"
#include "..\src\Photomosaic\CUDA\PhotomosaicGenerator.cuh"
#include "..\src\Photomosaic\CUDA\Reduction.cuh"

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
    const size_t noMainImage = 2;
    std::vector<std::vector<double>> h_hVariants(noMainImage);
    for (size_t i = 0; i < noMainImage; ++i)
        h_hVariants.at(i) = TestUtility::createRandom<double>(noLibraryImages, { {0, 100} });
    std::vector<double *> h_dVariants(noMainImage);
    for (size_t i = 0; i < noMainImage; ++i)
    {
        gpuErrchk(cudaMalloc((void **)&h_dVariants.at(i), noLibraryImages * sizeof(double)));
        gpuErrchk(cudaMemcpy(h_dVariants.at(i), h_hVariants.at(i).data(), noLibraryImages * sizeof(double), cudaMemcpyHostToDevice));
    }
    double **d_dVariants;
    gpuErrchk(cudaMalloc((void **)&d_dVariants, noMainImage * sizeof(double *)));
    gpuErrchk(cudaMemcpy(d_dVariants, h_dVariants.data(), noMainImage * sizeof(double *), cudaMemcpyHostToDevice));

    int cellY = 2, cellX = 2;
    //Calculate repeats at cell position (2, 2) with CUDA kernel
    calculateRepeatsKernelWrapper(d_dVariants, noMainImage,
                                  d_bestFit, noLibraryImages,
                                  size, cellX, cellY,
                                  0,
                                  repeatRange, repeatAddition);
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());

    //Create results
    std::vector<double *> h_dCUDAResults(noMainImage);
    gpuErrchk(cudaMemcpy(h_dCUDAResults.data(), d_dVariants, noMainImage * sizeof(double *), cudaMemcpyDeviceToHost));
    std::vector<std::vector<double>> h_hCUDAResults(noMainImage, std::vector<double>(noLibraryImages, 0));
    for (size_t i = 0; i < noMainImage; ++i)
        gpuErrchk(cudaMemcpy(h_hCUDAResults.at(i).data(), h_dCUDAResults.at(i), noLibraryImages * sizeof(double), cudaMemcpyDeviceToHost));

    gpuErrchk(cudaFree(d_bestFit));
    gpuErrchk(cudaFree(d_dVariants));
    for (size_t i = 0; i < noMainImage; ++i)
        gpuErrchk(cudaFree(h_dVariants.at(i)));

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
            {
                for (size_t mainI = 0; mainI < noMainImage; ++mainI)
                    h_hVariants[mainI][bestFit[cellIndex]] += repeatAddition;
            }
            else
                break;
        }
        if (static_cast<size_t>(y * size) >= cellPosition)
            break;
    }

    //Compare results
    for (size_t libI = 0; libI < noLibraryImages; ++libI)
        for (size_t mainI = 0; mainI < noMainImage; ++mainI)
            ASSERT_EQ(h_hCUDAResults.at(mainI).at(libI), h_hVariants.at(mainI).at(libI)) << "Lib " << libI << ", Main " << mainI;
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
    const size_t noMainImage = 2;
    std::vector<std::vector<double>> h_hVariants(noMainImage);
    for (size_t i = 0; i < noMainImage; ++i)
        h_hVariants.at(i) = TestUtility::createRandom<double>(noLibraryImages, { {0, 1000} });
    std::vector<double *> h_dVariants(noMainImage);
    for (size_t i = 0; i < noMainImage; ++i)
    {
        gpuErrchk(cudaMalloc((void **)&h_dVariants.at(i), noLibraryImages * sizeof(double)));
        gpuErrchk(cudaMemcpy(h_dVariants.at(i), h_hVariants.at(i).data(), noLibraryImages * sizeof(double), cudaMemcpyHostToDevice));
    }
    double **d_dVariants;
    gpuErrchk(cudaMalloc((void **)&d_dVariants, noMainImage * sizeof(double *)));
    gpuErrchk(cudaMemcpy(d_dVariants, h_dVariants.data(), noMainImage * sizeof(double *), cudaMemcpyHostToDevice));

    //Run find lowest kernel
    findLowestKernelWrapper(d_lowestVariant, d_bestFit, d_dVariants, noLibraryImages, noMainImage);
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
    gpuErrchk(cudaFree(d_dVariants));
    for (size_t i = 0; i < noMainImage; ++i)
        gpuErrchk(cudaFree(h_dVariants.at(i)));

    //Calculate find lowest on host
    for (size_t libI = 0; libI < noLibraryImages; ++libI)
    {
        for (size_t mainI = 0; mainI < noMainImage; ++mainI)
        {
            if (h_hVariants.at(mainI).at(libI) < lowestVariant)
            {
                lowestVariant = h_hVariants.at(mainI).at(libI);
                bestFit = libI;
            }
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
        variants.push_back(TestUtility::randNum<float>(0, 1000));
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
    reduceAddKernelWrapper(blockSize, imageSize * imageSize, d_variants, d_reductionMem, 0);
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

TEST(CUDAKernel, Flatten)
{
    srand(static_cast<unsigned int>(time(NULL)));

    const size_t noLibraryImages = 1 << 14;
    const size_t spacing = 5;

    //Create variants
    const size_t noMainImage = 2;
    std::vector<std::vector<double>> h_hVariants(noMainImage, std::vector<double>(noLibraryImages * spacing, 0));
    for (size_t mainI = 0; mainI < noMainImage; ++mainI)
        for (size_t libI = 0; libI < noLibraryImages; ++libI)
            h_hVariants.at(mainI).at(libI * spacing) = TestUtility::randNum<double>(0, 1000);
    std::vector<double *> h_dVariants(noMainImage);
    for (size_t i = 0; i < noMainImage; ++i)
    {
        gpuErrchk(cudaMalloc((void **)&h_dVariants.at(i), noLibraryImages * spacing * sizeof(double)));
        gpuErrchk(cudaMemcpy(h_dVariants.at(i), h_hVariants.at(i).data(), noLibraryImages * spacing * sizeof(double), cudaMemcpyHostToDevice));
    }
    double **d_dVariants;
    gpuErrchk(cudaMalloc((void **)&d_dVariants, noMainImage * sizeof(double *)));
    gpuErrchk(cudaMemcpy(d_dVariants, h_dVariants.data(), noMainImage * sizeof(double *), cudaMemcpyHostToDevice));

    //Get CUDA block size
    cudaDeviceProp deviceProp;
    gpuErrchk(cudaGetDeviceProperties(&deviceProp, 0));
    const size_t blockSize = deviceProp.maxThreadsPerBlock;

    flattenKernelWrapper(d_dVariants, noMainImage, noLibraryImages, spacing, blockSize);
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());

    //Create results
    std::vector<double *> h_dCUDAResults(noMainImage);
    gpuErrchk(cudaMemcpy(h_dCUDAResults.data(), d_dVariants, noMainImage * sizeof(double *), cudaMemcpyDeviceToHost));
    std::vector<std::vector<double>> h_hCUDAResults(noMainImage, std::vector<double>(noLibraryImages * spacing, 0));
    for (size_t i = 0; i < noMainImage; ++i)
        gpuErrchk(cudaMemcpy(h_hCUDAResults.at(i).data(), h_dCUDAResults.at(i), noLibraryImages * spacing * sizeof(double), cudaMemcpyDeviceToHost));

    gpuErrchk(cudaFree(d_dVariants));
    for (size_t i = 0; i < noMainImage; ++i)
        gpuErrchk(cudaFree(h_dVariants.at(i)));

    //Flatten without CUDA
    auto h_hCPUResults = h_hVariants;
    for (size_t libI = 0; libI < noLibraryImages; ++libI)
        for (size_t mainI = 0; mainI < noMainImage; ++mainI)
            h_hCPUResults[mainI][libI] = h_hCPUResults[mainI][libI * spacing];

    //Compare results
    for (size_t libI = 0; libI < noLibraryImages; ++libI)
    {
        for (size_t mainI = 0; mainI < noMainImage; ++mainI)
        {
            auto cudaDist = std::distance(h_hVariants.at(mainI).cbegin(), std::find(h_hVariants.at(mainI).cbegin(), h_hVariants.at(mainI).cend(), h_hCUDAResults.at(mainI).at(libI)));
            auto cpuDist = std::distance(h_hVariants.at(mainI).cbegin(), std::find(h_hVariants.at(mainI).cbegin(), h_hVariants.at(mainI).cend(), h_hCPUResults.at(mainI).at(libI)));
            ASSERT_EQ(h_hCUDAResults.at(mainI).at(libI), h_hCPUResults.at(mainI).at(libI)) << "Lib " << libI << ", Main " << mainI <<
                ", Came from " << cudaDist << " and " << cpuDist << " instead of " << libI * spacing;
        }
    }
}

#endif