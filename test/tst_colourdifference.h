#pragma once

#include <gtest/gtest.h>
#include <gmock/gmock-matchers.h>
#include <opencv2/core.hpp>

#include "..\src\Photomosaic\ColourDifference.h"
#include "testutility.h"

#ifdef CUDA
#include "..\src\Photomosaic\CUDA\CUDAUtility.h"
#include "..\src\Photomosaic\CUDA\PhotomosaicGenerator.cuh"
#endif

struct ColourDiffPair
{
    const cv::Vec3d first;
    const cv::Vec3d second;
    const double difference;
};

TEST(ColourDifference, RGBEuclidean)
{
    //Test data, difference rounded to 8 decimal places
    const std::vector<ColourDiffPair> colourDiffPairs = {
        {{0, 0, 0}, {0, 0, 0}, 0},
        {{255, 255, 255}, {255, 255, 255}, 0},
        {{0, 0, 0}, {255, 255, 255}, 441.67295593},
        {{0, 0, 0}, {255, 0, 0}, 255},
        {{0, 0, 0}, {0, 255, 0}, 255},
        {{0, 0, 0}, {0, 0, 255}, 255},
        {{0, 0, 0}, {10, 10, 10}, 17.32050808},
        {{2, 100, 197}, {220, 34, 0}, 301.14614392}
    };

    for (const auto &colourDiffPair: colourDiffPairs)
    {
        const double result = ColourDifference::calculateRGBEuclidean(colourDiffPair.first,
                                                                      colourDiffPair.second);

        EXPECT_NEAR(result, colourDiffPair.difference, 0.00000001);
    }
}

TEST(ColourDifference, CIE76)
{
    //Test data, difference rounded to 8 decimal places
    const std::vector<ColourDiffPair> colourDiffPairs = {
        {{0, 0, 0}, {0, 0, 0}, 0},
        {{100, 127, 127}, {100, 127, 127}, 0},
        {{0, -128, -128}, {100, 127, 127}, 374.23254802},
        {{0, -128, -128}, {100, -128, -128}, 100},
        {{0, -128, -128}, {0, 127, -128}, 255},
        {{0, -128, -128}, {0, -128, 127}, 255}
    };

    for (const auto &colourDiffPair: colourDiffPairs)
    {
        const double result = ColourDifference::calculateCIE76(colourDiffPair.first,
                                                               colourDiffPair.second);

        EXPECT_NEAR(result, colourDiffPair.difference, 0.00000001);
    }
}

TEST(ColourDifference, CIEDE2000)
{
    //Test data obtained from:
    //Sharma, Gaurav et al. “The CIEDE2000 color-difference formula: Implementation notes,
    //supplementary test data, and mathematical observations.”
    //Color Research and Application 30 (2005): 21-30.
    const std::vector<ColourDiffPair> colourDiffPairs = {
        {{50, 2.6772, -79.7751}, {50, 0, -82.7485}, 2.0425},
        {{50, 3.1571, -77.2803}, {50, 0, -82.7485}, 2.8615},
        {{50, 2.8361, -74.02}, {50, 0, -82.7485}, 3.4412},
        {{50, -1.3802, -84.2814}, {50, 0, -82.7485}, 1},
        {{50, -1.1848, -84.8006}, {50, 0, -82.7485}, 1},
        {{50, -0.9009, -85.5211}, {50, 0, -82.7485}, 1},
        {{50, 0, 0}, {50, -1, 2}, 2.3669},
        {{50, -1, 2}, {50, 0, 0}, 2.3669},
        {{50, 2.49, -0.001}, {50, -2.49, 0.0009}, 7.1792},
        {{50, 2.49, -0.001}, {50, -2.49, 0.001}, 7.1792},
        {{50, 2.49, -0.001}, {50, -2.49, 0.0011}, 7.2195},
        {{50, 2.49, -0.001}, {50, -2.49, 0.0012}, 7.2195},
        {{50, -0.001, 2.49}, {50, 0.0009, -2.49}, 4.8045},
        {{50, -0.001, 2.49}, {50, 0.001, -2.49}, 4.8045},
        {{50, -0.001, 2.49}, {50, 0.0011, -2.49}, 4.7461},
        {{50, 2.5, 0}, {50, 0, -2.5}, 4.3065},
        {{50, 2.5, 0}, {73, 25, -18}, 27.1492},
        {{50, 2.5, 0}, {61, -5, 29}, 22.8977},
        {{50, 2.5, 0}, {56, -27, -3}, 31.9030},
        {{50, 2.5, 0}, {58, 24, 15}, 19.4535},
        {{50, 2.5, 0}, {50, 3.1736, 0.5854}, 1},
        {{50, 2.5, 0}, {50, 3.2972, 0}, 1},
        {{50, 2.5, 0}, {50, 1.8634, 0.5757}, 1},
        {{50, 2.5, 0}, {50, 3.2592, 0.335}, 1},
        {{60.2574, -34.0099, 36.2677}, {60.4626, -34.1751, 39.4387}, 1.2644},
        {{63.0109, -31.0961, -5.8663}, {62.8187, -29.7946, -4.0864}, 1.263},
        {{61.2901, 3.7196, -5.3901}, {61.4292, 2.248, -4.962}, 1.8731},
        {{35.0831, -44.1164, 3.7933}, {35.0232, -40.0716, 1.5901}, 1.8645},
        {{22.7233, 20.0904, -46.694}, {23.0331, 14.973, -42.5619}, 2.0373},
        {{36.4612, 47.858, 18.3852}, {36.2715, 50.5065, 21.2231}, 1.4146},
        {{90.8027, -2.0831, 1.441}, {91.1528, -1.6435, 0.0447}, 1.4441},
        {{90.9257, -0.5406, -0.9208}, {88.6381, -0.8985, -0.7239}, 1.5381},
        {{6.7747, -0.2908, -2.4247}, {5.8714, -0.0985, -2.2286}, 0.6377},
        {{2.0776, 0.0795, -1.135}, {0.9033, -0.0636, -0.5514}, 0.9082}
    };

    for (const auto &colourDiffPair: colourDiffPairs)
    {
        const double result = ColourDifference::calculateCIEDE2000(colourDiffPair.first,
                                                                   colourDiffPair.second);

        EXPECT_NEAR(result, colourDiffPair.difference, 0.0001);
    }
}

#ifdef CUDA

struct ColourDiffPairFloat
{
    const cv::Vec3f first;
    const cv::Vec3f second;
    const double difference;
};

TEST(ColourDifference, CUDA_RGBEuclidean)
{
    //Test data, difference rounded to 8 decimal places
    const std::vector<ColourDiffPairFloat> colourDiffPairs = {
        {{0, 0, 0}, {0, 0, 0}, 0},
        {{255, 255, 255}, {255, 255, 255}, 0},
        {{0, 0, 0}, {255, 255, 255}, 441.67295593},
        {{0, 0, 0}, {255, 0, 0}, 255},
        {{0, 0, 0}, {0, 255, 0}, 255},
        {{0, 0, 0}, {0, 0, 255}, 255},
        {{0, 0, 0}, {10, 10, 10}, 17.32050808},
        {{2, 100, 197}, {220, 34, 0}, 301.14614392}
    };

    //Create mask on GPU
    uchar HOST_mask = 1;
    uchar *mask;
    gpuErrchk(cudaMalloc((void **)&mask, sizeof(uchar)));
    gpuErrchk(cudaMemcpy(mask, &HOST_mask, sizeof(uchar), cudaMemcpyHostToDevice));

    //Create target area on GPU
    size_t HOST_target_area[4] = {0, 1, 0, 1};
    size_t *target_area;
    gpuErrchk(cudaMalloc((void **)&target_area, 4 * sizeof(size_t)));
    gpuErrchk(cudaMemcpy(target_area, &HOST_target_area, 4 * sizeof(size_t),
                         cudaMemcpyHostToDevice));

    //Allocate pair on GPU
    float *first, *second;
    gpuErrchk(cudaMalloc((void **)&first, 3 * sizeof(float)));
    gpuErrchk(cudaMalloc((void **)&second, 3 * sizeof(float)));

    //Allocate memory on GPU for result
    double HOST_result;
    double *result;
    gpuErrchk(cudaMalloc((void **)&result, sizeof(double)));

    for (const auto &colourDiffPair: colourDiffPairs)
    {
        //Copy pair to GPU
        gpuErrchk(cudaMemcpy(first, colourDiffPair.first.val, 3 * sizeof(float),
                             cudaMemcpyHostToDevice));
        gpuErrchk(cudaMemcpy(second, colourDiffPair.second.val, 3 * sizeof(float),
                             cudaMemcpyHostToDevice));

        //Calculate differences
        cudaDeviceProp deviceProp;
        gpuErrchk(cudaGetDeviceProperties(&deviceProp, 0));
        const size_t blockSize = deviceProp.maxThreadsPerBlock;
        euclideanDifferenceKernelWrapper(first, second, 1, mask, 1, 3, target_area, result,
                                         blockSize);
        gpuErrchk(cudaPeekAtLastError());
        gpuErrchk(cudaDeviceSynchronize());

        //Copy result to host
        gpuErrchk(cudaMemcpy(&HOST_result, result, sizeof(double), cudaMemcpyDeviceToHost));

        //Check result
        EXPECT_NEAR(HOST_result, colourDiffPair.difference, 0.00000001);
    }

    //Free mask
    gpuErrchk(cudaFree(mask));
    //Free target area
    gpuErrchk(cudaFree(target_area));
    //Free pair
    gpuErrchk(cudaFree(first));
    gpuErrchk(cudaFree(second));
    //Free result
    gpuErrchk(cudaFree(result));
}

TEST(ColourDifference, CUDA_CIE76)
{
    //Test data, difference rounded to 8 decimal places
    const std::vector<ColourDiffPairFloat> colourDiffPairs = {
        {{0, 0, 0}, {0, 0, 0}, 0},
        {{100, 127, 127}, {100, 127, 127}, 0},
        {{0, -128, -128}, {100, 127, 127}, 374.23254802},
        {{0, -128, -128}, {100, -128, -128}, 100},
        {{0, -128, -128}, {0, 127, -128}, 255},
        {{0, -128, -128}, {0, -128, 127}, 255}
    };

    //Create mask on GPU
    uchar HOST_mask = 1;
    uchar *mask;
    gpuErrchk(cudaMalloc((void **)&mask, sizeof(uchar)));
    gpuErrchk(cudaMemcpy(mask, &HOST_mask, sizeof(uchar), cudaMemcpyHostToDevice));

    //Create target area on GPU
    size_t HOST_target_area[4] = {0, 1, 0, 1};
    size_t *target_area;
    gpuErrchk(cudaMalloc((void **)&target_area, 4 * sizeof(size_t)));
    gpuErrchk(cudaMemcpy(target_area, &HOST_target_area, 4 * sizeof(size_t),
                         cudaMemcpyHostToDevice));

    //Allocate pair on GPU
    float *first, *second;
    gpuErrchk(cudaMalloc((void **)&first, 3 * sizeof(float)));
    gpuErrchk(cudaMalloc((void **)&second, 3 * sizeof(float)));

    //Allocate memory on GPU for result
    double HOST_result;
    double *result;
    gpuErrchk(cudaMalloc((void **)&result, sizeof(double)));

    for (const auto &colourDiffPair: colourDiffPairs)
    {
        //Copy pair to GPU
        gpuErrchk(cudaMemcpy(first, colourDiffPair.first.val, 3 * sizeof(float),
                             cudaMemcpyHostToDevice));
        gpuErrchk(cudaMemcpy(second, colourDiffPair.second.val, 3 * sizeof(float),
                             cudaMemcpyHostToDevice));

        //Calculate differences
        cudaDeviceProp deviceProp;
        gpuErrchk(cudaGetDeviceProperties(&deviceProp, 0));
        const size_t blockSize = deviceProp.maxThreadsPerBlock;
        euclideanDifferenceKernelWrapper(first, second, 1, mask, 1, 3, target_area, result,
                                         blockSize);
        gpuErrchk(cudaPeekAtLastError());
        gpuErrchk(cudaDeviceSynchronize());

        //Copy result to host
        gpuErrchk(cudaMemcpy(&HOST_result, result, sizeof(double), cudaMemcpyDeviceToHost));

        //Check result
        EXPECT_NEAR(HOST_result, colourDiffPair.difference, 0.00000001);
    }

    //Free mask
    gpuErrchk(cudaFree(mask));
    //Free target area
    gpuErrchk(cudaFree(target_area));
    //Free pair
    gpuErrchk(cudaFree(first));
    gpuErrchk(cudaFree(second));
    //Free result
    gpuErrchk(cudaFree(result));
}

TEST(ColourDifference, CUDA_CIEDE2000)
{
    //Test data obtained from:
    //Sharma, Gaurav et al. “The CIEDE2000 color-difference formula: Implementation notes,
    //supplementary test data, and mathematical observations.”
    //Color Research and Application 30 (2005): 21-30.
    const std::vector<ColourDiffPairFloat> colourDiffPairs = {
        {{50, 2.6772, -79.7751}, {50, 0, -82.7485}, 2.0425},
        {{50, 3.1571, -77.2803}, {50, 0, -82.7485}, 2.8615},
        {{50, 2.8361, -74.02}, {50, 0, -82.7485}, 3.4412},
        {{50, -1.3802, -84.2814}, {50, 0, -82.7485}, 1},
        {{50, -1.1848, -84.8006}, {50, 0, -82.7485}, 1},
        {{50, -0.9009, -85.5211}, {50, 0, -82.7485}, 1},
        {{50, 0, 0}, {50, -1, 2}, 2.3669},
        {{50, -1, 2}, {50, 0, 0}, 2.3669},
        {{50, 2.49, -0.001}, {50, -2.49, 0.0009}, 7.1792},
        {{50, 2.49, -0.001}, {50, -2.49, 0.001}, 7.1792},
        {{50, 2.49, -0.001}, {50, -2.49, 0.0011}, 7.2195},
        {{50, 2.49, -0.001}, {50, -2.49, 0.0012}, 7.2195},
        {{50, -0.001, 2.49}, {50, 0.0009, -2.49}, 4.8045},
        {{50, -0.001, 2.49}, {50, 0.001, -2.49}, 4.8045},
        {{50, -0.001, 2.49}, {50, 0.0011, -2.49}, 4.7461},
        {{50, 2.5, 0}, {50, 0, -2.5}, 4.3065},
        {{50, 2.5, 0}, {73, 25, -18}, 27.1492},
        {{50, 2.5, 0}, {61, -5, 29}, 22.8977},
        {{50, 2.5, 0}, {56, -27, -3}, 31.9030},
        {{50, 2.5, 0}, {58, 24, 15}, 19.4535},
        {{50, 2.5, 0}, {50, 3.1736, 0.5854}, 1},
        {{50, 2.5, 0}, {50, 3.2972, 0}, 1},
        {{50, 2.5, 0}, {50, 1.8634, 0.5757}, 1},
        {{50, 2.5, 0}, {50, 3.2592, 0.335}, 1},
        {{60.2574, -34.0099, 36.2677}, {60.4626, -34.1751, 39.4387}, 1.2644},
        {{63.0109, -31.0961, -5.8663}, {62.8187, -29.7946, -4.0864}, 1.263},
        {{61.2901, 3.7196, -5.3901}, {61.4292, 2.248, -4.962}, 1.8731},
        {{35.0831, -44.1164, 3.7933}, {35.0232, -40.0716, 1.5901}, 1.8645},
        {{22.7233, 20.0904, -46.694}, {23.0331, 14.973, -42.5619}, 2.0373},
        {{36.4612, 47.858, 18.3852}, {36.2715, 50.5065, 21.2231}, 1.4146},
        {{90.8027, -2.0831, 1.441}, {91.1528, -1.6435, 0.0447}, 1.4441},
        {{90.9257, -0.5406, -0.9208}, {88.6381, -0.8985, -0.7239}, 1.5381},
        {{6.7747, -0.2908, -2.4247}, {5.8714, -0.0985, -2.2286}, 0.6377},
        {{2.0776, 0.0795, -1.135}, {0.9033, -0.0636, -0.5514}, 0.9082}
    };

    //Create mask on GPU
    uchar HOST_mask = 1;
    uchar *mask;
    gpuErrchk(cudaMalloc((void **)&mask, sizeof(uchar)));
    gpuErrchk(cudaMemcpy(mask, &HOST_mask, sizeof(uchar), cudaMemcpyHostToDevice));

    //Create target area on GPU
    size_t HOST_target_area[4] = {0, 1, 0, 1};
    size_t *target_area;
    gpuErrchk(cudaMalloc((void **)&target_area, 4 * sizeof(size_t)));
    gpuErrchk(cudaMemcpy(target_area, &HOST_target_area, 4 * sizeof(size_t),
                         cudaMemcpyHostToDevice));

    //Allocate pair on GPU
    float *first, *second;
    gpuErrchk(cudaMalloc((void **)&first, 3 * sizeof(float)));
    gpuErrchk(cudaMalloc((void **)&second, 3 * sizeof(float)));

    //Allocate memory on GPU for result
    double HOST_result;
    double *result;
    gpuErrchk(cudaMalloc((void **)&result, sizeof(double)));

    for (const auto &colourDiffPair: colourDiffPairs)
    {
        //Copy pair to GPU
        gpuErrchk(cudaMemcpy(first, colourDiffPair.first.val, 3 * sizeof(float),
                             cudaMemcpyHostToDevice));
        gpuErrchk(cudaMemcpy(second, colourDiffPair.second.val, 3 * sizeof(float),
                             cudaMemcpyHostToDevice));

        //Calculate differences
        cudaDeviceProp deviceProp;
        gpuErrchk(cudaGetDeviceProperties(&deviceProp, 0));
        const size_t blockSize = deviceProp.maxThreadsPerBlock;
        CIEDE2000DifferenceKernelWrapper(first, second, 1, mask, 1, 3, target_area, result,
                                         blockSize);
        gpuErrchk(cudaPeekAtLastError());
        gpuErrchk(cudaDeviceSynchronize());

        //Copy result to host
        gpuErrchk(cudaMemcpy(&HOST_result, result, sizeof(double), cudaMemcpyDeviceToHost));

        //Check result
        EXPECT_NEAR(HOST_result, colourDiffPair.difference, 0.0001);
    }

    //Free mask
    gpuErrchk(cudaFree(mask));
    //Free target area
    gpuErrchk(cudaFree(target_area));
    //Free pair
    gpuErrchk(cudaFree(first));
    gpuErrchk(cudaFree(second));
    //Free result
    gpuErrchk(cudaFree(result));
}

//Creates random colour pairs and compares results from RGBEuclidean and CUDA RGBEuclidean difference
TEST(ColourDifference, RANDOM_RGBEuclideanvsCUDA_RGBEuclidean)
{
    constexpr size_t iterations = 1ULL << 14;
    srand(static_cast<unsigned int>(time(NULL)));

    //Create mask on GPU
    uchar HOST_mask = 1;
    uchar *mask;
    gpuErrchk(cudaMalloc((void **)&mask, sizeof(uchar)));
    gpuErrchk(cudaMemcpy(mask, &HOST_mask, sizeof(uchar), cudaMemcpyHostToDevice));

    //Create target area on GPU
    size_t HOST_target_area[4] = { 0, 1, 0, 1 };
    size_t *target_area;
    gpuErrchk(cudaMalloc((void **)&target_area, 4 * sizeof(size_t)));
    gpuErrchk(cudaMemcpy(target_area, &HOST_target_area, 4 * sizeof(size_t),
        cudaMemcpyHostToDevice));

    //Allocate pair on GPU
    float *first, *second;
    gpuErrchk(cudaMalloc((void **)&first, 3 * sizeof(float)));
    gpuErrchk(cudaMalloc((void **)&second, 3 * sizeof(float)));

    //Allocate memory on GPU for result
    double HOST_result;
    double *result;
    gpuErrchk(cudaMalloc((void **)&result, sizeof(double)));

    for (size_t i = 0; i < iterations; ++i)
    {
        //Create random colour diff pair
        ColourDiffPairFloat colourDiffPair = {
            {TestUtility::randFloat(0, 255), TestUtility::randFloat(0, 255),
             TestUtility::randFloat(0, 255)},
            {TestUtility::randFloat(0, 255), TestUtility::randFloat(0, 255),
             TestUtility::randFloat(0, 255)}, 0 };

        //Copy pair to GPU
        gpuErrchk(cudaMemcpy(first, colourDiffPair.first.val, 3 * sizeof(float),
            cudaMemcpyHostToDevice));
        gpuErrchk(cudaMemcpy(second, colourDiffPair.second.val, 3 * sizeof(float),
            cudaMemcpyHostToDevice));

        //Calculate differences
        cudaDeviceProp deviceProp;
        gpuErrchk(cudaGetDeviceProperties(&deviceProp, 0));
        const size_t blockSize = deviceProp.maxThreadsPerBlock;
        euclideanDifferenceKernelWrapper(first, second, 1, mask, 1, 3, target_area, result,
            blockSize);
        gpuErrchk(cudaPeekAtLastError());
        gpuErrchk(cudaDeviceSynchronize());

        //Copy result to host
        gpuErrchk(cudaMemcpy(&HOST_result, result, sizeof(double), cudaMemcpyDeviceToHost));

        const double cpu_result = ColourDifference::calculateRGBEuclidean(colourDiffPair.first,
            colourDiffPair.second);

        //Check result
        EXPECT_NEAR(HOST_result, cpu_result, 0.0001);
    }

    //Free mask
    gpuErrchk(cudaFree(mask));
    //Free target area
    gpuErrchk(cudaFree(target_area));
    //Free pair
    gpuErrchk(cudaFree(first));
    gpuErrchk(cudaFree(second));
    //Free result
    gpuErrchk(cudaFree(result));
}

//Creates random colour pairs and compares results from CIE76 and CUDA CIE76 difference
TEST(ColourDifference, RANDOM_CIE76vsCUDA_CIE76)
{
    constexpr size_t iterations = 1ULL << 14;
    srand(static_cast<unsigned int>(time(NULL)));

    //Create mask on GPU
    uchar HOST_mask = 1;
    uchar *mask;
    gpuErrchk(cudaMalloc((void **)&mask, sizeof(uchar)));
    gpuErrchk(cudaMemcpy(mask, &HOST_mask, sizeof(uchar), cudaMemcpyHostToDevice));

    //Create target area on GPU
    size_t HOST_target_area[4] = {0, 1, 0, 1};
    size_t *target_area;
    gpuErrchk(cudaMalloc((void **)&target_area, 4 * sizeof(size_t)));
    gpuErrchk(cudaMemcpy(target_area, &HOST_target_area, 4 * sizeof(size_t),
                         cudaMemcpyHostToDevice));

    //Allocate pair on GPU
    float *first, *second;
    gpuErrchk(cudaMalloc((void **)&first, 3 * sizeof(float)));
    gpuErrchk(cudaMalloc((void **)&second, 3 * sizeof(float)));

    //Allocate memory on GPU for result
    double HOST_result;
    double *result;
    gpuErrchk(cudaMalloc((void **)&result, sizeof(double)));

    for (size_t i = 0; i < iterations; ++i)
    {
        //Create random colour diff pair
        ColourDiffPairFloat colourDiffPair = {
            {TestUtility::randFloat(0, 100), TestUtility::randFloat(-128, 127),
             TestUtility::randFloat(-128, 127)},
            {TestUtility::randFloat(0, 100), TestUtility::randFloat(-128, 127),
             TestUtility::randFloat(-128, 127)}, 0};

        //Copy pair to GPU
        gpuErrchk(cudaMemcpy(first, colourDiffPair.first.val, 3 * sizeof(float),
                             cudaMemcpyHostToDevice));
        gpuErrchk(cudaMemcpy(second, colourDiffPair.second.val, 3 * sizeof(float),
                             cudaMemcpyHostToDevice));

        //Calculate differences
        cudaDeviceProp deviceProp;
        gpuErrchk(cudaGetDeviceProperties(&deviceProp, 0));
        const size_t blockSize = deviceProp.maxThreadsPerBlock;
        euclideanDifferenceKernelWrapper(first, second, 1, mask, 1, 3, target_area, result,
                                         blockSize);
        gpuErrchk(cudaPeekAtLastError());
        gpuErrchk(cudaDeviceSynchronize());

        //Copy result to host
        gpuErrchk(cudaMemcpy(&HOST_result, result, sizeof(double), cudaMemcpyDeviceToHost));

        const double cpu_result = ColourDifference::calculateCIE76(colourDiffPair.first,
                                                                   colourDiffPair.second);

        //Check result
        EXPECT_NEAR(HOST_result, cpu_result, 0.0001);
    }

    //Free mask
    gpuErrchk(cudaFree(mask));
    //Free target area
    gpuErrchk(cudaFree(target_area));
    //Free pair
    gpuErrchk(cudaFree(first));
    gpuErrchk(cudaFree(second));
    //Free result
    gpuErrchk(cudaFree(result));
}

//Creates random colour pairs and compares results from CIEDE2000 and CUDA CIEDE2000 difference
TEST(ColourDifference, RANDOM_CIEDE2000vsCUDA_CIEDE2000)
{
    constexpr size_t iterations = 1ULL << 14;
    srand(static_cast<unsigned int>(time(NULL)));

    //Create mask on GPU
    uchar HOST_mask = 1;
    uchar *mask;
    gpuErrchk(cudaMalloc((void **)&mask, sizeof(uchar)));
    gpuErrchk(cudaMemcpy(mask, &HOST_mask, sizeof(uchar), cudaMemcpyHostToDevice));

    //Create target area on GPU
    size_t HOST_target_area[4] = { 0, 1, 0, 1 };
    size_t *target_area;
    gpuErrchk(cudaMalloc((void **)&target_area, 4 * sizeof(size_t)));
    gpuErrchk(cudaMemcpy(target_area, &HOST_target_area, 4 * sizeof(size_t),
        cudaMemcpyHostToDevice));

    //Allocate pair on GPU
    float *first, *second;
    gpuErrchk(cudaMalloc((void **)&first, 3 * sizeof(float)));
    gpuErrchk(cudaMalloc((void **)&second, 3 * sizeof(float)));

    //Allocate memory on GPU for result
    double HOST_result;
    double *result;
    gpuErrchk(cudaMalloc((void **)&result, sizeof(double)));

    for (size_t i = 0; i < iterations; ++i)
    {
        //Create random colour diff pair
        ColourDiffPairFloat colourDiffPair = {
            {TestUtility::randFloat(0, 100), TestUtility::randFloat(-128, 127),
             TestUtility::randFloat(-128, 127)},
            {TestUtility::randFloat(0, 100), TestUtility::randFloat(-128, 127),
             TestUtility::randFloat(-128, 127)}, 0 };

        //Copy pair to GPU
        gpuErrchk(cudaMemcpy(first, colourDiffPair.first.val, 3 * sizeof(float),
            cudaMemcpyHostToDevice));
        gpuErrchk(cudaMemcpy(second, colourDiffPair.second.val, 3 * sizeof(float),
            cudaMemcpyHostToDevice));

        //Calculate differences
        cudaDeviceProp deviceProp;
        gpuErrchk(cudaGetDeviceProperties(&deviceProp, 0));
        const size_t blockSize = deviceProp.maxThreadsPerBlock;
        CIEDE2000DifferenceKernelWrapper(first, second, 1, mask, 1, 3, target_area, result,
            blockSize);
        gpuErrchk(cudaPeekAtLastError());
        gpuErrchk(cudaDeviceSynchronize());

        //Copy result to host
        gpuErrchk(cudaMemcpy(&HOST_result, result, sizeof(double), cudaMemcpyDeviceToHost));

        const double cpu_result = ColourDifference::calculateCIEDE2000(colourDiffPair.first,
            colourDiffPair.second);

        //Check result
        EXPECT_NEAR(HOST_result, cpu_result, 0.0001);
    }

    //Free mask
    gpuErrchk(cudaFree(mask));
    //Free target area
    gpuErrchk(cudaFree(target_area));
    //Free pair
    gpuErrchk(cudaFree(first));
    gpuErrchk(cudaFree(second));
    //Free result
    gpuErrchk(cudaFree(result));
}
#endif