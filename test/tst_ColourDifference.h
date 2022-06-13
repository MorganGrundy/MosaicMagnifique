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

///////////////////////////////////////////////////////////// CPU colour difference against known values /////////////////////////////////////////////////////////////
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

std::vector<double> calculateCPUDiff(const std::vector<float> &firsts, const std::vector<float> &seconds, const size_t size, const ColourDifference::Type diffType, const size_t edgeCaseRows)
{
    const size_t fullSize = size * size;
    const size_t totalFloats = fullSize * 3;
    if (totalFloats != firsts.size() || totalFloats != seconds.size())
        throw std::invalid_argument("Vectors firsts and seconds must contain 3 * size^2 elements");

    std::vector<double> result(fullSize);

    for (size_t i = 0; i < fullSize; ++i)
    {
        const cv::Vec3f first(firsts.at(i*3), firsts.at(i*3 + 1), firsts.at(i*3 + 2));
        const cv::Vec3f second(seconds.at(i*3), seconds.at(i*3 + 1), seconds.at(i*3 + 2));

        //Set the result for the last edgeCaseRows rows to 0, allows us to compare against the CUDA edge case kernel
        if (i >= size * (size - edgeCaseRows))
            result.at(i) = 0;
        else
            result.at(i) = ColourDifference::getFunction(diffType)(first, second);
    }

    return result;
}

std::vector<double> calculateCUDADiff(const std::vector<float> &firsts, const std::vector<float> &seconds, const size_t size, const ColourDifference::Type diffType, const size_t edgeCaseRows)
{
    const size_t fullSize = size * size;
    const size_t totalFloats = fullSize * 3;

    //Allocate pair on GPU
    float *d_first, *d_second;
    gpuErrchk(cudaMalloc((void **)&d_first, totalFloats * sizeof(float)));
    gpuErrchk(cudaMalloc((void **)&d_second, totalFloats * sizeof(float)));
    //Copy pair to GPU
    gpuErrchk(cudaMemcpy(d_first, firsts.data(), totalFloats * sizeof(float), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_second, seconds.data(), totalFloats * sizeof(float), cudaMemcpyHostToDevice));

    //Create mask on GPU
    std::vector<uchar> HOST_mask(totalFloats, 1);
    uchar *mask;
    gpuErrchk(cudaMalloc((void **)&mask, fullSize * sizeof(uchar)));
    gpuErrchk(cudaMemcpy(mask, HOST_mask.data(), fullSize * sizeof(uchar), cudaMemcpyHostToDevice));

    //Create target area on GPU, this is only used if edgeCase is true
    size_t HOST_target_area[4] = { 0, size - edgeCaseRows, 0, size};
    size_t *target_area;
    gpuErrchk(cudaMalloc((void **)&target_area, 4 * sizeof(size_t)));
    gpuErrchk(cudaMemcpy(target_area, &HOST_target_area, 4 * sizeof(size_t), cudaMemcpyHostToDevice));

    //Allocate memory on GPU for result
    std::vector<double> HOST_result(fullSize, 0);
    double *result;
    gpuErrchk(cudaMalloc((void **)&result, fullSize * sizeof(double)));

    //Batch calculate difference with CUDA
    cudaDeviceProp deviceProp;
    gpuErrchk(cudaGetDeviceProperties(&deviceProp, 0));
#ifndef NDEBUG
    const size_t blockSize = deviceProp.maxThreadsPerBlock / 4;
#else
    const size_t blockSize = deviceProp.maxThreadsPerBlock;
#endif
    ColourDifference::getCUDAFunction(diffType, edgeCaseRows > 0)(d_first, d_second, mask, size, target_area, result, blockSize, 0);
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());

    //Copy result to host
    gpuErrchk(cudaMemcpy(HOST_result.data(), result, fullSize * sizeof(double), cudaMemcpyDeviceToHost));

    //Free mask
    gpuErrchk(cudaFree(mask));
    //Free target area
    gpuErrchk(cudaFree(target_area));
    //Free pair
    gpuErrchk(cudaFree(d_first));
    gpuErrchk(cudaFree(d_second));
    //Free result
    gpuErrchk(cudaFree(result));

    return HOST_result;
}

void compareCUDADiff(const std::vector<ColourDiffPairFloat> &colourDiffPairs, const ColourDifference::Type diffType)
{
    const size_t size = 1;
    const size_t fullSize = size * size;
    const size_t totalFloats = fullSize * 3;
    srand(static_cast<unsigned int>(time(NULL)));

    for (size_t i = 0; i < colourDiffPairs.size(); ++i)
    {
        //Convert the first and second Vec3f to a vector
        std::vector<float> h_firsts = { colourDiffPairs.at(i).first.val[0], colourDiffPairs.at(i).first.val[1], colourDiffPairs.at(i).first.val[2] };
        std::vector<float> h_seconds = { colourDiffPairs.at(i).second.val[0], colourDiffPairs.at(i).second.val[1], colourDiffPairs.at(i).second.val[2] };

        //Calculate differences with CUDA
        std::vector<double> cudaDifferences = calculateCUDADiff(h_firsts, h_seconds, size, diffType, false);

        //Compare result against expect difference
        EXPECT_NEAR(cudaDifferences.at(0), colourDiffPairs.at(i).difference, 0.0001);
    }
}

///////////////////////////////////////////////////////////// CUDA colour difference against known values /////////////////////////////////////////////////////////////

TEST(ColourDifference, RGBEuclidean_CUDA)
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

    compareCUDADiff(colourDiffPairs, ColourDifference::Type::RGB_EUCLIDEAN);
}

TEST(ColourDifference, CIE76_CUDA)
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

    compareCUDADiff(colourDiffPairs, ColourDifference::Type::CIE76);
}

TEST(ColourDifference, CIEDE2000_CUDA)
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

    compareCUDADiff(colourDiffPairs, ColourDifference::Type::CIEDE2000);
}

///////////////////////////////////////////////////////////// CPU VS CUDA colour difference of random values /////////////////////////////////////////////////////////////

//Creates random pixel and compares results from CPU RGBEuclidean and CUDA RGBEuclidean difference
//Only calculates the difference with CUDA one at a time
TEST(ColourDifference, RGBEuclidean_CPUvsCUDA)
{
    const size_t iterations = 1ULL << 14;

    const ColourDifference::Type diffType = ColourDifference::Type::RGB_EUCLIDEAN;
    const size_t size = 1;
    const size_t fullSize = size * size;
    const size_t totalFloats = fullSize * 3;
    srand(static_cast<unsigned int>(time(NULL)));

    for (size_t i = 0; i < iterations; ++i)
    {
        //Create random CIELab colours
        std::vector<float> h_firsts = TestUtility::createRandomFloats(totalFloats, { {0, 100}, {-128, 127}, {-128, 127} });
        std::vector<float> h_seconds = TestUtility::createRandomFloats(totalFloats, { {0, 100}, {-128, 127}, {-128, 127} });

        //Calculate differences with CPU and CUDA
        std::vector<double> cpuDifferences = calculateCPUDiff(h_firsts, h_seconds, size, diffType, 0);
        std::vector<double> cudaDifferences = calculateCUDADiff(h_firsts, h_seconds, size, diffType, 0);

        //Compare results
        for (size_t i = 0; i < fullSize; ++i)
            EXPECT_NEAR(cudaDifferences.at(i), cpuDifferences.at(i), 0.0001);
    }
}

//Creates random pixel and compares results from CPU CIE76 and CUDA CIE76 difference
//Only calculates the difference with CUDA one at a time
TEST(ColourDifference, CIE76_CPUvsCUDA)
{
    const size_t iterations = 1ULL << 14;

    const ColourDifference::Type diffType = ColourDifference::Type::CIE76;
    const size_t size = 1;
    const size_t fullSize = size * size;
    const size_t totalFloats = fullSize * 3;
    srand(static_cast<unsigned int>(time(NULL)));

    for (size_t i = 0; i < iterations; ++i)
    {
        //Create random CIELab colours
        std::vector<float> h_firsts = TestUtility::createRandomFloats(totalFloats, { {0, 100}, {-128, 127}, {-128, 127} });
        std::vector<float> h_seconds = TestUtility::createRandomFloats(totalFloats, { {0, 100}, {-128, 127}, {-128, 127} });

        //Calculate differences with CPU and CUDA
        std::vector<double> cpuDifferences = calculateCPUDiff(h_firsts, h_seconds, size, diffType, 0);
        std::vector<double> cudaDifferences = calculateCUDADiff(h_firsts, h_seconds, size, diffType, 0);

        //Compare results
        for (size_t i = 0; i < fullSize; ++i)
            EXPECT_NEAR(cudaDifferences.at(i), cpuDifferences.at(i), 0.0001);
    }
}

//Creates random pixel and compares results from CPU CIEDE2000 and CUDA CIEDE2000 difference
//Only calculates the difference with CUDA one at a time
TEST(ColourDifference, CIEDE2000_CPUvsCUDA)
{
    const size_t iterations = 1ULL << 14;

    const ColourDifference::Type diffType = ColourDifference::Type::CIEDE2000;
    const size_t size = 1;
    const size_t fullSize = size * size;
    const size_t totalFloats = fullSize * 3;
    srand(static_cast<unsigned int>(time(NULL)));

    for (size_t i = 0; i < iterations; ++i)
    {
        //Create random CIELab colours
        std::vector<float> h_firsts = TestUtility::createRandomFloats(totalFloats, { {0, 100}, {-128, 127}, {-128, 127} });
        std::vector<float> h_seconds = TestUtility::createRandomFloats(totalFloats, { {0, 100}, {-128, 127}, {-128, 127} });

        //Calculate differences with CPU and CUDA
        std::vector<double> cpuDifferences = calculateCPUDiff(h_firsts, h_seconds, size, diffType, 0);
        std::vector<double> cudaDifferences = calculateCUDADiff(h_firsts, h_seconds, size, diffType, 0);

        //Compare results
        for (size_t i = 0; i < fullSize; ++i)
            EXPECT_NEAR(cudaDifferences.at(i), cpuDifferences.at(i), 0.0001);
    }
}

///////////////////////////////////////////////////////////// CPU VS batch CUDA colour difference of random values /////////////////////////////////////////////////////////////

//Creates random pixels and compares results from CPU RGBEuclidean and CUDA RGBEuclidean difference
//Calculates the difference with CUDA all at once
TEST(ColourDifference, RGBEuclidean_CPUvsBatchCUDA)
{
    const ColourDifference::Type diffType = ColourDifference::Type::RGB_EUCLIDEAN;
    const size_t size = 1000;
    const size_t fullSize = size * size;
    const size_t totalFloats = fullSize * 3;
    srand(static_cast<unsigned int>(time(NULL)));

    //Create random RGB colours
    std::vector<float> h_firsts = TestUtility::createRandomFloats(totalFloats, { {0, 255} });
    std::vector<float> h_seconds = TestUtility::createRandomFloats(totalFloats, { {0, 255} });

    //Calculate differences with CPU and CUDA
    std::vector<double> cpuDifferences = calculateCPUDiff(h_firsts, h_seconds, size, diffType, 0);
    std::vector<double> cudaDifferences = calculateCUDADiff(h_firsts, h_seconds, size, diffType, 0);

    //Compare results
    for (size_t i = 0; i < fullSize; ++i)
        EXPECT_NEAR(cudaDifferences.at(i), cpuDifferences.at(i), 0.0001);
}

//Creates random pixels and compares results from CPU CIE76 and CUDA CIE76 difference
//Calculates the difference with CUDA all at once
TEST(ColourDifference, CIE76_CPUvsBatchCUDA)
{
    const ColourDifference::Type diffType = ColourDifference::Type::CIE76;
    const size_t size = 1000;
    const size_t fullSize = size * size;
    const size_t totalFloats = fullSize * 3;
    srand(static_cast<unsigned int>(time(NULL)));

    //Create random CIELab colours
    std::vector<float> h_firsts = TestUtility::createRandomFloats(totalFloats, { {0, 100}, {-128, 127}, {-128, 127} });
    std::vector<float> h_seconds = TestUtility::createRandomFloats(totalFloats, { {0, 100}, {-128, 127}, {-128, 127} });

    //Calculate differences with CPU and CUDA
    std::vector<double> cpuDifferences = calculateCPUDiff(h_firsts, h_seconds, size, diffType, 0);
    std::vector<double> cudaDifferences = calculateCUDADiff(h_firsts, h_seconds, size, diffType, 0);

    //Compare results
    for (size_t i = 0; i < fullSize; ++i)
        EXPECT_NEAR(cudaDifferences.at(i), cpuDifferences.at(i), 0.0001);
}

//Creates random pixels and compares results from CPU CIEDE2000 and CUDA CIEDE2000 difference
//Calculates the difference with CUDA all at once
TEST(ColourDifference, CIEDE2000_CPUvsBatchCUDA)
{
    const ColourDifference::Type diffType = ColourDifference::Type::CIEDE2000;
    const size_t size = 1000;
    const size_t fullSize = size * size;
    const size_t totalFloats = fullSize * 3;
    srand(static_cast<unsigned int>(time(NULL)));

    //Create random CIELab colours
    std::vector<float> h_firsts = TestUtility::createRandomFloats(totalFloats, { {0, 100}, {-128, 127}, {-128, 127} });
    std::vector<float> h_seconds = TestUtility::createRandomFloats(totalFloats, { {0, 100}, {-128, 127}, {-128, 127} });

    //Calculate differences with CPU and CUDA
    std::vector<double> cpuDifferences = calculateCPUDiff(h_firsts, h_seconds, size, diffType, 0);
    std::vector<double> cudaDifferences = calculateCUDADiff(h_firsts, h_seconds, size, diffType, 0);

    //Compare results
    for (size_t i = 0; i < fullSize; ++i)
        EXPECT_NEAR(cudaDifferences.at(i), cpuDifferences.at(i), 0.0001);
}

///////////////////////////////////////////////////////////// CPU VS CUDA edge case colour difference of random values /////////////////////////////////////////////////////////////

//Creates random pixels and compares results from CPU RGBEuclidean and CUDA RGBEuclidean difference
//Uses the edge case CUDA kernel, ignoring the last quarter of rows
TEST(ColourDifference, RGBEuclidean_CUDAEdgeCase)
{
    const ColourDifference::Type diffType = ColourDifference::Type::RGB_EUCLIDEAN;
    const size_t size = 1000;
    const size_t fullSize = size * size;
    const size_t totalFloats = fullSize * 3;
    srand(static_cast<unsigned int>(time(NULL)));

    const size_t edgeCaseRows = size / 4;

    //Create random RGB colours
    std::vector<float> h_firsts = TestUtility::createRandomFloats(totalFloats, { {0, 255} });
    std::vector<float> h_seconds = TestUtility::createRandomFloats(totalFloats, { {0, 255} });

    //Calculate differences with CPU and CUDA
    std::vector<double> cpuDifferences = calculateCPUDiff(h_firsts, h_seconds, size, diffType, edgeCaseRows);
    std::vector<double> cudaDifferences = calculateCUDADiff(h_firsts, h_seconds, size, diffType, edgeCaseRows);

    //Compare results
    for (size_t i = 0; i < fullSize; ++i)
        ASSERT_NEAR(cudaDifferences.at(i), cpuDifferences.at(i), 0.0001) << "i = " << i;
}

//Creates random pixels and compares results from CPU CIE76 and CUDA CIE76 difference
//Uses the edge case CUDA kernel, ignoring the last quarter of rows
TEST(ColourDifference, CIE76_CUDAEdgeCase)
{
    const ColourDifference::Type diffType = ColourDifference::Type::CIE76;
    const size_t size = 1000;
    const size_t fullSize = size * size;
    const size_t totalFloats = fullSize * 3;
    srand(static_cast<unsigned int>(time(NULL)));

    const size_t edgeCaseRows = size / 4;

    //Create random RGB colours
    std::vector<float> h_firsts = TestUtility::createRandomFloats(totalFloats, { {0, 100}, {-128, 127}, {-128, 127} });
    std::vector<float> h_seconds = TestUtility::createRandomFloats(totalFloats, { {0, 100}, {-128, 127}, {-128, 127} });

    //Calculate differences with CPU and CUDA
    std::vector<double> cpuDifferences = calculateCPUDiff(h_firsts, h_seconds, size, diffType, edgeCaseRows);
    std::vector<double> cudaDifferences = calculateCUDADiff(h_firsts, h_seconds, size, diffType, edgeCaseRows);

    //Compare results
    for (size_t i = 0; i < fullSize; ++i)
        ASSERT_NEAR(cudaDifferences.at(i), cpuDifferences.at(i), 0.0001) << "i = " << i;
}

//Creates random pixels and compares results from CPU CIEDE2000 and CUDA CIEDE2000 difference
//Uses the edge case CUDA kernel, ignoring the last quarter of rows
TEST(ColourDifference, CIEDE2000_CUDAEdgeCase)
{
    const ColourDifference::Type diffType = ColourDifference::Type::CIEDE2000;
    const size_t size = 1000;
    const size_t fullSize = size * size;
    const size_t totalFloats = fullSize * 3;
    srand(static_cast<unsigned int>(time(NULL)));

    const size_t edgeCaseRows = size / 4;

    //Create random RGB colours
    std::vector<float> h_firsts = TestUtility::createRandomFloats(totalFloats, { {0, 100}, {-128, 127}, {-128, 127} });
    std::vector<float> h_seconds = TestUtility::createRandomFloats(totalFloats, { {0, 100}, {-128, 127}, {-128, 127} });

    //Calculate differences with CPU and CUDA
    std::vector<double> cpuDifferences = calculateCPUDiff(h_firsts, h_seconds, size, diffType, edgeCaseRows);
    std::vector<double> cudaDifferences = calculateCUDADiff(h_firsts, h_seconds, size, diffType, edgeCaseRows);

    //Compare results
    for (size_t i = 0; i < fullSize; ++i)
        ASSERT_NEAR(cudaDifferences.at(i), cpuDifferences.at(i), 0.0001) << "i = " << i;
}
#endif