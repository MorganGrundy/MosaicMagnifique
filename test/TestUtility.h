#pragma once

#include <gtest/gtest.h>
#include <opencv2/core.hpp>
#include <opencv2/cudaarithm.hpp>
#include <QString>

namespace TestUtility
{
    static const QString LIB_FILE = "../Library/lib.mil";
    static const QString BIG_LIB_FILE = "../Library/big-lib.mil";

    static const std::string EDGAR_PEREZ = "../SampleImages/edgar-perez-424673-unsplash.jpg";

    //Generates a random of T in [min, max]
    template<typename T>
    T randNum(T min, T max)
    {
        return ((rand() * (max - min)) / static_cast<T>(RAND_MAX)) + min;
    }

    //Generates a random image of given size
    cv::Mat createRandomImage(const int width, const int height, const bool singleChannel = false);

    //Generates a vector of random type T
    template<typename T>
    std::vector<T> createRandom(const size_t count, const std::vector<std::pair<T, T>> ranges);

    //Compares a vector of OpenCV mat against a vector of OpenCV CUDA GpuMat
    ::testing::AssertionResult compareImages(const std::vector<cv::Mat> &set1, const std::vector<cv::cuda::GpuMat> &set2);
};