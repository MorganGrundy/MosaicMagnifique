#include "testutility.h"

#include <QDir>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

//Generates a random image of given size
cv::Mat TestUtility::createRandomImage(const int width, const int height, const bool singleChannel)
{
    cv::Mat randIm(height, width, (singleChannel) ? CV_8UC1 : CV_8UC3);

    uchar *p_im;
    for (int row = 0; row < height; ++row)
    {
        p_im = randIm.ptr<uchar>(row);
        for (int col = 0; col < width * randIm.channels(); ++col)
        {
            p_im[col] = randNum<uchar>(0, 255);
        }
    }

    return randIm;
}

//Generates a vector of random type T
template <typename T>
std::vector<T> TestUtility::createRandom(const size_t count, const std::vector<std::pair<T, T>> ranges)
{
    std::vector<T> results(count, 0);
    for (size_t i = 0; i < count; ++i)
    {
        const std::pair<T,T> &limit = ranges.at(i % ranges.size());
        results.at(i) = randNum<T>(limit.first, limit.second);
    }

    return results;
}

//Explicit instantiation of templates
template std::vector<float> TestUtility::createRandom<float>(const size_t count, const std::vector<std::pair<float, float>> ranges);


//Compares a vector of OpenCV mat against a vector of OpenCV CUDA GpuMat
testing::AssertionResult TestUtility::compareImages(const std::vector<cv::Mat> &set1, const std::vector<cv::cuda::GpuMat> &set2)
{
    //Check both sets have an equal number of images
    if (set1.size() != set2.size())
        return ::testing::AssertionFailure() << "Vector sizes don't match. Set 1 = " << set1.size() << ", set 2 = " << set2.size() << ". ";

    //Download GpuMat to Mat
    std::vector<cv::Mat> downloadedSet2(set2.size());
    for (size_t i = 0; i < set2.size(); ++i)
        set2.at(i).download(downloadedSet2.at(i));

    for (size_t i = 0; i < set1.size(); ++i)
    {
        //Check image size matches
        if (set1.at(i).size != downloadedSet2.at(i).size)
            return ::testing::AssertionFailure() << "Size of images at index " << i << " don't match. ";

        //return ASSERT_MAT_NEAR(set1.at(i), downloadedSet2.at(i), set1.at(i).depth() == CV_32F ? 1e-2 : 1.0);
        //Check image contents match
        const auto totalDiff = cv::sum(set1.at(i) != downloadedSet2.at(i))[0];
        if (totalDiff != 0)
        {
            return ::testing::AssertionFailure() << "Contents of images at index " << i << " don't match. " << totalDiff << " differences. ";
        }
    }

    return ::testing::AssertionSuccess();
}