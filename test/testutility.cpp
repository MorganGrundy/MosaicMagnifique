#include "testutility.h"

//Generates a random float in [min, max]
float TestUtility::randFloat(float min, float max)
{
    return (static_cast<float>(rand()) / static_cast<float>(RAND_MAX)) * (max - min) + min;
}

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
            p_im[col] = (rand() * 255) / RAND_MAX;
        }
    }

    return randIm;
}

//Generates a vector of random floats
std::vector<float> TestUtility::createRandomFloats(const size_t count, const std::vector<std::pair<float, float>> ranges)
{
    std::vector<float> results(count, 0);
    for (size_t i = 0; i < count; ++i)
    {
        const auto &limit = ranges.at(i % ranges.size());
        results.at(i) = randFloat(limit.first, limit.second);
    }

    return results;
}