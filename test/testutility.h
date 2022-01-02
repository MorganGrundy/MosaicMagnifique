#pragma once

#include <opencv2/core.hpp>
#include <QString>

namespace TestUtility
{
    static const QString LIB_FILE = "../Library/lib.mil";
    static const QString BIG_LIB_FILE = "../Library/big-lib.mil";

    static const std::string EDGAR_PEREZ = "../SampleImages/edgar-perez-424673-unsplash.jpg";

    //Generates a random float in [min, max]
    float randFloat(float min, float max);

    //Generates a random image of given size
    cv::Mat createRandomImage(const int width, const int height, const bool singleChannel = false);
};
