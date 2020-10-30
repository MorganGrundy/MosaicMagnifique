#ifndef TESTUTILITY_H
#define TESTUTILITY_H

#include <opencv2/core.hpp>

namespace TestUtility
{
//Generates a random float in [min, max]
float randFloat(float min, float max);

//Generates a random image of given size
cv::Mat createRandomImage(const int width, const int height, const bool singleChannel = false);
};

#endif // TESTUTILITY_H
