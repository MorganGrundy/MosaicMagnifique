#ifndef TESTUTILITY_H
#define TESTUTILITY_H

#include <opencv2/core.hpp>

namespace TestUtility
{
//Generates a random float in [min, max]
float randFloat(float min, float max);

enum class ColourSpace {BGR, CIELAB};

//Generates a random image of given size and colour space
cv::Mat createRandomImage(const int width, const int height, const ColourSpace colourSpace);
};

#endif // TESTUTILITY_H
