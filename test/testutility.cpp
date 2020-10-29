#include "testutility.h"

//Generates a random float in [min, max]
float TestUtility::randFloat(float min, float max)
{
    return (static_cast<float>(rand()) / static_cast<float>(RAND_MAX)) * (max - min) + min;
}

//Generates a random image of given size and colour space
cv::Mat TestUtility::createRandomImage(const int width, const int height,
                                       const ColourSpace colourSpace)
{
    cv::Mat randIm(height, width, CV_32FC3);

    cv::Vec3f *p_im;
    for (int row = 0; row < height; ++row)
    {
        p_im = randIm.ptr<cv::Vec3f>(row);
        for (int col = 0; col < width; ++col)
        {
            if (colourSpace == ColourSpace::BGR)
            {
                p_im[col][0] = static_cast<float>((rand() * 100) / RAND_MAX);
                p_im[col][1] = static_cast<float>((rand() * 100) / RAND_MAX);
                p_im[col][2] = static_cast<float>((rand() * 100) / RAND_MAX);
            }
            else if (colourSpace == ColourSpace::CIELAB)
            {
                p_im[col][0] = randFloat(0, 100);
                p_im[col][1] = randFloat(-128, 127);
                p_im[col][2] = randFloat(-128, 127);
            }
        }
    }

    return randIm;
}
