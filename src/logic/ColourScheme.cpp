#include "ColourScheme.h"
#include <stdexcept>
#include <opencv2/imgproc.hpp>

//Converts type string to type enum
ColourScheme::Type ColourScheme::strToEnum(const QString& t_type)
{
    for (size_t i = 0; i < static_cast<size_t>(Type::MAX); ++i)
    {
        if (Type_STR.at(i).compare(t_type) == 0)
            return static_cast<Type>(i);
    }

    return Type::MAX;
}

//Returns function wrapper from enum
ColourScheme::FunctionType ColourScheme::getFunction(const Type& t_type)
{
    switch (t_type)
    {
    case Type::NONE: return getColourSchemeNone;
    case Type::COMPLEMENTARY: return getColourSchemeComplementary;
    default: throw std::invalid_argument(Q_FUNC_INFO " No function for given type");
    }
}

//Returns image variants for colour scheme "none"
//Original image
std::vector<cv::Mat> ColourScheme::getColourSchemeNone(const cv::Mat& t_image)
{
    return { t_image };
}

//Returns image variants for colour scheme "complementary"
//Original image
//Hue rotated 180°; H = (H + 180°) mod 360°
std::vector<cv::Mat> ColourScheme::getColourSchemeComplementary(const cv::Mat& t_image)
{
    cv::Mat hsvImage;
    //Need to convert image to float and use HSV_FULL (not HSV) to prevent precision loss
    t_image.convertTo(hsvImage, CV_32F);
    cv::cvtColor(hsvImage, hsvImage, cv::COLOR_BGR2HSV_FULL);

    //Rotate hue 180°
    hsvImage.forEach<cv::Point3_<float>>([](cv::Point3_<float>&p, const int *position) { p.x = fmod(p.x + 180.0f, 360.0f); });

    //Convert back to BGR and original depth
    cv::cvtColor(hsvImage, hsvImage, cv::COLOR_HSV2BGR_FULL);
    hsvImage.convertTo(hsvImage, CV_MAT_DEPTH(t_image.type()));

    return { t_image, hsvImage };
}
