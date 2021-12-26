#pragma once

#include <QString>
#include <vector>
#include <functional>
#include <opencv2/core/mat.hpp>

namespace ColourScheme
{
    enum class Type
    {
        NONE = 0,
        COMPLEMENTARY = 1,
        TRIADIC = 2,
        COMPOUND = 3,
        MAX = 4
    };

    static const std::vector<QString> Type_STR = {
        "None",
        "Complementary",
        "Triadic",
        "Compound"
    };

    //Converts type string to type enum
    Type strToEnum(const QString& t_type);

    //Alias for function wrapper
    using FunctionType = std::function<std::vector<cv::Mat>(const cv::Mat &)>;

    //Returns function wrapper from enum
    FunctionType getFunction(const Type& t_type);

    //Returns image variants for colour scheme "none"
    //Original image
    std::vector<cv::Mat> getColourSchemeNone(const cv::Mat &t_image);

    //Returns image variants for colour scheme "complementary"
    //Original image
    //Hue rotated 180°; H = (H + 180°) mod 360°
    std::vector<cv::Mat> getColourSchemeComplementary(const cv::Mat& t_image);
};

