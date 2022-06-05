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
        TETRADIC = 4,
        ANALAGOUS = 5,
        MAX = 6
    };

    static const std::vector<QString> Type_STR = {
        "None",
        "Complementary",
        "Triadic",
        "Compound",
        "Tetradic",
        "Analagous"
    };

    //Converts type string to type enum
    Type strToEnum(const QString& t_type);

    //Alias for function wrapper
    using FunctionType = std::function<std::vector<cv::Mat>(const cv::Mat &)>;

    //Returns function wrapper from enum
    FunctionType getFunction(const Type& t_type);

    //Returns image variants for colour scheme "None"
    //Original image
    std::vector<cv::Mat> getColourSchemeNone(const cv::Mat &t_image);

    //Returns image variants for colour scheme "Complementary"
    //Original image
    //Hue rotated 180°; H = (H + 180°) mod 360°
    std::vector<cv::Mat> getColourSchemeComplementary(const cv::Mat& t_image);

    //Returns image variants for colour scheme "Triadic"
    //Original image
    //Hue rotated 120°; H = (H + 120°) mod 360°
    //Hue rotated 240°; H = (H + 240°) mod 360°
    std::vector<cv::Mat> getColourSchemeTriadic(const cv::Mat& t_image);

    //Returns image variants for colour scheme "Compound"
    //Original image
    //Hue rotated 150°; H = (H + 150°) mod 360°
    //Hue rotated 210°; H = (H + 210°) mod 360°
    std::vector<cv::Mat> getColourSchemeCompound(const cv::Mat& t_image);

    //Returns image variants for colour scheme "Tetradic"
    //Original image
    //Hue rotated 90°; H = (H + 90°) mod 360°
    //Hue rotated 180°; H = (H + 180°) mod 360°
    //Hue rotated 270°; H = (H + 270°) mod 360°
    std::vector<cv::Mat> getColourSchemeTetradic(const cv::Mat& t_image);

    //Returns image variants for colour scheme "Analagous"
    //Original image
    //Hue rotated 30°; H = (H + 30°) mod 360°
    //Hue rotated 60°; H = (H + 60°) mod 360°
    //Hue rotated 90°; H = (H + 90°) mod 360°
    std::vector<cv::Mat> getColourSchemeAnalagous(const cv::Mat& t_image);
};

