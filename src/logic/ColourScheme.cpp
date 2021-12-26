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
    case Type::TRIADIC: return getColourSchemeTriadic;
    case Type::COMPOUND: return getColourSchemeCompound;
    case Type::TETRADIC: return getColourSchemeTetradic;
    case Type::ANALAGOUS: return getColourSchemeAnalagous;
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
    cv::Mat complement;
    //Need to convert image to float and use HSV_FULL (not HSV) to prevent precision loss
    t_image.convertTo(complement, CV_32F);
    cv::cvtColor(complement, complement, cv::COLOR_BGR2HSV_FULL);

    //Rotate hue 180°
    complement.forEach<cv::Point3_<float>>([](cv::Point3_<float>&p, [[maybe_unused]] const int *position) { p.x = fmod(p.x + 180.0f, 360.0f); });

    //Convert back to BGR and original depth
    cv::cvtColor(complement, complement, cv::COLOR_HSV2BGR_FULL);
    complement.convertTo(complement, CV_MAT_DEPTH(t_image.type()));

    return { t_image, complement };
}

//Returns image variants for colour scheme "Triadic"
//Original image
//Hue rotated 120°; H = (H + 120°) mod 360°
//Hue rotated 240°; H = (H + 240°) mod 360°
std::vector<cv::Mat> ColourScheme::getColourSchemeTriadic(const cv::Mat& t_image)
{
    cv::Mat triadic1, triadic2;
    //Need to convert image to float and use HSV_FULL (not HSV) to prevent precision loss
    t_image.convertTo(triadic1, CV_32F);
    cv::cvtColor(triadic1, triadic1, cv::COLOR_BGR2HSV_FULL);
    triadic2 = triadic1.clone();

    //Rotate hue 120°
    triadic1.forEach<cv::Point3_<float>>([](cv::Point3_<float>& p, [[maybe_unused]] const int* position) { p.x = fmod(p.x + 120.0f, 360.0f); });
    //Rotate hue 240°
    triadic2.forEach<cv::Point3_<float>>([](cv::Point3_<float>& p, [[maybe_unused]] const int* position) { p.x = fmod(p.x + 240.0f, 360.0f); });

    //Convert back to BGR and original depth
    cv::cvtColor(triadic1, triadic1, cv::COLOR_HSV2BGR_FULL);
    triadic1.convertTo(triadic1, CV_MAT_DEPTH(t_image.type()));
    cv::cvtColor(triadic2, triadic2, cv::COLOR_HSV2BGR_FULL);
    triadic2.convertTo(triadic2, CV_MAT_DEPTH(t_image.type()));

    return { t_image, triadic1, triadic2 };
}

//Returns image variants for colour scheme "Compound"
//Original image
//Hue rotated 150°; H = (H + 150°) mod 360°
//Hue rotated 210°; H = (H + 210°) mod 360°
std::vector<cv::Mat> ColourScheme::getColourSchemeCompound(const cv::Mat& t_image)
{
    cv::Mat compound1, compound2;
    //Need to convert image to float and use HSV_FULL (not HSV) to prevent precision loss
    t_image.convertTo(compound1, CV_32F);
    cv::cvtColor(compound1, compound1, cv::COLOR_BGR2HSV_FULL);
    compound2 = compound1.clone();

    //Rotate hue 120°
    compound1.forEach<cv::Point3_<float>>([](cv::Point3_<float>& p, [[maybe_unused]] const int* position) { p.x = fmod(p.x + 120.0f, 360.0f); });
    //Rotate hue 240°
    compound2.forEach<cv::Point3_<float>>([](cv::Point3_<float>& p, [[maybe_unused]] const int* position) { p.x = fmod(p.x + 240.0f, 360.0f); });

    //Convert back to BGR and original depth
    cv::cvtColor(compound1, compound1, cv::COLOR_HSV2BGR_FULL);
    compound1.convertTo(compound1, CV_MAT_DEPTH(t_image.type()));
    cv::cvtColor(compound2, compound2, cv::COLOR_HSV2BGR_FULL);
    compound2.convertTo(compound2, CV_MAT_DEPTH(t_image.type()));

    return { t_image, compound1, compound2 };
}

//Returns image variants for colour scheme "Tetradic"
//Original image
//Hue rotated 90°; H = (H + 90°) mod 360°
//Hue rotated 180°; H = (H + 180°) mod 360°
//Hue rotated 270°; H = (H + 270°) mod 360°
std::vector<cv::Mat> ColourScheme::getColourSchemeTetradic(const cv::Mat& t_image)
{
    cv::Mat tetradic1, tetradic2, tetradic3;
    //Need to convert image to float and use HSV_FULL (not HSV) to prevent precision loss
    t_image.convertTo(tetradic1, CV_32F);
    cv::cvtColor(tetradic1, tetradic1, cv::COLOR_BGR2HSV_FULL);
    tetradic2 = tetradic1.clone();
    tetradic3 = tetradic1.clone();

    //Rotate hue 90°
    tetradic1.forEach<cv::Point3_<float>>([](cv::Point3_<float>& p, [[maybe_unused]] const int* position) { p.x = fmod(p.x + 90.0f, 360.0f); });
    //Rotate hue 180°
    tetradic2.forEach<cv::Point3_<float>>([](cv::Point3_<float>& p, [[maybe_unused]] const int* position) { p.x = fmod(p.x + 180.0f, 360.0f); });
    //Rotate hue 270°
    tetradic3.forEach<cv::Point3_<float>>([](cv::Point3_<float>& p, [[maybe_unused]] const int* position) { p.x = fmod(p.x + 270.0f, 360.0f); });

    //Convert back to BGR and original depth
    cv::cvtColor(tetradic1, tetradic1, cv::COLOR_HSV2BGR_FULL);
    tetradic1.convertTo(tetradic1, CV_MAT_DEPTH(t_image.type()));
    cv::cvtColor(tetradic2, tetradic2, cv::COLOR_HSV2BGR_FULL);
    tetradic2.convertTo(tetradic2, CV_MAT_DEPTH(t_image.type()));
    cv::cvtColor(tetradic3, tetradic3, cv::COLOR_HSV2BGR_FULL);
    tetradic3.convertTo(tetradic3, CV_MAT_DEPTH(t_image.type()));

    return { t_image, tetradic1, tetradic2, tetradic3 };
}

//Returns image variants for colour scheme "Analagous"
//Original image
//Hue rotated 30°; H = (H + 30°) mod 360°
//Hue rotated 60°; H = (H + 60°) mod 360°
//Hue rotated 90°; H = (H + 90°) mod 360°
std::vector<cv::Mat> ColourScheme::getColourSchemeAnalagous(const cv::Mat& t_image)
{
    cv::Mat analagous1, analagous2, analagous3;
    //Need to convert image to float and use HSV_FULL (not HSV) to prevent precision loss
    t_image.convertTo(analagous1, CV_32F);
    cv::cvtColor(analagous1, analagous1, cv::COLOR_BGR2HSV_FULL);
    analagous2 = analagous1.clone();
    analagous3 = analagous1.clone();

    //Rotate hue 30°
    analagous1.forEach<cv::Point3_<float>>([](cv::Point3_<float>& p, [[maybe_unused]] const int* position) { p.x = fmod(p.x + 30.0f, 360.0f); });
    //Rotate hue 60°
    analagous2.forEach<cv::Point3_<float>>([](cv::Point3_<float>& p, [[maybe_unused]] const int* position) { p.x = fmod(p.x + 60.0f, 360.0f); });
    //Rotate hue 90°
    analagous3.forEach<cv::Point3_<float>>([](cv::Point3_<float>& p, [[maybe_unused]] const int* position) { p.x = fmod(p.x + 90.0f, 360.0f); });

    //Convert back to BGR and original depth
    cv::cvtColor(analagous1, analagous1, cv::COLOR_HSV2BGR_FULL);
    analagous1.convertTo(analagous1, CV_MAT_DEPTH(t_image.type()));
    cv::cvtColor(analagous2, analagous2, cv::COLOR_HSV2BGR_FULL);
    analagous2.convertTo(analagous2, CV_MAT_DEPTH(t_image.type()));
    cv::cvtColor(analagous3, analagous3, cv::COLOR_HSV2BGR_FULL);
    analagous3.convertTo(analagous3, CV_MAT_DEPTH(t_image.type()));

    return { t_image, analagous1, analagous2, analagous3 };
}
