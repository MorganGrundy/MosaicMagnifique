#include "ColourScheme.h"
#include <stdexcept>
#include <opencv2/imgproc.hpp>

#include "MatUtility.h"

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
    complement.forEach<cv::Point3f>([](cv::Point3f &p, [[maybe_unused]] const int *position) { p.x = fmod(p.x + 180.0f, 360.0f); });

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
    //Need to convert image to float and use HSV_FULL (not HSV) to prevent precision loss
    cv::Mat_<cv::Point3f> triadic1(t_image), triadic2(t_image.rows, t_image.cols);
    cv::cvtColor(triadic1, triadic1, cv::COLOR_BGR2HSV_FULL);

    //Calculate hue rotations
    MatUtility::forEach_2_impl<cv::Point3f>(const_cast<cv::Mat_<cv::Point3f> *>(&triadic1), const_cast<cv::Mat_<cv::Point3f> *>(&triadic2), [](cv::Point3f &p1, cv::Point3f &p2, [[maybe_unused]] const int *position)
        {
            p2.x = fmod(p1.x + 240.0f, 360.0f); p2.y = p1.y; p2.z = p1.z; //Rotate hue 240°
            p1.x = fmod(p1.x + 120.0f, 360.0f); //Rotate hue 120°
        });

    //Convert back to BGR and original depth
    cv::Mat triadicResult1, triadicResult2;
    cv::cvtColor(triadic1, triadic1, cv::COLOR_HSV2BGR_FULL);
    triadic1.convertTo(triadicResult1, CV_MAT_DEPTH(t_image.type()));
    cv::cvtColor(triadic2, triadic2, cv::COLOR_HSV2BGR_FULL);
    triadic2.convertTo(triadicResult2, CV_MAT_DEPTH(t_image.type()));

    return { t_image, triadicResult1, triadicResult2 };
}

//Returns image variants for colour scheme "Compound"
//Original image
//Hue rotated 150°; H = (H + 150°) mod 360°
//Hue rotated 210°; H = (H + 210°) mod 360°
std::vector<cv::Mat> ColourScheme::getColourSchemeCompound(const cv::Mat& t_image)
{
    //Need to convert image to float and use HSV_FULL (not HSV) to prevent precision loss
    cv::Mat_<cv::Point3f> compound1(t_image), compound2(t_image.rows, t_image.cols);
    cv::cvtColor(compound1, compound1, cv::COLOR_BGR2HSV_FULL);

    //Calculate hue rotations
    MatUtility::forEach_2_impl<cv::Point3f>(const_cast<cv::Mat_<cv::Point3f> *>(&compound1), const_cast<cv::Mat_<cv::Point3f> *>(&compound2), [](cv::Point3f &p1, cv::Point3f &p2, [[maybe_unused]] const int *position)
        {
            p2.x = fmod(p1.x + 210.0f, 360.0f); p2.y = p1.y; p2.z = p1.z; //Rotate hue 210°
            p1.x = fmod(p1.x + 150.0f, 360.0f); //Rotate hue 150°
        });

    //Convert back to BGR and original depth
    cv::Mat compoundResult1, compoundResult2;
    cv::cvtColor(compound1, compound1, cv::COLOR_HSV2BGR_FULL);
    compound1.convertTo(compoundResult1, CV_MAT_DEPTH(t_image.type()));
    cv::cvtColor(compound2, compound2, cv::COLOR_HSV2BGR_FULL);
    compound2.convertTo(compoundResult2, CV_MAT_DEPTH(t_image.type()));

    return { t_image, compoundResult1, compoundResult2 };
}

//Returns image variants for colour scheme "Tetradic"
//Original image
//Hue rotated 90°; H = (H + 90°) mod 360°
//Hue rotated 180°; H = (H + 180°) mod 360°
//Hue rotated 270°; H = (H + 270°) mod 360°
std::vector<cv::Mat> ColourScheme::getColourSchemeTetradic(const cv::Mat& t_image)
{
    //Need to convert image to float and use HSV_FULL (not HSV) to prevent precision loss
    cv::Mat_<cv::Point3f> tetradic1(t_image), tetradic2(t_image.rows, t_image.cols), tetradic3(t_image.rows, t_image.cols);
    cv::cvtColor(tetradic1, tetradic1, cv::COLOR_BGR2HSV_FULL);

    //Calculate hue rotations
    MatUtility::forEach_3_impl<cv::Point3f>(const_cast<cv::Mat_<cv::Point3f> *>(&tetradic1), const_cast<cv::Mat_<cv::Point3f> *>(&tetradic2), const_cast<cv::Mat_<cv::Point3f> *>(&tetradic3),
        [](cv::Point3f &p1, cv::Point3f &p2, cv::Point3f &p3, [[maybe_unused]] const int *position)
        {
            p3.x = fmod(p1.x + 270.0f, 360.0f); p3.y = p1.y; p3.z = p1.z; //Rotate hue 270°
            p2.x = fmod(p1.x + 180.0f, 360.0f); p2.y = p1.y; p2.z = p1.z; //Rotate hue 180°
            p1.x = fmod(p1.x + 90.0f, 360.0f); //Rotate hue 90°
        });

    //Convert back to BGR and original depth
    cv::Mat tetradicResult1, tetradicResult2, tetradicResult3;
    cv::cvtColor(tetradic1, tetradic1, cv::COLOR_HSV2BGR_FULL);
    tetradic1.convertTo(tetradicResult1, CV_MAT_DEPTH(t_image.type()));
    cv::cvtColor(tetradic2, tetradic2, cv::COLOR_HSV2BGR_FULL);
    tetradic2.convertTo(tetradicResult2, CV_MAT_DEPTH(t_image.type()));
    cv::cvtColor(tetradic3, tetradic3, cv::COLOR_HSV2BGR_FULL);
    tetradic3.convertTo(tetradicResult3, CV_MAT_DEPTH(t_image.type()));

    return { t_image, tetradicResult1, tetradicResult2, tetradicResult3 };
}

//Returns image variants for colour scheme "Analagous"
//Original image
//Hue rotated 30°; H = (H + 30°) mod 360°
//Hue rotated 60°; H = (H + 60°) mod 360°
//Hue rotated 90°; H = (H + 90°) mod 360°
std::vector<cv::Mat> ColourScheme::getColourSchemeAnalagous(const cv::Mat& t_image)
{
    //Need to convert image to float and use HSV_FULL (not HSV) to prevent precision loss
    cv::Mat_<cv::Point3f> analagous1(t_image), analagous2(t_image.rows, t_image.cols), analagous3(t_image.rows, t_image.cols);
    cv::cvtColor(analagous1, analagous1, cv::COLOR_BGR2HSV_FULL);

    //Calculate hue rotations
    MatUtility::forEach_3_impl<cv::Point3f>(const_cast<cv::Mat_<cv::Point3f> *>(&analagous1), const_cast<cv::Mat_<cv::Point3f> *>(&analagous2), const_cast<cv::Mat_<cv::Point3f> *>(&analagous3),
        [](cv::Point3f &p1, cv::Point3f &p2, cv::Point3f &p3, [[maybe_unused]] const int *position)
        {
            p3.x = fmod(p1.x + 90.0f, 360.0f); p3.y = p1.y; p3.z = p1.z; //Rotate hue 90°
            p2.x = fmod(p1.x + 60.0f, 360.0f); p2.y = p1.y; p2.z = p1.z; //Rotate hue 60°
            p1.x = fmod(p1.x + 30.0f, 360.0f); //Rotate hue 30°
        });

    //Convert back to BGR and original depth
    cv::Mat analagousResult1, analagousResult2, analagousResult3;
    cv::cvtColor(analagous1, analagous1, cv::COLOR_HSV2BGR_FULL);
    analagous1.convertTo(analagousResult1, CV_MAT_DEPTH(t_image.type()));
    cv::cvtColor(analagous2, analagous2, cv::COLOR_HSV2BGR_FULL);
    analagous2.convertTo(analagousResult2, CV_MAT_DEPTH(t_image.type()));
    cv::cvtColor(analagous3, analagous3, cv::COLOR_HSV2BGR_FULL);
    analagous3.convertTo(analagousResult3, CV_MAT_DEPTH(t_image.type()));

    return { t_image, analagousResult1, analagousResult2, analagousResult3 };
}
