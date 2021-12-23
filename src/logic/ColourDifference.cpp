#include "ColourDifference.h"

//Converts type string to type enum
ColourDifference::Type ColourDifference::strToEnum(const QString& t_type)
{
    for (size_t i = 0; i < static_cast<size_t>(Type::MAX); ++i)
    {
        if (Type_STR.at(i).compare(t_type) == 0)
            return static_cast<Type>(i);
    }

    return Type::MAX;
}

//Returns function wrapper from enum
ColourDifference::FunctionType ColourDifference::getFunction(const Type& t_type)
{
    switch (t_type)
    {
    case Type::RGB_EUCLIDEAN: return calculateRGBEuclidean;
    case Type::CIE76: return calculateRGBEuclidean;
    case Type::CIEDE2000: return calculateCIEDE2000;
    default: throw std::invalid_argument(Q_FUNC_INFO " No function for given type");
    }
}

//Calculates difference between two colours using Euclidean distance
double ColourDifference::calculateRGBEuclidean(const cv::Vec3d &t_first, const cv::Vec3d &t_second)
{
    return sqrt(pow(t_first[0] - t_second[0], 2) +
                pow(t_first[1] - t_second[1], 2) +
                pow(t_first[2] - t_second[2], 2));
}

//Converts degrees to radians
constexpr double ColourDifference::degToRad(const double deg)
{
    return (deg * M_PI) / 180;
}

//Calculates difference between two CIELAB value using CIEDE2000
double ColourDifference::calculateCIEDE2000(const cv::Vec3d &t_first, const cv::Vec3d &t_second)
{
    const double k_L = 1.0, k_C = 1.0, k_H = 1.0;
    constexpr double deg360InRad = degToRad(360.0);
    constexpr double deg180InRad = degToRad(180.0);
    const double pow25To7 = 6103515625.0; //pow(25, 7)

    const double C1 = sqrt((t_first[1] * t_first[1]) +
                           (t_first[2] * t_first[2]));
    const double C2 = sqrt((t_second[1] * t_second[1]) +
                           (t_second[2] * t_second[2]));
    const double barC = (C1 + C2) / 2.0;

    const double G = 0.5 * (1 - sqrt(pow(barC, 7) / (pow(barC, 7) + pow25To7)));

    const double a1Prime = (1.0 + G) * t_first[1];
    const double a2Prime = (1.0 + G) * t_second[1];

    const double CPrime1 = sqrt((a1Prime * a1Prime) +
                                (t_first[2] * t_first[2]));
    const double CPrime2 = sqrt((a2Prime * a2Prime) +(t_second[2] * t_second[2]));

    double hPrime1;
    if (t_first[2] == 0 && a1Prime == 0.0)
        hPrime1 = 0.0;
    else
    {
        hPrime1 = atan2(t_first[2], a1Prime);
        //This must be converted to a hue angle in degrees between 0 and 360 by
        //addition of 2 pi to negative hue angles.
        if (hPrime1 < 0)
            hPrime1 += deg360InRad;
    }

    double hPrime2;
    if (t_second[2] == 0 && a2Prime == 0.0)
        hPrime2 = 0.0;
    else
    {
        hPrime2 = atan2(t_second[2], a2Prime);
        //This must be converted to a hue angle in degrees between 0 and 360 by
        //addition of 2pi to negative hue angles.
        if (hPrime2 < 0)
            hPrime2 += deg360InRad;
    }

    const double deltaLPrime = t_second[0] - t_first[0];
    const double deltaCPrime = CPrime2 - CPrime1;

    double deltahPrime;
    const double CPrimeProduct = CPrime1 * CPrime2;
    if (CPrimeProduct == 0.0)
        deltahPrime = 0;
    else
    {
        //Avoid the fabs() call
        deltahPrime = hPrime2 - hPrime1;
        if (deltahPrime < -deg180InRad)
            deltahPrime += deg360InRad;
        else if (deltahPrime > deg180InRad)
            deltahPrime -= deg360InRad;
    }

    const double deltaHPrime = 2.0 * sqrt(CPrimeProduct) * sin(deltahPrime / 2.0);

    const double barLPrime = (t_first[0] + t_second[0]) / 2.0;
    const double barCPrime = (CPrime1 + CPrime2) / 2.0;

    double barhPrime;
    const double hPrimeSum = hPrime1 + hPrime2;
    if (CPrime1 * CPrime2 == 0.0)
        barhPrime = hPrimeSum;
    else
    {
        if (fabs(hPrime1 - hPrime2) <= deg180InRad)
            barhPrime = hPrimeSum / 2.0;
        else
        {
            if (hPrimeSum < deg360InRad)
                barhPrime = (hPrimeSum + deg360InRad) / 2.0;
            else
                barhPrime = (hPrimeSum - deg360InRad) / 2.0;
        }
    }

    const double T = 1.0 - (0.17 * cos(barhPrime - degToRad(30.0))) +
                     (0.24 * cos(2.0 * barhPrime)) +
                     (0.32 * cos((3.0 * barhPrime) + degToRad(6.0))) -
                     (0.20 * cos((4.0 * barhPrime) - degToRad(63.0)));

    const double deltaTheta = degToRad(30.0) *
                              exp(-pow((barhPrime - degToRad(275.0)) / degToRad(25.0), 2.0));

    const double R_C = 2.0 * sqrt(pow(barCPrime, 7.0) /
                                  (pow(barCPrime, 7.0) + pow25To7));

    const double S_L = 1 + ((0.015 * pow(barLPrime - 50.0, 2.0)) /
                            sqrt(20 + pow(barLPrime - 50.0, 2.0)));
    const double S_C = 1 + (0.045 * barCPrime);
    const double S_H = 1 + (0.015 * barCPrime * T);

    const double R_T = (-sin(2.0 * deltaTheta)) * R_C;


    return sqrt(pow(deltaLPrime / (k_L * S_L), 2.0) +
                pow(deltaCPrime / (k_C * S_C), 2.0) +
                pow(deltaHPrime / (k_H * S_H), 2.0) +
                (R_T * (deltaCPrime / (k_C * S_C)) * (deltaHPrime / (k_H * S_H))));
}

#ifdef CUDA
#include "photomosaicgenerator.cuh"

//Returns CUDA function wrapper from enum
ColourDifference::CUDAFunctionType ColourDifference::getCUDAFunction(const Type& t_type)
{
    switch (t_type)
    {
    case Type::RGB_EUCLIDEAN: return euclideanDifferenceKernelWrapper;
    case Type::CIE76: return euclideanDifferenceKernelWrapper;
    case Type::CIEDE2000: return CIEDE2000DifferenceKernelWrapper;
    default: throw std::invalid_argument(Q_FUNC_INFO " No function for given type");
    }
}
#endif