#ifndef COLOURDIFFERENCE_H
#define COLOURDIFFERENCE_H

#include <opencv2/core.hpp>

//Used for calculating difference between two colours
namespace ColourDifference
{
//Calculates difference between two RGB (any order) value using RGB Euclidean
double calculateRGBEuclidean(const cv::Vec3b &t_first, const cv::Vec3b &t_second);

//Calculates difference between two CIELAB value using CIE76
double calculateCIE76(const cv::Vec3d &t_first, const cv::Vec3d &t_second);

//Converts degrees to radians
constexpr double degToRad(const double deg);

//Calculates difference between two CIELAB value using CIEDE2000
double calculateCIEDE2000(const cv::Vec3d &t_first, const cv::Vec3d &t_second);
}

#endif // COLOURDIFFERENCE_H
