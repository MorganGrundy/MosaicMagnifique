#ifndef COLOURDIFFERENCE_H
#define COLOURDIFFERENCE_H

#include <opencv2/core.hpp>
#include <QString>

//Used for calculating difference between two colours
namespace ColourDifference
{
	enum class Type
	{
		RGB_EUCLIDEAN = 0,
		CIE76 = 1,
		CIEDE2000 = 2,
		MAX = 3
	};

	static const std::vector<QString> Type_STR = {
		"RGB Euclidean",
		"CIE76",
		"CIEDE2000"
	};

	//Converts type string to type enum
	Type strToEnum(const QString& t_type);

	//Alias for function wrapper
	using FunctionType = std::function<double(const cv::Vec3d&, const cv::Vec3d&)>;

	//Returns function wrapper from enum
	FunctionType getFunction(const Type &t_type);

	//Calculates difference between two colours using Euclidean distance
	double calculateRGBEuclidean(const cv::Vec3d &t_first, const cv::Vec3d &t_second);

	//Calculates difference between two CIELAB value using CIE76
	//Just an alias for RGB Euclidean, only difference is values in CIELAB colour space
	constexpr auto calculateCIE76 = calculateRGBEuclidean;

	//Converts degrees to radians
	constexpr double degToRad(const double deg);

	//Calculates difference between two CIELAB value using CIEDE2000
	double calculateCIEDE2000(const cv::Vec3d &t_first, const cv::Vec3d &t_second);

#ifdef CUDA
	//Alias for function wrapper
	using CUDAFunctionType = std::function<void(float*, float*, size_t, unsigned char*, size_t, size_t, size_t*, double*, size_t)>;

	//Returns CUDA function wrapper from enum
	CUDAFunctionType getCUDAFunction(const Type& t_type);
#endif
}

#endif // COLOURDIFFERENCE_H
