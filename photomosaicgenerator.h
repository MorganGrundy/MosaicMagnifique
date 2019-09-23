#ifndef PHOTOMOSAICGENERATOR_H
#define PHOTOMOSAICGENERATOR_H

#include <opencv2/core/mat.hpp>

class PhotomosaicGenerator
{
public:
    static cv::Mat generate(const cv::Mat &mainImage, const std::vector<cv::Mat> &library);
    static int findBestImage(const cv::Mat &mainImage, const std::vector<cv::Mat> &library);

private:
    PhotomosaicGenerator() {}
};

#endif // PHOTOMOSAICGENERATOR_H
