#ifndef PHOTOMOSAICGENERATOR_H
#define PHOTOMOSAICGENERATOR_H

#include <QProgressDialog>
#include <opencv2/core/mat.hpp>

class PhotomosaicGenerator
{
public:
    enum class Mode {RGB_EUCLIDEAN, CIE76, CIEDE2000};

    static cv::Mat generate(cv::Mat &mainImage, const std::vector<cv::Mat> &library,
                            Mode mode, QProgressDialog &progress);

    static int findBestFitEuclidean(const cv::Mat &cell, const std::vector<cv::Mat> &library);
    static int findBestFitCIEDE2000(const cv::Mat &cell, const std::vector<cv::Mat> &library);

    static double degToRad(const double deg);

private:
    PhotomosaicGenerator() {}
};

#endif // PHOTOMOSAICGENERATOR_H
