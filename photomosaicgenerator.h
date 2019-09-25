#ifndef PHOTOMOSAICGENERATOR_H
#define PHOTOMOSAICGENERATOR_H

#include <QProgressDialog>
#include <opencv2/core/mat.hpp>

class PhotomosaicGenerator
{
public:
    static cv::Mat generate(const cv::Mat &mainImage, const std::vector<cv::Mat> &library,
                            QProgressDialog *progress);
    static int findBestImage(const cv::Mat &mainImage, const std::vector<cv::Mat> &library);

private:
    PhotomosaicGenerator() {}
};

#endif // PHOTOMOSAICGENERATOR_H
