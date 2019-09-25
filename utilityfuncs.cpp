#ifndef SHARED_CPP_
#define SHARED_CPP_

#include "utilityfuncs.h"

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <cstdio>
#include <boost/filesystem.hpp>
#include <string>
#include <cmath>

#ifdef OPENCV_WITH_CUDA
#include <opencv2/cudawarping.hpp>
#endif

//Returns a resized copy of the given image such that
//type = INCLUSIVE:
//(height = targetHeight && width <= targetWidth) || (height <= targetHeight && width = targetWidth)
//type = EXCLUSIVE:
//(height = targetHeight && width >= targetWidth) || (height >= targetHeight && width = targetWidth)
cv::Mat UtilityFuncs::resizeImage(const cv::Mat &t_img,
                                  const int t_targetHeight, const int t_targetWidth,
                                  const ResizeType t_type)
{
    //Calculates resize factor
    double resizeFactor = static_cast<double>(t_targetHeight) / t_img.rows;
    if ((t_type == ResizeType::EXCLUSIVE && t_targetWidth < resizeFactor * t_img.cols) ||
            (t_type == ResizeType::INCLUSIVE && t_targetWidth > resizeFactor * t_img.cols))
        resizeFactor = static_cast<double>(t_targetWidth) / t_img.cols;

    if (resizeFactor == 1.0)
        return t_img;

    //Use INTER_AREA for decreasing, INTER_CUBIC for increasing
    cv::InterpolationFlags flags = (resizeFactor < 1) ? cv::INTER_AREA : cv::INTER_CUBIC;

    //Resizes image
    cv::Mat result;
    cv::resize(t_img, result, cv::Size(static_cast<int>(resizeFactor * t_img.cols),
                                       static_cast<int>(resizeFactor * t_img.rows)), 0, 0, flags);
    return result;
}

//Ensures image rows == cols, result image focus at centre of original
void UtilityFuncs::imageToSquare(cv::Mat& t_img)
{
    if (t_img.cols < t_img.rows)
    {
        int diff = (t_img.rows - t_img.cols)/2;
        t_img = t_img(cv::Range(diff, t_img.cols + diff), cv::Range(0, t_img.cols));
    }
    else if (t_img.cols > t_img.rows)
    {
        int diff = (t_img.cols - t_img.rows)/2;
        t_img = t_img(cv::Range(0, t_img.rows), cv::Range(diff, t_img.rows + diff));
    }
}

//Converts an OpenCV Mat to a QPixmap and returns
QPixmap UtilityFuncs::matToQPixmap(const cv::Mat &t_mat)
{
    return QPixmap::fromImage(QImage(t_mat.data, t_mat.cols, t_mat.rows,
                                     static_cast<int>(t_mat.step),
                                     QImage::Format_RGB888).rgbSwapped());
}

//Returns copy of all mats resized to the target height
//Assumes all mat are same size
//If OpenCV CUDA is available then can resize on gpu for better performance (main purpose of func)
//type = INCLUSIVE:
//(height = targetHeight && width <= targetWidth) || (height <= targetHeight && width = targetWidth)
//type = EXCLUSIVE:
//(height = targetHeight && width >= targetWidth) || (height >= targetHeight && width = targetWidth)
std::vector<cv::Mat> UtilityFuncs::batchResizeMat(const std::vector<cv::Mat> &images,
                                               const int t_targetHeight, const int t_targetWidth,
                                               const ResizeType t_type, QProgressBar *progressBar)
{
    std::vector<cv::Mat> result;

    progressBar->setValue(0);
    progressBar->setVisible(true);

#ifdef OPENCV_WITH_CUDA
    progressBar->setRange(0, static_cast<int>(images.size()) * 2);

    //Upload all Mat to GpuMat
    std::vector<cv::cuda::GpuMat> src, dst(static_cast<size_t>(images.size()));
    for (auto image: images)
        src.push_back(cv::cuda::GpuMat(image));

    //Calculates resize factor
    double resizeFactor = static_cast<double>(t_targetHeight) / images.front().rows;
    if ((t_type == ResizeType::EXCLUSIVE && t_targetWidth < resizeFactor * images.front().cols) ||
            (t_type == ResizeType::INCLUSIVE && t_targetWidth > resizeFactor * images.front().cols))
        resizeFactor = static_cast<double>(t_targetWidth) / images.front().cols;

    //Use INTER_AREA for decreasing, INTER_CUBIC for increasing
    cv::InterpolationFlags flags = (resizeFactor < 1) ? cv::INTER_AREA : cv::INTER_CUBIC;

    //Resize GpuMat
    auto dstIt = dst.begin(), dstEnd = dst.end();
    for (auto srcIt = src.cbegin(), srcEnd = src.cend();
         srcIt != srcEnd; ++srcIt, ++dstIt)
    {
        cv::cuda::resize(*srcIt, *dstIt, cv::Size(static_cast<int>(resizeFactor * srcIt->cols),
                                                  static_cast<int>(resizeFactor * srcIt->rows)),
                         0, 0, flags);
        progressBar->setValue(progressBar->value() + 1);
    }

    //Download resized GpuMat to Mat and update GUI
    for (auto gpuDst: dst)
    {
        cv::Mat resized;
        gpuDst.download(resized);
        result.push_back(resized);
        progressBar->setValue(progressBar->value() + 1);
    }
#else
    progressBar->setRange(0, static_cast<int>(images.size()));

    for (auto image: images)
    {
        result.push_back(UtilityFuncs::resizeImage(image, t_targetHeight, t_targetWidth, t_type));
        progressBar->setValue(progressBar->value() + 1);
    }
#endif
    progressBar->setVisible(false);

    return result;
}

#endif //SHARED_CPP_
