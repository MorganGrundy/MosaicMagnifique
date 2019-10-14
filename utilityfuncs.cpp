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
    std::vector<cv::Mat> result(images.size(), cv::Mat());

#ifdef OPENCV_WITH_CUDA
    if (progressBar != nullptr)
    {
        progressBar->setMaximum(0);
        progressBar->setValue(0);
        progressBar->setVisible(true);
    }

    cv::cuda::Stream stream;
    std::vector<cv::cuda::GpuMat> src(images.size()), dst(images.size());
    for (size_t i = 0; i < images.size(); ++i)
    {
        //Calculates resize factor
        double resizeFactor = static_cast<double>(t_targetHeight) / images.at(i).rows;
        if ((t_type == ResizeType::EXCLUSIVE &&
             t_targetWidth < resizeFactor * images.at(i).cols) ||
                (t_type == ResizeType::INCLUSIVE &&
                 t_targetWidth > resizeFactor * images.at(i).cols))
            resizeFactor = static_cast<double>(t_targetWidth) / images.at(i).cols;

        //Use INTER_AREA for decreasing, INTER_CUBIC for increasing
        cv::InterpolationFlags flags = (resizeFactor < 1) ? cv::INTER_AREA : cv::INTER_CUBIC;

        //Resize image
        src.at(i).upload(images.at(i), stream);
        cv::cuda::resize(src.at(i), dst.at(i),
                         cv::Size(static_cast<int>(resizeFactor * src.at(i).cols),
                                  static_cast<int>(resizeFactor * src.at(i).rows)),
                         0, 0, flags, stream);
        dst.at(i).download(result.at(i), stream);
    }
    stream.waitForCompletion();
#else
    if (progressBar != nullptr)
    {
        progressBar->setMaximum(static_cast<int>(images.size()));
        progressBar->setValue(0);
        progressBar->setVisible(true);
    }

    for (size_t i = 0; i < images.size(); ++i)
    {
        result.at(i) = UtilityFuncs::resizeImage(images.at(i), t_targetHeight, t_targetWidth,
                                                 t_type);
        if (progressBar != nullptr)
            progressBar->setValue(progressBar->value() + 1);
    }
#endif
    progressBar->setVisible(false);

    return result;
}

//Outputs a OpenCV mat to a QDataStream
//Can be used to save a OpenCV mat to a file
QDataStream &operator<<(QDataStream &t_out, const cv::Mat &t_mat)
{
    t_out << t_mat.type() << t_mat.rows << t_mat.cols;

    const int dataSize = t_mat.cols * t_mat.rows * static_cast<int>(t_mat.elemSize());
    QByteArray data = QByteArray::fromRawData(reinterpret_cast<const char *>(t_mat.ptr()),
                                              dataSize);
    t_out << data;

    return t_out;
}

//Inputs a OpenCV mat from a QDataStream
//Can be used to load a OpenCV mat from a file
QDataStream &operator>>(QDataStream &t_in, cv::Mat &t_mat)
{
    int type, rows, cols;
    QByteArray data;
    t_in >> type >> rows >> cols;
    t_in >> data;

    t_mat = cv::Mat(rows, cols, type, data.data()).clone();

    return t_in;
}

#endif //SHARED_CPP_
