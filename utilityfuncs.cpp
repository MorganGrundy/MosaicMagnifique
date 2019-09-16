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

    //Resizes image
    cv::Mat result;
    if (resizeFactor < 1)
        resize(t_img, result, cv::Size(static_cast<int>(resizeFactor * t_img.cols),
                                     static_cast<int>(resizeFactor * t_img.rows)), 0, 0,
               cv::INTER_AREA);
    else if (resizeFactor > 1)
        resize(t_img, result, cv::Size(static_cast<int>(resizeFactor * t_img.cols),
                                     static_cast<int>(resizeFactor * t_img.rows)), 0, 0,
               cv::INTER_CUBIC);
    else
        result = t_img;
    return result;
}

//Ensures image rows == cols, result image focus at centre of original
void UtilityFuncs::imageToSquare(cv::Mat& t_img)
{
    if (t_img.cols < t_img.rows)
    {
        int diff = (t_img.rows - t_img.cols)/2;
        t_img = t_img(cv::Range(diff, t_img.rows - diff), cv::Range(0, t_img.cols));
    }
    else if (t_img.cols > t_img.rows)
    {
        int diff = (t_img.cols - t_img.rows)/2;
        t_img = t_img(cv::Range(0, t_img.rows), cv::Range(diff, t_img.cols - diff));
    }
}

//Converts an OpenCV Mat to a QPixmap and returns
QPixmap UtilityFuncs::matToQPixmap(const cv::Mat &t_mat)
{
    return QPixmap::fromImage(QImage(t_mat.data, t_mat.cols, t_mat.rows,
                                     static_cast<int>(t_mat.step),
                                     QImage::Format_RGB888).rgbSwapped());
}

#endif //SHARED_CPP_
