/*
	Copyright © 2018-2020, Morgan Grundy

	This file is part of Mosaic Magnifique.

    Mosaic Magnifique is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    Mosaic Magnifique is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with Mosaic Magnifique.  If not, see <https://www.gnu.org/licenses/>.
*/

#ifndef SHARED_H
#define SHARED_H

#include <opencv2/core.hpp>
#include <QPixmap>
#include <QProgressBar>

namespace ImageUtility
{
    const int MIL_VERSION = 2;
    const unsigned int MIL_MAGIC = 0xADBE2480;

    //Represents the different resize types
    //INCLUSIVE: Resize such that new size is a subsection of target size
    //EXCLUSIVE: Resize such that target size is a subsection of new size
    //EXACT: Resize to exact target size
    enum class ResizeType {INCLUSIVE, EXCLUSIVE, EXACT};

    //Returns copy of image resized to target size with given resize type
    cv::Mat resizeImage(const cv::Mat &t_img, const int t_targetHeight, const int t_targetWidth,
                        const ResizeType t_type);

    //Returns copy of images resized to target size with given resize type
    //If OpenCV CUDA is available then will resize on gpu
    std::vector<cv::Mat> batchResizeMat(const std::vector<cv::Mat> &images,
                                        const int t_targetHeight, const int t_targetWidth,
                                        const ResizeType t_type,
                                        QProgressBar *progressBar = nullptr);

    //Returns copy of images resized to (first image size * ratio)
    std::vector<cv::Mat> batchResizeMat(const std::vector<cv::Mat> &t_images,
                                        const double t_ratio = 0.5);

    //Enum class that represents the two different methods of squaring an image
    enum class SquareMethod {PAD, CROP};

    //Ensures image rows == cols
    //method = PAD:
    //Pads the image's smaller dimension with black pixels
    //method = CROP:
    //Crops the image's larger dimension with focus at image centre
    void imageToSquare(cv::Mat& t_img, const SquareMethod t_method);

    //Converts an OpenCV Mat to a QPixmap and returns
    QPixmap matToQPixmap(const cv::Mat &t_mat,
                         const QImage::Format t_format = QImage::Format_RGB888);

    //Takes a grayscale image as src
    //Converts to RGBA and makes pixels of target value transparent
    //Returns result in dst
    void matMakeTransparent(const cv::Mat &t_src, cv::Mat &t_dst, const int t_targetValue);

    //Replace dst with edge detected version of src
    void edgeDetect(const cv::Mat &t_src, cv::Mat &t_dst);

    //Adds an alpha channel to the given images
    void addAlphaChannel(std::vector<cv::Mat> &t_images);
};

//Outputs a OpenCV mat to a QDataStream
//Can be used to save a OpenCV mat to a file
QDataStream &operator<<(QDataStream &t_out, const cv::Mat &t_mat);

//Inputs a OpenCV mat from a QDataStream
//Can be used to load a OpenCV mat from a file
QDataStream &operator>>(QDataStream &t_in, cv::Mat &t_mat);

#endif //SHARED_H
