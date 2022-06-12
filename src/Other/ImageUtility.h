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

#pragma once

#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/objdetect.hpp>
#include <QPixmap>
#include <QProgressBar>

#ifdef CUDA
#include <opencv2/cudaimgproc.hpp>
#endif

namespace ImageUtility
{
    //Represents the different resize types
    //INCLUSIVE: Resize such that new size is a subsection of target size
    //EXCLUSIVE: Resize such that target size is a subsection of new size
    //EXACT: Resize to exact target size
    enum class ResizeType {INCLUSIVE, EXCLUSIVE, EXACT};

    //Returns copy of image resized to target size with given resize type
    cv::Mat resizeImage(const cv::Mat &t_img, const int t_targetHeight, const int t_targetWidth,
                        const ResizeType t_type);

    //Populates dst with copy of images resized to target size with given resize type from src
    //If OpenCV CUDA is available then will resize on gpu
    void batchResizeMat(const std::vector<cv::Mat> &t_src, std::vector<cv::Mat> &t_dst,
                        const int t_targetHeight, const int t_targetWidth, const ResizeType t_type,
                        QProgressBar *progressBar = nullptr);

    //Resizes images to (first image size * ratio)
    bool batchResizeMat(std::vector<cv::Mat> &t_images, const double t_ratio = 0.5);

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

    //Maximum possible entropy value
    [[maybe_unused]] const double MAX_ENTROPY = 8.0;
    //Calculates entropy of an image, can take a mask image
    double calculateEntropy(const cv::Mat &t_in, const cv::Mat &t_mask = cv::Mat());

    //Enum class that represents the two different methods of squaring an image
    enum class SquareMethod {PAD, CROP};

    //Ensures image rows == cols
    //method = PAD:
    //Pads the image's smaller dimension with black pixels
    //method = CROP:
    //Crops the image's larger dimension with focus at image centre
    void imageToSquare(cv::Mat& t_img, const SquareMethod t_method);

    //Crop image to square, such that maximum number of features in crop
    void squareToFeatures(const cv::Mat &t_in, cv::Mat &t_out,
                          const std::shared_ptr<cv::FastFeatureDetector> &t_featureDetector =
                              cv::FastFeatureDetector::create());

    //Crop image to square, such that maximum entropy in crop
    void squareToEntropy(const cv::Mat &t_in, cv::Mat &t_out);

    //Crop image to square, such that maximum number of objects in crop
    void squareToCascadeClassifier(const cv::Mat &t_in, cv::Mat &t_out,
                                   cv::CascadeClassifier &t_cascadeClassifier);
};