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
    along with Foobar.  If not, see <https://www.gnu.org/licenses/>.
*/

#ifndef SHARED_H
#define SHARED_H

#include <opencv2/core.hpp>
#include <QPixmap>
#include <QProgressBar>

class UtilityFuncs
{
public:
    //Current version number
    static const int VERSION_NO = 100;
    static const int MIL_VERSION = 2;
    static const unsigned int MIL_MAGIC = 0xADBE2480;

    //Enum class that represents the two different resize types, used in resizeImage
    enum class ResizeType {INCLUSIVE, EXCLUSIVE, EXACT};

    //Returns a resized copy of the given image such that
    //type = INCLUSIVE:
    //(height = targetHeight && width <= targetWidth) ||
    //    (height <= targetHeight && width = targetWidth)
    //type = EXCLUSIVE:
    //(height = targetHeight && width >= targetWidth) ||
    //    (height >= targetHeight && width = targetWidth)
    static cv::Mat resizeImage(const cv::Mat &t_img,
                               const int t_targetHeight, const int t_targetWidth,
                               const ResizeType t_type);

    //Enum class that represents the two different methods of squaring an image
    enum class SquareMethod {PAD, CROP};

    //Ensures image rows == cols
    //method = PAD:
    //Pads the image's smaller dimension with black pixels
    //method = CROP:
    //Crops the image's larger dimension with focus at image centre
    static void imageToSquare(cv::Mat& t_img, const SquareMethod t_method);

    //Converts an OpenCV Mat to a QPixmap and returns
    static QPixmap matToQPixmap(const cv::Mat &t_mat);

    static std::vector<cv::Mat> batchResizeMat(const std::vector<cv::Mat> &images,
                                               const int t_targetHeight, const int t_targetWidth,
                                               const ResizeType t_type,
                                               QProgressBar *progressBar = nullptr);

    //Takes a grayscale image as src
    //Converts to RGBA and makes pixels of target value transparent
    //Returns result in dst
    static void matMakeTransparent(const cv::Mat &t_src, cv::Mat &t_dst, const int t_targetValue);

    //Replace dst with edge detected version of src
    static void edgeDetect(const cv::Mat &t_src, cv::Mat &t_dst);

private:
    UtilityFuncs() {}
};

//Outputs a OpenCV mat to a QDataStream
//Can be used to save a OpenCV mat to a file
QDataStream &operator<<(QDataStream &t_out, const cv::Mat &t_mat);

//Inputs a OpenCV mat from a QDataStream
//Can be used to load a OpenCV mat from a file
QDataStream &operator>>(QDataStream &t_in, cv::Mat &t_mat);

#endif //SHARED_H
