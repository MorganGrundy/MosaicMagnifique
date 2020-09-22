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

#ifndef SHARED_CPP_
#define SHARED_CPP_

#include "imageutility.h"

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <QDebug>

#ifdef OPENCV_W_CUDA
#include <opencv2/cudawarping.hpp>
#endif

//Returns copy of image resized to target size with given resize type
cv::Mat ImageUtility::resizeImage(const cv::Mat &t_img,
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
    if (t_type == ResizeType::EXACT)
        cv::resize(t_img, result, cv::Size(t_targetWidth, t_targetHeight), 0, 0, flags);
    else
        cv::resize(t_img, result, cv::Size(std::round(resizeFactor * t_img.cols),
                                           std::round(resizeFactor * t_img.rows)),
                   0, 0, flags);
    return result;
}

//Populates dst with copy of images resized to target size with given resize type from src
//If OpenCV CUDA is available then will resize on gpu
void ImageUtility::batchResizeMat(const std::vector<cv::Mat> &t_src, std::vector<cv::Mat> &t_dst,
                                  const int t_targetHeight, const int t_targetWidth,
                                  const ResizeType t_type, QProgressBar *progressBar)
{
    t_dst.resize(t_src.size());

#ifdef OPENCV_W_CUDA
    if (progressBar != nullptr)
    {
        progressBar->setMaximum(0);
        progressBar->setValue(0);
        progressBar->setVisible(true);
    }

    cv::cuda::Stream stream;
    std::vector<cv::cuda::GpuMat> src(t_src.size()), dst(t_src.size());
    for (size_t i = 0; i < t_src.size(); ++i)
    {
        //Calculates resize factor
        double resizeFactor = static_cast<double>(t_targetHeight) / t_src.at(i).rows;
        if ((t_type == ResizeType::EXCLUSIVE &&
             t_targetWidth < resizeFactor * t_src.at(i).cols) ||
            (t_type == ResizeType::INCLUSIVE &&
             t_targetWidth > resizeFactor * t_src.at(i).cols))
            resizeFactor = static_cast<double>(t_targetWidth) / t_src.at(i).cols;

        //Use INTER_AREA for decreasing, INTER_CUBIC for increasing
        cv::InterpolationFlags flags = (resizeFactor < 1) ? cv::INTER_AREA : cv::INTER_CUBIC;

        //Resize image
        src.at(i).upload(t_src.at(i), stream);
        if (t_type == ResizeType::EXACT)
            cv::cuda::resize(src.at(i), dst.at(i), cv::Size(t_targetWidth, t_targetHeight),
                             0, 0, flags, stream);
        else
            cv::cuda::resize(src.at(i), dst.at(i),
                             cv::Size(std::round(resizeFactor * src.at(i).cols),
                                      std::round(resizeFactor * src.at(i).rows)),
                             0, 0, flags, stream);
        dst.at(i).download(t_dst.at(i), stream);
    }
    stream.waitForCompletion();
#else
    if (progressBar != nullptr)
    {
        progressBar->setMaximum(static_cast<int>(t_src.size()));
        progressBar->setValue(0);
        progressBar->setVisible(true);
    }

    for (size_t i = 0; i < t_src.size(); ++i)
    {
        t_dst.at(i) = ImageUtility::resizeImage(t_src.at(i), t_targetHeight, t_targetWidth, t_type);
        if (progressBar != nullptr)
            progressBar->setValue(progressBar->value() + 1);
    }
#endif
    if (progressBar != nullptr)
        progressBar->setVisible(false);
}

//Resizes images to (first image size * ratio)
bool ImageUtility::batchResizeMat(std::vector<cv::Mat> &t_images, const double t_ratio)
{
    //No images to resize
    if (t_images.empty())
        return false;

    batchResizeMat(t_images, t_images, std::round(t_ratio * t_images.front().rows),
                   std::round(t_ratio * t_images.front().cols), ResizeType::EXACT);

    return true;
}

//Ensures image rows == cols
//method = PAD:
//Pads the image's smaller dimension with black pixels
//method = CROP:
//Crops the image's larger dimension with focus at image centre
void ImageUtility::imageToSquare(cv::Mat& t_img, const SquareMethod t_method)
{
    //Already square
    if (t_img.cols == t_img.rows)
        return;

    if (t_method == SquareMethod::CROP)
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
    else if (t_method == SquareMethod::PAD)
    {
        const int newSize = std::max(t_img.cols, t_img.rows);
        cv::Mat result(newSize, newSize, t_img.type());
        cv::copyMakeBorder(t_img, result, 0, newSize - t_img.rows, 0, newSize - t_img.cols,
                           cv::BORDER_CONSTANT, cv::Scalar(0));
        t_img = result;
    }
}

//Converts an OpenCV Mat to a QPixmap and returns
QPixmap ImageUtility::matToQPixmap(const cv::Mat &t_mat,
                                   const QImage::Format t_format)
{
    return QPixmap::fromImage(QImage(t_mat.data, t_mat.cols, t_mat.rows,
                                     static_cast<int>(t_mat.step), t_format).rgbSwapped());
}

//Takes a grayscale image as src
//Converts to RGBA and makes pixels of target value transparent
//Returns result in dst
void ImageUtility::matMakeTransparent(const cv::Mat &t_src, cv::Mat &t_dst, const int t_targetValue)
{
    cv::Mat tmp;
    cv::cvtColor(t_src, tmp, cv::COLOR_GRAY2RGBA);

    //Make black pixels transparent
    int nRows = tmp.rows;
    int nCols = tmp.cols;
    if (tmp.isContinuous())
    {
        nCols *= nRows;
        nRows = 1;
    }
    cv::Vec4b *p;
    for (int i = 0; i < nRows; ++i)
    {
        p = tmp.ptr<cv::Vec4b>(i);
        for (int j = 0; j < nCols; ++j)
        {
            if (p[j][0] == t_targetValue)
                p[j][3] = 0;
        }
    }
    t_dst = tmp;
}

//Replace dst with edge detected version of src
void ImageUtility::edgeDetect(const cv::Mat &t_src, cv::Mat &t_dst)
{
    float kernelData[9] = {-1, -1, -1, -1, 8, -1, -1, -1, -1};
    const cv::Mat kernel(3, 3, CV_32FC1, kernelData);
    cv::filter2D(t_src, t_dst, -1, kernel, cv::Point(-1, -1), 0, cv::BORDER_CONSTANT);
}

//Adds an alpha channel to the given images
void ImageUtility::addAlphaChannel(std::vector<cv::Mat> &t_images)
{
    for (auto &image: t_images)
    {
        //Split image channels
        std::vector<cv::Mat> channels(3);
        cv::split(image, channels);
        //Add alpha channel
        channels.push_back(
            cv::Mat(image.size(), channels.front().type(), cv::Scalar(255)));
        //Merge image channels
        cv::merge(channels, image);
    }
}

//Calculates entropy of an image, can take a mask image
double ImageUtility::calculateEntropy(const cv::Mat &t_in, const cv::Mat &t_mask)
{
    if (t_in.empty())
        return 0;

    if (!t_mask.empty())
    {
        if (t_mask.rows != t_in.rows || t_mask.cols != t_in.cols)
        {
            qDebug() << Q_FUNC_INFO << "Mask size differs from image";
            return 0;
        }
        else if (t_mask.channels() != 1)
        {
            qDebug() << Q_FUNC_INFO << "Mask should be single channel, was" << t_mask.channels();
            return 0;
        }
    }

    //Convert image to grayscale
    cv::Mat grayImage;
    cv::cvtColor(t_in, grayImage, cv::COLOR_BGR2GRAY);

    //Calculate histogram in cell shape
    size_t pixelCount = 0;
    std::vector<size_t> histogram(256, 0);

    const uchar *p_im, *p_mask;
    for (int row = 0; row < grayImage.rows; ++row)
    {
        p_im = grayImage.ptr<uchar>(row);
        if (!t_mask.empty())
            p_mask = t_mask.ptr<uchar>(row);
        for (int col = 0; col < grayImage.cols; ++col)
        {
            if (t_mask.empty() || p_mask[col] != 0)
            {
                ++histogram.at(p_im[col]);
                ++pixelCount;
            }
        }
    }

    //Calculate entropy
    double entropy = 0;
    for (auto value: histogram)
    {
        const double probability = value / static_cast<double>(pixelCount);
        if (probability > 0)
            entropy -= probability * std::log2(probability);
    }

    return entropy;
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
