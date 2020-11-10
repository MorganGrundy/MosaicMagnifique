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
    if (t_src.channels() != 1)
        throw std::invalid_argument(Q_FUNC_INFO " only supports single channel images");

    cv::Mat tmp(t_src.rows, t_src.cols, t_src.type());
    //Calculate width of edge as percentage of image size
    const size_t edgeWidth = std::ceil(0.03 * std::min(t_src.rows, t_src.cols)) - 1;

    //Edge detect
    float edgeKernelData[9] = {-1, -1, -1,
                               -1, 8, -1,
                               -1, -1, -1};
    const cv::Mat edgeKernel(3, 3, CV_32FC1, edgeKernelData);
    cv::filter2D(t_src, tmp, -1, edgeKernel, cv::Point(-1, -1), 0, cv::BORDER_CONSTANT);

    //Dilate edge
    if (edgeWidth > 0)
    {
        const cv::Mat dilateKernel =
            cv::getStructuringElement(cv::MorphShapes::MORPH_RECT,
                                      cv::Size(edgeWidth * 2 + 1, edgeWidth * 2 + 1));
        cv::dilate(tmp, tmp, dilateKernel);
    }

    //Remove parts of edge outside active area
    cv::bitwise_and(tmp, t_src, tmp);
    t_dst = tmp.clone();
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

//Crop image to square, such that maximum number of features in crop
void ImageUtility::squareToFeatures(const cv::Mat &t_in, cv::Mat &t_out,
                                    const std::shared_ptr<cv::FastFeatureDetector> &t_featureDetector)
{
    //Check for empty image
    if (t_in.empty())
        throw std::invalid_argument("t_in was empty.");

    //Check if image already square
    if (t_in.rows == t_in.cols)
    {
        t_out = t_in;
        return;
    }

    //Check for empty feature detector
    if (t_featureDetector == nullptr)
        throw std::invalid_argument("t_featureDetector was empty.");

    //Detect features
    std::vector<cv::KeyPoint> keyPoints;
    t_featureDetector->detect(t_in, keyPoints);
    if (keyPoints.empty())
        throw std::invalid_argument("No features detected in t_in.");

    //Find shortest side, use as size of cropped image
    const int cropSize = std::min(t_in.rows,  t_in.cols);

    //Stores current best crop of image and it's feature count
    cv::Rect bestCrop;
    size_t bestCropFeatureCount = 0;
    //Distance between crop center and keypoint average
    double bestCropDistFromCenter = 0;
    //Find crop with highest feature count
    for (int cropOffset = 0; cropOffset + cropSize < std::max(t_in.rows, t_in.cols);
         cropOffset += 4)
    {
        const cv::Rect crop((t_in.rows > t_in.cols) ? cv::Point(0, cropOffset)
                                                    : cv::Point(cropOffset, 0),
                            cv::Size(cropSize, cropSize));

        //Calculate average position of all keypoints in crop
        cv::Point2f keypointAverage(0, 0);
        //Count features in crop
        size_t cropFeatureCount = 0;
        for (const auto &keyPoint : keyPoints)
        {
            if (crop.contains(keyPoint.pt))
            {
                keypointAverage += keyPoint.pt;
                ++cropFeatureCount;
            }
        }
        keypointAverage = cv::Point2f(keypointAverage.x / cropFeatureCount,
                                      keypointAverage.y / cropFeatureCount);
        //Calculate distance between keypoint average and crop center
        const double distFromCenter = std::sqrt(
            std::pow(keypointAverage.x - (crop.x + crop.width / 2), 2) +
            std::pow(keypointAverage.y - (crop.y + crop.height / 2), 2));

        //New best crop if more features, or equal features but average closer to crop center
        if (cropFeatureCount > bestCropFeatureCount ||
            (cropFeatureCount == bestCropFeatureCount && distFromCenter < bestCropDistFromCenter))
        {
            bestCropFeatureCount = cropFeatureCount;
            bestCropDistFromCenter = distFromCenter;
            bestCrop = crop;
        }
    }

    //Copy best crop of image to output
    t_out = t_in(bestCrop);
}

//Crop image to square, such that maximum entropy in crop
void ImageUtility::squareToEntropy(const cv::Mat &t_in, cv::Mat &t_out)
{
    //Check for empty image
    if (t_in.empty())
        throw std::invalid_argument("t_in was empty.");

    //Check if image already square
    if (t_in.rows == t_in.cols)
    {
        t_out = t_in;
        return;
    }

    //Find shortest side, use as size of cropped image
    const int cropSize = std::min(t_in.rows,  t_in.cols);
    //Checking every possible crop takes a long time, so only check some
    const int cropStepSize = cropSize / 16;

    //Stores current best crop of image and it's entropy value
    cv::Rect bestCrop;
    double bestCropEntropy = 0;
    //Find crop with highest entropy
    for (int cropOffset = 0; cropOffset + cropSize < std::max(t_in.rows, t_in.cols);
         cropOffset += cropStepSize)
    {
        const cv::Rect crop((t_in.rows > t_in.cols) ? cv::Point(0, cropOffset)
                                                    : cv::Point(cropOffset, 0),
                            cv::Size(cropSize, cropSize));

        double cropEntropy = ImageUtility::calculateEntropy(t_in(crop));

        //New best crop if higher entropy
        if (cropEntropy > bestCropEntropy)
        {
            bestCropEntropy = cropEntropy;
            bestCrop = crop;
        }
    }

    t_out = t_in(bestCrop);
}

//Crop image to square, such that maximum number of objects in crop
void ImageUtility::squareToCascadeClassifier(const cv::Mat &t_in, cv::Mat &t_out,
                                             cv::CascadeClassifier &t_cascadeClassifier)
{
    //Check for empty image
    if (t_in.empty())
        throw std::invalid_argument("t_in was empty.");

    //Check if image already square
    if (t_in.rows == t_in.cols)
    {
        t_out = t_in;
        return;
    }

    //Check that cascade classifier is loaded
    if (t_cascadeClassifier.empty())
        throw std::invalid_argument("t_cascadeClassifier was empty.");

    //Find shortest side, use as size of cropped image
    const int cropSize = std::min(t_in.rows,  t_in.cols);

    //Convert image to grayscale
    cv::Mat gray;
    cv::cvtColor(t_in, gray, cv::COLOR_BGR2GRAY);
    cv::equalizeHist(gray, gray);

    //Detect objects
    std::vector<cv::Rect> objects;
    t_cascadeClassifier.detectMultiScale(gray, objects);
    if (objects.empty())
        throw std::invalid_argument("No objects detected in t_in.");

    //Stores current best crop of image, and it's badness value
    cv::Rect bestCrop;
    double bestCropBadnessValue = std::numeric_limits<double>::max();
    //Find crop with lowest badness value
    for (int cropOffset = 0; cropOffset + cropSize < std::max(t_in.rows, t_in.cols);
         cropOffset += 8)
    {
        const cv::Rect crop((t_in.rows > t_in.cols) ? cv::Point(0, cropOffset)
                                                    : cv::Point(cropOffset, 0),
                            cv::Size(cropSize, cropSize));

        //Calculate how well objects fit in crop
        double cropBadnessValue = (objects.empty()) ? std::numeric_limits<double>::max() : 0;
        for (const auto &object : objects)
        {
            //Calculate rect of object visible in crop
            const cv::Rect objectVisible = crop & object;

            //Calculate distance between object and crop center
            const cv::Point objectCenter(object.x + object.width / 2, object.y + object.height / 2);
            const cv::Point distFromCenter(crop.x + crop.width / 2 - objectCenter.x,
                                           crop.y + crop.height / 2 - objectCenter.y);

            //Calculate how well object fits in crop, scales with distance from center
            double objectBadnessValue = std::sqrt(std::pow(distFromCenter.x, 2) +
                                                  std::pow(distFromCenter.y, 2));

            //Increase badness value if object not fully visible in crop
            if (objectVisible.area() < object.area())
            {
                //Increase badness value even more if object not visible at all
                if (objectVisible.area() > 0)
                    objectBadnessValue *= 5.0 * (object.area() / objectVisible.area());
                else
                    objectBadnessValue *= 10.0 * object.area();
            }

            cropBadnessValue += objectBadnessValue;
        }

        //If badness value less than current best then new best crop
        if (cropBadnessValue < bestCropBadnessValue)
        {
            bestCropBadnessValue = cropBadnessValue;
            bestCrop = crop;
        }
    }

    //Copy best crop of image to output
    t_out = t_in(bestCrop);
}

//Outputs a OpenCV mat to a QDataStream
//Can be used to save a OpenCV mat to a file
QDataStream &operator<<(QDataStream &t_out, const cv::Mat &t_mat)
{
    t_out << static_cast<quint32>(t_mat.type()) << static_cast<quint32>(t_mat.rows)
          << static_cast<quint32>(t_mat.cols);

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
    quint32 type, rows, cols;
    QByteArray data;
    t_in >> type >> rows >> cols;
    t_in >> data;

    t_mat = cv::Mat(rows, cols, type, data.data()).clone();

    return t_in;
}

#endif //SHARED_CPP_
