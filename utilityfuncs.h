#ifndef SHARED_HPP_
#define SHARED_HPP_

#include <opencv2/core.hpp>
#include <QPixmap>
#include <QProgressBar>

#define OPENCV_WITH_CUDA
#define CUDA

class UtilityFuncs
{
public:
    //Current version number
    static const int VERSION_NO = 100;
    static const int MIL_VERSION = 2;
    static const int MCS_VERSION = 3;

    //Enum class that represents the two different resize types, used in resizeImage
    enum class ResizeType {INCLUSIVE, EXCLUSIVE};

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

private:
    UtilityFuncs() {}
};

//Outputs a OpenCV mat to a QDataStream
//Can be used to save a OpenCV mat to a file
QDataStream &operator<<(QDataStream &t_out, const cv::Mat &t_mat);

//Inputs a OpenCV mat from a QDataStream
//Can be used to load a OpenCV mat from a file
QDataStream &operator>>(QDataStream &t_in, cv::Mat &t_mat);

#endif //SHARED_HPP_
