#ifndef SHARED_HPP_
#define SHARED_HPP_

#include <opencv2/core.hpp>
#include <QPixmap>

class UtilityFuncs
{
public:
    //Current version number
    static const int VERSION_NO = 100;
    static const int FILE_VERSION = 2;

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

    //Ensures image rows == cols, result image focus at centre of original
    static void imageToSquare(cv::Mat& t_img);

    //Converts an OpenCV Mat to a QPixmap and returns
    static QPixmap matToQPixmap(const cv::Mat &t_mat);

private:
    UtilityFuncs() {}
};
#endif //SHARED_HPP_
