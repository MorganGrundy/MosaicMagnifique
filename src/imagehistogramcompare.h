#ifndef IMAGEHISTOGRAMCOMPARE_H
#define IMAGEHISTOGRAMCOMPARE_H

#include <opencv2/core/mat.hpp>

namespace ImageHistogramCompare
{
typedef std::vector<std::pair<std::tuple<int, int, int>, float>> colourPriorityList;

//Returns a list of colours in main image that are lacking in library images
//Sorted by descending priority
//Uses image histograms with colour grouped into 30x30x30 bins
colourPriorityList getColourPriorityList(const cv::Mat &t_image, const std::vector<cv::Mat> &t_libImages);
};

#endif // IMAGEHISTOGRAMCOMPARE_H
