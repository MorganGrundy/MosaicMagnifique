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

#include "imagehistogramcompare.h"

#include <opencv2/imgproc.hpp>

//Returns a list of colours in main image that are lacking in library images
//Sorted by descending priority
//Uses image histograms with colour grouped into 30x30x30 bins
ImageHistogramCompare::colourPriorityList
ImageHistogramCompare::getColourPriorityList(const cv::Mat &t_image,
                                             const std::vector<cv::Mat> &t_libImages)
{
    //Number of bins in histogram
    const int noOfBins = 30;
    const int histogramSize[3] = {noOfBins, noOfBins, noOfBins};

    //Histogram range
    const float RGBRanges[2] = {0, 256};
    const float *ranges[3] = {RGBRanges, RGBRanges, RGBRanges};

    const int channels[3] = {0, 1, 2};

    //Create main histogram
    cv::Mat mainHistogram;
    cv::calcHist(&t_image, 1, channels, cv::Mat(), mainHistogram, 3, histogramSize, ranges,
                 true, false);

    //Create library histogram
    cv::Mat libraryHistogram;
    cv::calcHist(t_libImages.data(), static_cast<int>(t_libImages.size()), channels, cv::Mat(),
                 libraryHistogram, 3, histogramSize, ranges, true, false);

    //Stores needed colour bins and priorities
    colourPriorityList colourPriority;

    //Iterates over all bins in histogram
    for (int b = 0; b < histogramSize[2]; ++b)
    {
        for (int g = 0; g < histogramSize[1]; ++g)
        {
            for (int r = 0; r < histogramSize[0]; ++r)
            {
                float mainBin = mainHistogram.at<float>(b, g, r);
                //Bin not empty in main image
                if (mainBin > 0)
                {
                    float libraryBin = libraryHistogram.at<float>(b, g, r);

                    //Calculate priority
                    float priority = 0;
                    //Library bin empty instead treat as 0.5
                    if (libraryBin == 0)
                        priority = mainBin * 2;
                    else
                        priority = mainBin / libraryBin;

                    //Get median RGB colour of bin
                    std::tuple<int, int, int> binMedianRGB = std::make_tuple(
                        static_cast<int>((r + 0.5) * ((RGBRanges[1] - 1) / noOfBins)),
                        static_cast<int>((g + 0.5) * ((RGBRanges[1] - 1) / noOfBins)),
                        static_cast<int>((b + 0.5) * ((RGBRanges[1] - 1) / noOfBins)));
                    colourPriority.push_back({binMedianRGB, priority});
                }
            }
        }
    }

    //Sort colour bins by descending priority
    std::sort(colourPriority.begin(), colourPriority.end(),
              [](const auto &a, const auto &b)
    {
        return a.second > b.second;
    });

    return colourPriority;
}
