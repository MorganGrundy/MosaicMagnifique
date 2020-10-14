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
