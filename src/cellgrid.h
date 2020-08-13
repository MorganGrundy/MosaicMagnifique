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

#ifndef CELLGRID_H
#define CELLGRID_H

#include <opencv2/core.hpp>
#include <optional>

#include "cellshape.h"

namespace CellGrid
{
    typedef std::optional<size_t> cellBestFit;
    typedef std::vector<std::vector<cellBestFit>> stepBestFit;
    typedef std::vector<stepBestFit> mosaicBestFit;

    const int PAD_GRID = 2;

    cv::Point calculateGridSize(const CellShape &t_cellShape,
                                const int t_imageWidth, const int t_imageHeight,
                                const int t_pad);

    cv::Rect getRectAt(const CellShape &t_cellShape, const int t_x, const int t_y);

    std::pair<bool, bool> getFlipStateAt(const CellShape &t_cellShape,
                                         const int t_x, const int t_y, const int t_pad);

    //Maximum possible entropy value
    const double MAX_ENTROPY = 8.0;
    double calculateEntropy(const cv::Mat &t_mask, const cv::Mat &t_image);
};

#endif // CELLGRID_H
