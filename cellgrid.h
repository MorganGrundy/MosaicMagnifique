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

#ifndef CELLGRID_H
#define CELLGRID_H

#include <opencv2/core.hpp>
#include <optional>

#include "cellshape.h"

class CellGrid
{
public:
    typedef std::optional<size_t> cellBestFit;
    typedef std::vector<std::vector<cellBestFit>> stepBestFit;
    typedef std::vector<stepBestFit> mosaicBestFit;

    static const int PAD_GRID = 2;

    static cv::Point calculateGridSize(const CellShape &t_cellShape,
                                       const int t_imageWidth, const int t_imageHeight,
                                       const int t_pad);

    static cv::Rect getRectAt(const CellShape &t_cellShape, const int t_x, const int t_y);

    static std::pair<bool, bool> getFlipStateAt(const CellShape &t_cellShape,
                                                const int t_x, const int t_y, const int t_pad);

    //Maximum possible entropy value
    static double MAX_ENTROPY() {return 8.0;};
    static double calculateEntropy(const cv::Mat &t_mask, const cv::Mat &t_image);

private:
    CellGrid() {}
};

#endif // CELLGRID_H
