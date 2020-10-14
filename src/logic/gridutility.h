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

#include <opencv2/core/mat.hpp>
#include <optional>

#include "cellshape.h"

namespace GridUtility
{
    using CellBestFit = std::optional<size_t>;
    using StepBestFit = std::vector<std::vector<CellBestFit>>;
    using MosaicBestFit = std::vector<StepBestFit>;

    [[maybe_unused]] const int PAD_GRID = 2;

    //Calculates the number of given cells needed to fill an image of given size
    cv::Point calculateGridSize(const CellShape &t_cellShape,
                                const int t_imageWidth, const int t_imageHeight,
                                const int t_pad);

    //Returns rect of cell shape at the given grid position
    cv::Rect getRectAt(const CellShape &t_cellShape, const int t_x, const int t_y);

    //Stores horizontal and vertical flip state
    struct FlipState
    {
        FlipState(const bool t_horizontal, const bool t_vertical)
            : horizontal{t_horizontal}, vertical{t_vertical}
        {}
        FlipState() : FlipState{false, false}
        {}

        bool horizontal{false};
        bool vertical{false};
    };

    //Returns the flip state of the given cell at given grid position
    FlipState getFlipStateAt(const CellShape &t_cellShape, const int t_x, const int t_y,
                             const int t_pad);
};

#endif // CELLGRID_H
