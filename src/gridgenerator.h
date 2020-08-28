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

#ifndef GRIDGENERATOR_H
#define GRIDGENERATOR_H

#include <opencv2/core/mat.hpp>
#include <QImage>

#include "cellshape.h"
#include "gridutility.h"
#include "gridbounds.h"
#include "cellgroup.h"

class GridGenerator
{
public:
    static GridUtility::MosaicBestFit getGridState(const CellGroup &t_cells,
                                                const cv::Mat &t_mainImage,
                                                const int height, const int width);

private:
    GridGenerator();

    static std::pair<GridUtility::CellBestFit, bool>
    findCellState(const CellGroup &t_cells, const cv::Mat &t_mainImage, const int x, const int y,
                  const GridBounds &t_bounds, const size_t t_step = 0);
};

#endif // GRIDGENERATOR_H
