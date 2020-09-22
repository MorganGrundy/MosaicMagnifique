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

#ifndef CPUPHOTOMOSAICGENERATOR_H
#define CPUPHOTOMOSAICGENERATOR_H

#include "photomosaicgeneratorbase.h"

//Generates a Photomosaic on CPU
class CPUPhotomosaicGenerator : public PhotomosaicGeneratorBase
{
public:
    CPUPhotomosaicGenerator(QWidget *t_parent = nullptr);

    //Generate best fits for Photomosaic cells
    //Returns true if successful
    bool generateBestFits() override;

private:
    //Returns best fit index for cell if it is the grid
    std::optional<size_t> findCellBestFit(const CellShape &t_cellShape,
                                          const CellShape &t_detailCellShape,
                                          const int x, const int y, const bool t_pad,
                                          const cv::Mat &t_image, const std::vector<cv::Mat> &t_lib,
                                          const GridUtility::StepBestFit &t_grid) const;

    //Calculates the repeat value of each library image in repeat range around x,y
    //Only needs to look at first half of cells as the latter half are not yet used
    std::map<size_t, int> calculateRepeats(const GridUtility::StepBestFit &grid,
                                           const int x, const int y) const;
};

#endif // CPUPHOTOMOSAICGENERATOR_H
