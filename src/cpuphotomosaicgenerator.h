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

    //Returns Photomosaic best fits
    GridUtility::MosaicBestFit generate() override;

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

    //Compares pixels in the cell against the library images
    //Returns the index of the library image with the smallest difference
    //Used for mode RGB_EUCLIDEAN and CIE76
    //(CIE76 is just a euclidean formulae in a different colour space)
    int findBestFitEuclidean(const cv::Mat &cell, const cv::Mat &mask,
                             const std::vector<cv::Mat> &library,
                             const std::map<size_t, int> &repeats, const cv::Rect &t_bounds) const;

    //Compares pixels in the cell against the library images
    //Returns the index of the library image with the smallest difference
    //Used for mode CIEDE2000
    int findBestFitCIEDE2000(const cv::Mat &cell, const cv::Mat &mask,
                             const std::vector<cv::Mat> &library,
                             const std::map<size_t, int> &repeats, const cv::Rect &t_bounds) const;

    //Converts degrees to radians
    double degToRad(const double deg) const;
};

#endif // CPUPHOTOMOSAICGENERATOR_H
