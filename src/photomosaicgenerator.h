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

#ifndef PHOTOMOSAICGENERATOR_H
#define PHOTOMOSAICGENERATOR_H

#include <QProgressDialog>
#include <opencv2/core/mat.hpp>

#include "cellshape.h"
#include "utilityfuncs.h"
#include "gridbounds.h"
#include "gridutility.h"
#include "cellgroup.h"

class PhotomosaicGenerator : private QProgressDialog
{
    Q_OBJECT
public:
    enum class Mode {RGB_EUCLIDEAN, CIE76, CIEDE2000};

    PhotomosaicGenerator(QWidget *t_parent = nullptr);
    ~PhotomosaicGenerator();

    //Sets main image
    void setMainImage(const cv::Mat &t_img);

    //Sets library images
    void setLibrary(const std::vector<cv::Mat> &t_lib);

    //Sets photomosaic mode
    void setMode(const Mode t_mode = Mode::RGB_EUCLIDEAN);

    //Sets cell group
    void setCellGroup(const CellGroup &t_cellGroup);

    //Sets grid state
    void setGridState(const GridUtility::mosaicBestFit &t_gridState);

    //Sets repeat range and addition
    void setRepeat(int t_repeatRange = 0, int t_repeatAddition = 0);

    //Returns a Photomosaic of the main image made of the library images
    cv::Mat generate();
#ifdef CUDA
    //Returns a Photomosaic of the main image made of the library images
    //Generates using CUDA
    cv::Mat cudaGenerate();
#endif

private:
    cv::Mat m_img;
    std::vector<cv::Mat> m_lib;

    Mode m_mode;

    CellGroup m_cells;
    GridUtility::mosaicBestFit m_gridState;

    int m_repeatRange, m_repeatAddition;

    //Converts colour space of main image and library images
    //Resizes library based on detail level
    //Returns results
    std::pair<cv::Mat, std::vector<cv::Mat>> resizeAndCvtColor();
    //Resizes vector of images based on ratio
    void resizeImages(std::vector<cv::Mat> &t_images, const double t_ratio = 0.5);

    //Returns best fit index for cell if it is the grid
    std::optional<size_t> findCellBestFit(
        const CellShape &t_cellShape,
        const CellShape &t_detailCellShape,
        const int x, const int y, const bool t_pad,
        const cv::Mat &t_image, const std::vector<cv::Mat> &t_lib,
        const GridUtility::stepBestFit &t_grid) const;

    //Returns the cell image at given position and it's local bounds
    std::pair<cv::Mat, cv::Rect> getCellAt(
        const CellShape &t_cellShape, const CellShape &t_detailCellShape,
        const int x, const int y, const bool t_pad,
        const cv::Mat &t_image) const;

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

    //Calculates the repeat value of each library image in repeat range around x,y
    //Only needs to look at first half of cells as the latter half are not yet used
    std::map<size_t, int> calculateRepeats(const GridUtility::stepBestFit &grid,
                                           const int x, const int y) const;

    //Converts degrees to radians
    double degToRad(const double deg) const;

    //Combines results into a Photomosaic
    cv::Mat combineResults(const GridUtility::mosaicBestFit &results);
};

#endif // PHOTOMOSAICGENERATOR_H
