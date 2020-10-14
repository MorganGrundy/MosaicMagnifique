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

#ifndef PHOTOMOSAICGENERATORBASE_H
#define PHOTOMOSAICGENERATORBASE_H

#include <opencv2/core/mat.hpp>
#include <QProgressDialog>

#include "cellshape.h"
#include "imageutility.h"
#include "gridbounds.h"
#include "gridutility.h"
#include "cellgroup.h"

class PhotomosaicGeneratorBase : protected QProgressDialog
{
public:
    enum class Mode {RGB_EUCLIDEAN, CIE76, CIEDE2000};

    PhotomosaicGeneratorBase(QWidget *t_parent = nullptr);
    ~PhotomosaicGeneratorBase();

    //Sets main image
    void setMainImage(const cv::Mat &t_img);

    //Sets library images
    void setLibrary(const std::vector<cv::Mat> &t_lib);

    //Sets photomosaic mode
    void setMode(const Mode t_mode = Mode::RGB_EUCLIDEAN);

    //Sets cell group
    void setCellGroup(const CellGroup &t_cellGroup);

    //Sets grid state
    void setGridState(const GridUtility::MosaicBestFit &t_gridState);

    //Sets repeat range and addition
    void setRepeat(int t_repeatRange = 0, int t_repeatAddition = 0);

    //Generate best fits for Photomosaic cells
    //Returns true if successful
    virtual bool generateBestFits();

    //Builds photomosaic from mosaic state
    cv::Mat buildPhotomosaic(const cv::Scalar &t_backgroundColour = cv::Scalar(0, 0, 0)) const;

protected:
    cv::Mat m_img;
    std::vector<cv::Mat> m_lib;

    Mode m_mode;

    CellGroup m_cells;

    //Represents grid state and best fits for Photomosaic cells (after generate)
    GridUtility::MosaicBestFit m_bestFits;

    int m_repeatRange, m_repeatAddition;

    //Converts colour space of main image and library images
    //Resizes library based on detail level
    //Returns results
    std::pair<cv::Mat, std::vector<cv::Mat>> resizeAndCvtColor();

    //Returns the cell image at given position and it's local bounds
    std::pair<cv::Mat, cv::Rect> getCellAt(
        const CellShape &t_cellShape, const CellShape &t_detailCellShape,
        const int x, const int y, const bool t_pad,
        const cv::Mat &t_image) const;
};

#endif // PHOTOMOSAICGENERATORBASE_H
