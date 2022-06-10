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

#pragma once

#include <opencv2/core/mat.hpp>

#include "..\CellShape\CellShape.h"
#include "..\Other\ImageUtility.h"
#include "..\Grid\GridBounds.h"
#include "..\Grid\GridUtility.h"
#include "..\CellShape\CellGroup.h"
#include "ColourDifference.h"
#include "ColourScheme.h"

class PhotomosaicGeneratorBase : public QObject
{
    Q_OBJECT
public:
    PhotomosaicGeneratorBase();
    ~PhotomosaicGeneratorBase();

    //Sets main image
    void setMainImage(const cv::Mat &t_img);

    //Sets library images
    void setLibrary(const std::vector<cv::Mat> &t_lib);

    //Sets colour difference type
    void setColourDifference(const ColourDifference::Type t_type = ColourDifference::Type::RGB_EUCLIDEAN);

    //Sets colour scheme type
    void setColourScheme(const ColourScheme::Type t_type = ColourScheme::Type::NONE);

    //Sets cell group
    void setCellGroup(const CellGroup &t_cellGroup);
    //Gets cell group
    const CellGroup &getCellGroup() const;

    //Sets grid state
    void setGridState(const GridUtility::MosaicBestFit &t_gridState);

    //Sets repeat range and addition
    void setRepeat(int t_repeatRange = 0, int t_repeatAddition = 0);

    //Generate best fits for Photomosaic cells
    //Returns true if successful
    virtual bool generateBestFits();

    //Returns best fits
    GridUtility::MosaicBestFit getBestFits() const;

    //Builds photomosaic from mosaic state
    cv::Mat buildPhotomosaic(const cv::Scalar &t_backgroundColour = cv::Scalar(0, 0, 0)) const;

    //Returns maximum progress
    int getMaxProgress();

public slots:
    //Cancel generation
    void cancel();

signals:
    //Emitted when progress changes
    void progress(const int t_progressStep);

protected:
    int m_progress;
    bool m_wasCanceled;

    cv::Mat m_img;
    std::vector<cv::Mat> m_lib;

    ColourDifference::Type m_colourDiffType;
    ColourDifference::FunctionType m_colourDiffFunc;
    ColourScheme::Type m_colourSchemeType;
    ColourScheme::FunctionType m_colourSchemeFunc;

    CellGroup m_cells;

    //Represents grid state and best fits for Photomosaic cells (after generate)
    GridUtility::MosaicBestFit m_bestFits;

    int m_repeatRange, m_repeatAddition;

    //Performs preprocessing steps on main image: resize, create variants (colour theory), convert colour space
    std::vector<cv::Mat> preprocessMainImage();

    //Performs preprocessing steps on library images: resize, convert colour space
    std::vector<cv::Mat> preprocessLibraryImages();

    //Returns the cell image at given position and it's local bounds
    std::pair<std::vector<cv::Mat>, cv::Rect> getCellAt(
        const CellShape &t_cellShape, const CellShape &t_detailCellShape,
        const int x, const int y, const std::vector<cv::Mat> &t_mains) const;
};