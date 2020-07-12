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

#ifndef PHOTOMOSAICGENERATOR_H
#define PHOTOMOSAICGENERATOR_H

#include <QProgressDialog>
#include <opencv2/core/mat.hpp>

#include "cellshape.h"
#include "utilityfuncs.h"
#include "gridbounds.h"
#include "cellgrid.h"

class PhotomosaicGenerator : private QProgressDialog
{
    Q_OBJECT
public:
    enum class Mode {RGB_EUCLIDEAN, CIE76, CIEDE2000};

    PhotomosaicGenerator(QWidget *t_parent = nullptr);
    ~PhotomosaicGenerator();

    void setMainImage(const cv::Mat &t_img);
    void setLibrary(const std::vector<cv::Mat> &t_lib);
    void setDetail(const int t_detail = 100);
    void setMode(const Mode t_mode = Mode::RGB_EUCLIDEAN);

    void setCellShape(const CellShape &t_cellShape);
    void setGridState(const CellGrid::mosaicBestFit &t_gridState);
    void setSizeSteps(const size_t t_steps);

    void setRepeat(int t_repeatRange = 0, int t_repeatAddition = 0);

    cv::Mat generate();
#ifdef CUDA
    cv::Mat cudaGenerate();
#endif

private:
    cv::Mat m_img;
    std::vector<cv::Mat> m_lib;

    double m_detail;
    Mode m_mode;

    CellShape m_cellShape;
    CellGrid::mosaicBestFit m_gridState;
    size_t sizeSteps;

    int m_repeatRange, m_repeatAddition;

    std::pair<cv::Mat, std::vector<cv::Mat>> resizeAndCvtColor();
    void resizeImages(std::vector<cv::Mat> &t_images, const double t_ratio = 0.5);

    std::optional<size_t> findCellBestFit(
        const CellShape &t_cellShape,
        const CellShape &t_detailCellShape,
        const int x, const int y, const bool t_pad,
        const cv::Mat &t_image, const std::vector<cv::Mat> &t_lib,
        const CellGrid::stepBestFit &t_grid) const;

    std::pair<cv::Mat, cv::Rect> getCellAt(
        const CellShape &t_cellShape, const CellShape &t_detailCellShape,
        const int x, const int y, const bool t_pad,
        const cv::Mat &t_image) const;

    int findBestFitEuclidean(const cv::Mat &cell, const cv::Mat &mask,
                             const std::vector<cv::Mat> &library,
                             const std::map<size_t, int> &repeats, const cv::Rect &t_bounds) const;
    int findBestFitCIEDE2000(const cv::Mat &cell, const cv::Mat &mask,
                             const std::vector<cv::Mat> &library,
                             const std::map<size_t, int> &repeats, const cv::Rect &t_bounds) const;

    std::map<size_t, int> calculateRepeats(const CellGrid::stepBestFit &grid,
                                           const int x, const int y) const;

    double degToRad(const double deg) const;
    cv::Mat combineResults(const CellGrid::mosaicBestFit &results);
};

#endif // PHOTOMOSAICGENERATOR_H
