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

#ifndef IMAGEVIEWER_H
#define IMAGEVIEWER_H

#include <QMainWindow>
#include <opencv2/core/mat.hpp>
#include <QProgressBar>
#include <QGraphicsScene>

#include "cellgroup.h"
#include "gridutility.h"

namespace Ui {
class PhotomosaicViewer;
}

class PhotomosaicViewer : public QMainWindow
{
    Q_OBJECT

public:
    explicit PhotomosaicViewer(QWidget *t_parent, const cv::Mat &t_img,
                               const std::vector<cv::Mat> &t_lib, const CellGroup &t_cells,
                               const GridUtility::MosaicBestFit &t_mosaicState,
                               const double t_duration = 0);
    explicit PhotomosaicViewer(QWidget *t_parent = nullptr);
    ~PhotomosaicViewer();

public slots:
    //Allows user to save the Photomosaic as an image file
    void savePhotomosaic();

private:
    //Builds photomosaic from mosaic state
    void buildPhotomosaic();

    Ui::PhotomosaicViewer *ui;
    QGraphicsScene *scene;

    const cv::Mat m_img;
    const std::vector<cv::Mat> m_lib;

    const CellGroup m_cells;
    const GridUtility::MosaicBestFit m_mosaicState;

    cv::Mat m_photomosaic;
};

#endif // IMAGEVIEWER_H
