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

#ifndef GRIDVIEWER_H
#define GRIDVIEWER_H

#include <QWidget>
#include <opencv2/core/mat.hpp>
#include <QGridLayout>
#include <QLabel>
#include <QDoubleSpinBox>
#include <QCheckBox>
#include <QSpacerItem>

#include "cellshape.h"
#include "gridbounds.h"
#include "gridutility.h"
#include "cellgroup.h"

class GridViewer : public QWidget
{
    Q_OBJECT
public:
    explicit GridViewer(QWidget *parent = nullptr);
    //Changes state of edge detection in grid preview
    void setEdgeDetect(bool t_state);

    //Gets grid state of current options and creates grid
    void updateGrid();

    //Returns reference to cell group
    CellGroup &getCellGroup();

    //Sets the background image in grid
    void setBackground(const cv::Mat &t_background);

    //Returns state of current grid
    GridUtility::mosaicBestFit getGridState() const;

public slots:
    //Called when the spinbox value is changed, updates grid zoom
    void zoomChanged(double t_value);

    //Changes if grid preview shows edge detected or normal cells
    void edgeDetectChanged(int t_state);

protected:
    //Displays grid
    void paintEvent(QPaintEvent *event) override;

    //Updates display of grid
    //If no back image then creates new grid
    void resizeEvent(QResizeEvent *event) override;

    //Change zoom of grid preview based on mouse scrollwheel movement
    //Ctrl is a modifier key that allows for faster zooming (x10)
    void wheelEvent(QWheelEvent *event) override;

private:
    //Creates grid from grid state and cells
    void createGrid(const int gridHeight, const int gridWidth);

    QGridLayout *layout;
    QLabel *labelZoom;
    QDoubleSpinBox *spinZoom;
    QCheckBox *checkEdgeDetect;
    QSpacerItem *hSpacer, *vSpacer;

    cv::Mat backImage;
    QImage background;

    CellGroup m_cells;

    QImage grid, edgeGrid;

    const double MIN_ZOOM, MAX_ZOOM;
    double zoom;

    GridUtility::mosaicBestFit gridState;
};

#endif // GRIDVIEWER_H
