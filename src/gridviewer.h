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
#include <QGraphicsPixmapItem>

#include "cellshape.h"
#include "gridbounds.h"
#include "gridutility.h"
#include "cellgroup.h"
#include "customgraphicsview.h"

class GridViewer : public CustomGraphicsView
{
    Q_OBJECT
public:
    explicit GridViewer(QWidget *parent = nullptr);
    ~GridViewer();

    //Changes state of edge detection in grid preview
    void setEdgeDetect(bool t_state);

    //Gets grid state of current options and creates grid
    void updateGrid();

    //Clears scene and sets new background and grid
    void updateView(bool t_updateTransform = true);

    //Sets cell group
    void setCellGroup(const CellGroup &t_cellGroup);
    //Returns reference to cell group
    CellGroup &getCellGroup();

    //Sets the background image in grid
    void setBackground(const cv::Mat &t_background);

    //Sets grid state and updates grid
    void setGridState(const GridUtility::mosaicBestFit &t_gridState);
    //Returns state of current grid
    GridUtility::mosaicBestFit getGridState() const;

public slots:
    //Changes if grid preview shows edge detected or normal cells
    void edgeDetectChanged(int t_state);

protected:
    //Updates display of grid
    //If no back image then creates new grid
    void resizeEvent(QResizeEvent *event) override;

    //Creates grid from grid state and cells
    void createGrid(const int gridHeight, const int gridWidth);

    CellGroup m_cells;
    GridUtility::mosaicBestFit gridState;

    cv::Mat backImage;

private:
    QGridLayout *layout;
    QCheckBox *checkEdgeDetect;
    QSpacerItem *hSpacer, *vSpacer;

    QGraphicsScene *scene;
    QGraphicsPixmapItem *sceneBackground;
    QGraphicsPixmapItem *sceneGrid;

    QPixmap background;

    QPixmap grid, edgeGrid;
};

#endif // GRIDVIEWER_H
