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

#include <opencv2/core/mat.hpp>
#include <QGridLayout>
#include <QCheckBox>
#include <QSpacerItem>
#include <QGraphicsPixmapItem>

#include "customgraphicsview.h"
#include "cellshape.h"
#include "gridbounds.h"
#include "gridutility.h"
#include "cellgroup.h"

class GridViewer : public CustomGraphicsView
{
    Q_OBJECT
public:
    explicit GridViewer(QWidget *parent = nullptr);
    ~GridViewer();

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
    //Returns background image
    const cv::Mat getBackground() const;

    //Sets grid state and updates grid
    void setGridState(const GridUtility::MosaicBestFit &t_gridState);
    //Returns state of current grid
    GridUtility::MosaicBestFit getGridState() const;

public slots:
    //Switches between a white and black grid
    void gridColorToggle(int t_state);

protected:
    //Updates display of grid
    //If no back image then creates new grid
    void resizeEvent(QResizeEvent *event) override;

    //Creates grid from grid state and cells
    void createGrid(const int gridHeight, const int gridWidth);

    CellGroup m_cells;
    GridUtility::MosaicBestFit gridState;

    cv::Mat backImage;

    QGraphicsScene *scene;

private:
    QGridLayout *layout;
    QCheckBox *checkGridColor;
    QSpacerItem *hSpacer, *vSpacer;

    QGraphicsPixmapItem *sceneBackground;
    QGraphicsPixmapItem *sceneGrid;

    QPixmap background;

    QPixmap whiteGrid, blackGrid;
};

#endif // GRIDVIEWER_H
