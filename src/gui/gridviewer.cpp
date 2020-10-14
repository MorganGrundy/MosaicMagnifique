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

#include "gridviewer.h"

#include <opencv2/imgproc.hpp>

#include "gridutility.h"
#include "imageutility.h"
#include "gridgenerator.h"

GridViewer::GridViewer(QWidget *parent)
    : CustomGraphicsView(parent), m_cells{}, scene{nullptr}
{
    layout = new QGridLayout(this);

    checkEdgeDetect = new QCheckBox("Edge Detect:", this);
    checkEdgeDetect->setLayoutDirection(Qt::LayoutDirection::RightToLeft);
    checkEdgeDetect->setStyleSheet("QWidget {"
                                   "background-color: rgb(60, 60, 60);"
                                   "color: rgb(255, 255, 255);"
                                   "border-color: rgb(0, 0, 0);"
                                   "}");
    checkEdgeDetect->setCheckState(Qt::Checked);
    connect(checkEdgeDetect, &QCheckBox::stateChanged, this, &GridViewer::edgeDetectChanged);
    layout->addWidget(checkEdgeDetect, 0, 0);

    hSpacer = new QSpacerItem(10, 10, QSizePolicy::Expanding, QSizePolicy::Minimum);
    layout->addItem(hSpacer, 0, 1);

    vSpacer = new QSpacerItem(10, 10, QSizePolicy::Minimum, QSizePolicy::Expanding);
    layout->addItem(vSpacer, 1, 0);

    //Create new scene
    scene = new QGraphicsScene(this);
    scene->setItemIndexMethod(QGraphicsScene::NoIndex);
    setScene(scene);

    //Create scene item for background image
    sceneBackground = new QGraphicsPixmapItem();
    scene->addItem(sceneBackground);

    //Create scene item for grid image
    sceneGrid = new QGraphicsPixmapItem();
    scene->addItem(sceneGrid);
}

GridViewer::~GridViewer()
{
    if (scene != nullptr)
        delete scene;
}

//Changes state of edge detection in grid preview
void GridViewer::setEdgeDetect(bool t_state)
{
    checkEdgeDetect->setChecked(t_state);
}

//Gets grid state of current options and creates grid
void GridViewer::updateGrid()
{
    //Height & Width to use when no back image
    const int viewerHeight = viewport()->rect().height();
    const int viewerWidth = viewport()->rect().width();
    gridState = GridGenerator::getGridState(m_cells, backImage, viewerHeight, viewerWidth);

    //Calculate grid size
    const int gridHeight = (backImage.empty()) ? viewerHeight : backImage.rows;
    const int gridWidth = (backImage.empty()) ? viewerWidth : backImage.cols;

    createGrid(gridHeight, gridWidth);
    updateView();
}

//Clears scene and sets new background and grid
void GridViewer::updateView(bool t_updateTransform)
{
    //Stores rect of new scene
    QRectF newSceneRect(0, 0, 0, 0);

    //Set background image
    if (!background.isNull())
    {
        sceneBackground->setPixmap(background);
        //Update scene rect
        newSceneRect.setWidth(background.width());
        newSceneRect.setHeight(background.height());
    }

    //Set grid image
    if (checkEdgeDetect->isChecked())
    {
        if (!edgeGrid.isNull())
        {
            sceneGrid->setPixmap(edgeGrid);

            //Update scene rect
            if (newSceneRect.width() == 0)
            {
                newSceneRect.setWidth(edgeGrid.width());
                newSceneRect.setHeight(edgeGrid.height());
            }
        }
    }
    else if (!grid.isNull())
    {
        sceneGrid->setPixmap(grid);

        //Update scene rect
        if (newSceneRect.width() == 0)
        {
            newSceneRect.setWidth(grid.width());
            newSceneRect.setHeight(grid.height());
        }
    }

    //Set new scene rect
    setSceneRect(newSceneRect);

    //Fits scene to view
    if (t_updateTransform)
        fitToView();
}

//Sets cell group
void GridViewer::setCellGroup(const CellGroup &t_cellGroup)
{
    m_cells = t_cellGroup;
}

//Returns reference to cell group
CellGroup &GridViewer::getCellGroup()
{
    return m_cells;
}

//Sets the background image in grid
void GridViewer::setBackground(const cv::Mat &t_background)
{
    backImage = t_background;
    if (t_background.empty())
        background = QPixmap();
    else
        background = ImageUtility::matToQPixmap(t_background);
}

//Returns background image
const cv::Mat GridViewer::getBackground() const
{
    return backImage;
}

//Sets grid state and updates grid
void GridViewer::setGridState(const GridUtility::MosaicBestFit &t_gridState)
{
    gridState = t_gridState;

    //Height & Width to use when no back image
    const int viewerHeight = viewport()->rect().height();
    const int viewerWidth = viewport()->rect().width();

    //Calculate grid size
    const int gridHeight = (backImage.empty()) ? viewerHeight : backImage.rows;
    const int gridWidth = (backImage.empty()) ? viewerWidth : backImage.cols;

    createGrid(gridHeight, gridWidth);
}

//Returns state of current grid
GridUtility::MosaicBestFit GridViewer::getGridState() const
{
    return gridState;
}

//Changes if grid preview shows edge detected or normal cells
void GridViewer::edgeDetectChanged([[maybe_unused]] int t_state)
{
    updateView();
}

//Updates display of grid
//If no back image then creates new grid
void GridViewer::resizeEvent([[maybe_unused]] QResizeEvent *event)
{
    if (background.isNull())
        updateGrid();
}

//Creates grid from grid state and cells
void GridViewer::createGrid(const int gridHeight, const int gridWidth)
{
    std::vector<cv::Mat> newGrid, newEdgeGrid;

    //For all size steps in results
    for (size_t step = 0; step < gridState.size(); ++step)
    {
        //Create new grids
        newGrid.push_back(cv::Mat(gridHeight, gridWidth, CV_8UC1, cv::Scalar(0)));
        newEdgeGrid.push_back(cv::Mat(gridHeight, gridWidth, CV_8UC4, cv::Scalar(0, 0, 0, 0)));

        //For all cells
        for (int y = -GridUtility::PAD_GRID;
             y < static_cast<int>(gridState.at(step).size()) - GridUtility::PAD_GRID; ++y)
        {
            for (int x = -GridUtility::PAD_GRID;
                 x < static_cast<int>(gridState.at(step).at(y + GridUtility::PAD_GRID).size())
                         - GridUtility::PAD_GRID; ++x)
            {
                //Cell in valid state
                if (gridState.at(step).at(y + GridUtility::PAD_GRID).
                    at(x + GridUtility::PAD_GRID).has_value())
                {
                    const cv::Rect unboundedRect = GridUtility::getRectAt(m_cells.getCell(step), x, y);

                    //Bound cell within grid dimensions
                    const int yStart = std::clamp(unboundedRect.y, 0, gridHeight);
                    const int yEnd = std::clamp(unboundedRect.br().y, 0, gridHeight);
                    const int xStart = std::clamp(unboundedRect.x, 0, gridWidth);
                    const int xEnd = std::clamp(unboundedRect.br().x, 0, gridWidth);

                    //Cell in bounds
                    if (yStart == yEnd || xStart == xEnd)
                        continue;

                    const cv::Rect roi = cv::Rect(xStart, yStart, xEnd - xStart, yEnd - yStart);

                    cv::Mat gridPart(newGrid.at(step), roi);
                    cv::Mat edgeGridPart(newEdgeGrid.at(step), roi);

                    //Calculate if and how current cell is flipped
                    const auto flipState = GridUtility::getFlipStateAt(m_cells.getCell(step), x, y,
                                                                       GridUtility::PAD_GRID);

                    //Create bounded mask
                    const cv::Mat mask(
                        m_cells.getCell(step).getCellMask(flipState.horizontal, flipState.vertical),
                        cv::Range(yStart - unboundedRect.y, yEnd - unboundedRect.y),
                        cv::Range(xStart - unboundedRect.x, xEnd - unboundedRect.x));

                    //Create bounded edge mask
                    const cv::Mat edgeMask(
                        m_cells.getEdgeCell(step, flipState.horizontal, flipState.vertical),
                        cv::Range(yStart - unboundedRect.y, yEnd - unboundedRect.y),
                        cv::Range(xStart - unboundedRect.x, xEnd - unboundedRect.x));

                    //Copy cell to grid
                    cv::bitwise_or(gridPart, mask, gridPart);
                    cv::bitwise_or(edgeGridPart, edgeMask, edgeGridPart);
                }
            }
        }
    }

    //Combine grids
    for (size_t i = newGrid.size() - 1; i > 0; --i)
    {
        cv::Mat mask;
        cv::bitwise_not(newGrid.at(i - 1), mask);

        cv::bitwise_or(newGrid.at(i - 1), newGrid.at(i), newGrid.at(i - 1), mask);
        cv::bitwise_or(newEdgeGrid.at(i - 1), newEdgeGrid.at(i), newEdgeGrid.at(i - 1), mask);
    }

    //Make black pixels transparent
    ImageUtility::matMakeTransparent(newGrid.at(0), newGrid.at(0), 0);

    grid = ImageUtility::matToQPixmap(newGrid.at(0), QImage::Format_RGBA8888);
    edgeGrid = ImageUtility::matToQPixmap(newEdgeGrid.at(0), QImage::Format_RGBA8888);
}
