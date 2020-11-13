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

    switchGridColour = new Switch(this);
    switchGridColour->setText("White", Switch::SwitchState::LEFT);
    switchGridColour->setColour(Qt::white, Switch::SwitchState::LEFT);
    switchGridColour->setText("Black", Switch::SwitchState::RIGHT);
    switchGridColour->setColour(Qt::black, Switch::SwitchState::RIGHT);
    connect(switchGridColour, &QAbstractButton::toggled, this, &GridViewer::gridColourChanged);
    layout->addWidget(switchGridColour, 0, 0);

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
    if (switchGridColour->getState() == Switch::SwitchState::RIGHT)
    {
        if (!blackGrid.isNull())
        {
            sceneGrid->setPixmap(blackGrid);

            //Update scene rect
            if (newSceneRect.width() == 0)
            {
                newSceneRect.setWidth(blackGrid.width());
                newSceneRect.setHeight(blackGrid.height());
            }
        }
    }
    else if (!whiteGrid.isNull())
    {
        sceneGrid->setPixmap(whiteGrid);

        //Update scene rect
        if (newSceneRect.width() == 0)
        {
            newSceneRect.setWidth(whiteGrid.width());
            newSceneRect.setHeight(whiteGrid.height());
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

//Switches between a white and black grid
void GridViewer::gridColourChanged([[maybe_unused]] bool t_state)
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
    std::vector<cv::Mat> gridMaskParts, gridParts;

    //For all size steps in results
    for (size_t step = 0; step < gridState.size(); ++step)
    {
        //Create new grids
        gridMaskParts.push_back(cv::Mat(gridHeight, gridWidth, CV_8UC1, cv::Scalar(0)));
        gridParts.push_back(cv::Mat(gridHeight, gridWidth, CV_8UC4, cv::Scalar(0, 0, 0, 0)));

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

                    cv::Mat gridPart(gridMaskParts.at(step), roi);
                    cv::Mat edgeGridPart(gridParts.at(step), roi);

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
    for (size_t i = gridMaskParts.size() - 1; i > 0; --i)
    {
        cv::Mat mask;
        cv::bitwise_not(gridMaskParts.at(i - 1), mask);

        cv::bitwise_or(gridMaskParts.at(i - 1), gridMaskParts.at(i), gridMaskParts.at(i - 1), mask);
        cv::bitwise_or(gridParts.at(i - 1), gridParts.at(i), gridParts.at(i - 1), mask);
    }

    //Copy grid and set to black
    cv::Mat newBlackGrid = gridParts.at(0).clone();
    int nRows = newBlackGrid.rows;
    int nCols = newBlackGrid.cols;
    if (newBlackGrid.isContinuous())
    {
        nCols *= nRows;
        nRows = 1;
    }
    cv::Vec4b *p;
    for (int i = 0; i < nRows; ++i)
    {
        p = newBlackGrid.ptr<cv::Vec4b>(i);
        for (int j = 0; j < nCols; ++j)
        {
            if (p[j][3] != 0)
            {
                p[j][0] = 0;
                p[j][1] = 0;
                p[j][2] = 0;
            }
        }
    }

    //Convert mat grids to QPixmap
    whiteGrid = ImageUtility::matToQPixmap(gridParts.at(0), QImage::Format_RGBA8888);
    blackGrid = ImageUtility::matToQPixmap(newBlackGrid, QImage::Format_RGBA8888);
}
