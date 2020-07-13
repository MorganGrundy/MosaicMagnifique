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

#include "gridviewer.h"

#include <QPainter>
#include <QDebug>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <QWheelEvent>

#include "cellgrid.h"
#include "utilityfuncs.h"

GridViewer::GridViewer(QWidget *parent)
    : QWidget(parent), sizeSteps{0}, detail{1.0}, cells{1},
      detailCells{1},
      edgeCells{std::vector<cv::Mat>(4)},
      MIN_ZOOM{0.5}, MAX_ZOOM{10}, zoom{1}
{
    layout = new QGridLayout(this);

    labelZoom = new QLabel("Zoom:", this);
    labelZoom->setStyleSheet("QWidget {"
                             "background-color: rgb(60, 60, 60);"
                             "color: rgb(255, 255, 255);"
                             "border-color: rgb(0, 0, 0);"
                             "}");
    layout->addWidget(labelZoom, 0, 0);

    spinZoom = new QDoubleSpinBox(this);
    spinZoom->setStyleSheet("QWidget {"
                           "background-color: rgb(60, 60, 60);"
                           "color: rgb(255, 255, 255);"
                           "}"
                           "QDoubleSpinBox {"
                           "border: 1px solid dimgray;"
                           "}");
    spinZoom->setRange(MIN_ZOOM * 100, MAX_ZOOM * 100);
    spinZoom->setValue(zoom * 100);
    spinZoom->setSuffix("%");
    spinZoom->setButtonSymbols(QDoubleSpinBox::PlusMinus);
    connect(spinZoom, SIGNAL(valueChanged(double)), this, SLOT(zoomChanged(double)));
    layout->addWidget(spinZoom, 0, 1);

    checkEdgeDetect = new QCheckBox("Edge Detect:", this);
    checkEdgeDetect->setLayoutDirection(Qt::LayoutDirection::RightToLeft);
    checkEdgeDetect->setStyleSheet("QWidget {"
                                   "background-color: rgb(60, 60, 60);"
                                   "color: rgb(255, 255, 255);"
                                   "border-color: rgb(0, 0, 0);"
                                   "}");
    checkEdgeDetect->setCheckState(Qt::Checked);
    connect(checkEdgeDetect, SIGNAL(stateChanged(int)), this, SLOT(edgeDetectChanged(int)));
    layout->addWidget(checkEdgeDetect, 0, 2);

    hSpacer = new QSpacerItem(10, 10, QSizePolicy::Expanding, QSizePolicy::Minimum);
    layout->addItem(hSpacer, 0, 3);

    vSpacer = new QSpacerItem(10, 10, QSizePolicy::Minimum, QSizePolicy::Expanding);
    layout->addItem(vSpacer, 1, 0);
}

//Changes state of edge detection in grid preview
void GridViewer::setEdgeDetect(bool t_state)
{
    checkEdgeDetect->setChecked(t_state);
}

//Finds state of cell at current position and step in detail image
std::pair<CellGrid::cellBestFit, bool> GridViewer::findCellState(const int x, const int y,
                                                                 const GridBounds &t_bounds,
                                                                 const size_t t_step) const
{
    const cv::Rect unboundedRect = CellGrid::getRectAt(cells.at(t_step), x, y);

    //Cell bounded positions
    int yStart, yEnd, xStart, xEnd;

    //Check that cell is within a bound
    bool inBounds = false;
    for (auto it = t_bounds.cbegin(); it != t_bounds.cend() && !inBounds; ++it)
    {
        yStart = std::clamp(unboundedRect.y, it->y, it->br().y);
        yEnd = std::clamp(unboundedRect.br().y, it->y, it->br().y);
        xStart = std::clamp(unboundedRect.x, it->x, it->br().x);
        xEnd = std::clamp(unboundedRect.br().x, it->x, it->br().x);

        //Cell in bounds
        if (yStart != yEnd && xStart != xEnd)
            inBounds = true;
    }

    //Cell completely out of bounds, just skip
    if (!inBounds)
        return {std::nullopt, false};

    //If cell not at lowest size
    if (!backImage.empty() && t_step < sizeSteps)
    {
        //Cell bounded positions (in image)
        yStart = std::clamp(unboundedRect.tl().y, 0, backImage.rows);
        yEnd = std::clamp(unboundedRect.br().y, 0, backImage.rows);
        xStart = std::clamp(unboundedRect.tl().x, 0, backImage.cols);
        xEnd = std::clamp(unboundedRect.br().x, 0, backImage.cols);

        const cv::Rect boundedRect(xStart - unboundedRect.x, yStart - unboundedRect.y,
                                   xEnd - xStart, yEnd - yStart);

        //Copies visible part of image to cell
        cv::Mat cell(backImage, cv::Range(yStart, yEnd), cv::Range(xStart, xEnd));

        //Calculate if and how current cell is flipped
        auto [flipHorizontal, flipVertical] = CellGrid::getFlipStateAt(cells.at(t_step),
                                                                       x, y, CellGrid::PAD_GRID);

        //Resizes bounded rect for detail cells
        const cv::Rect boundedDetailRect(boundedRect.x * detail, boundedRect.y * detail,
                boundedRect.width * detail, boundedRect.height * detail);

        //Create bounded mask of detail cell
        const cv::Mat mask(detailCells.at(t_step).getCellMask(flipHorizontal, flipVertical),
                           boundedDetailRect);

        //Resize image cell to size of mask
        cell = UtilityFuncs::resizeImage(cell, mask.rows, mask.cols,
                UtilityFuncs::ResizeType::EXCLUSIVE);

        //If cell entropy exceeds threshold return true
        if (CellGrid::calculateEntropy(mask, cell) >= CellGrid::MAX_ENTROPY() * 0.7)
            return {std::nullopt, true};
    }

    //Cell is valid
    return {0, false};
}

//Creates grid from states
void GridViewer::createGrid(const CellGrid::mosaicBestFit &states,
                            const int gridHeight, const int gridWidth)
{
    std::vector<cv::Mat> newGrid, newEdgeGrid;

    //For all size steps in results
    for (size_t step = 0; step < states.size(); ++step)
    {
        //Create new grids
        newGrid.push_back(cv::Mat(gridHeight, gridWidth, CV_8UC1, cv::Scalar(0)));
        newEdgeGrid.push_back(cv::Mat(gridHeight, gridWidth, CV_8UC4, cv::Scalar(0, 0, 0, 0)));

        //For all cells
        for (int y = -CellGrid::PAD_GRID;
             y < static_cast<int>(states.at(step).size()) - CellGrid::PAD_GRID; ++y)
        {
            for (int x = -CellGrid::PAD_GRID;
                 x < static_cast<int>(states.at(step).at(y + CellGrid::PAD_GRID).size())
                         - CellGrid::PAD_GRID; ++x)
            {
                //Cell in valid state
                if (states.at(step).at(y + CellGrid::PAD_GRID).
                    at(x + CellGrid::PAD_GRID).has_value())
                {
                    const cv::Rect unboundedRect = CellGrid::getRectAt(cells.at(step), x, y);

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
                    auto [flipHorizontal, flipVertical] = CellGrid::getFlipStateAt(cells.at(step),
                                                                        x, y, CellGrid::PAD_GRID);

                    //Create bounded mask
                    const cv::Mat mask(cells.at(step).getCellMask(flipHorizontal, flipVertical),
                                       cv::Range(yStart - unboundedRect.y, yEnd - unboundedRect.y),
                                       cv::Range(xStart - unboundedRect.x, xEnd - unboundedRect.x));

                    //Create bounded edge mask
                    const cv::Mat edgeMask(getEdgeCell(step, flipHorizontal, flipVertical),
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
    UtilityFuncs::matMakeTransparent(newGrid.at(0), newGrid.at(0), 0);

    grid = QImage(newGrid.at(0).data, gridWidth, gridHeight, static_cast<int>(newGrid.at(0).step),
                  QImage::Format_RGBA8888).copy();
    edgeGrid = QImage(newEdgeGrid.at(0).data, gridWidth, gridHeight,
                      static_cast<int>(newEdgeGrid.at(0).step), QImage::Format_RGBA8888).copy();
}

//Generates grid preview
void GridViewer::updateGrid()
{
    gridState.clear();
    //No cell mask, no grid
    if (cells.at(0).getCellMask(0, 0).empty())
    {
        grid = QImage();
        return;
    }

    //Calculate grid size
    const int gridHeight = (!background.isNull()) ? backImage.rows
                                                  : std::floor(height() / MIN_ZOOM);
    const int gridWidth = (!background.isNull()) ? backImage.cols
                                                 : std::floor(width() / MIN_ZOOM);

    //Stores grid bounds starting with full grid size
    std::vector<GridBounds> bounds(2);
    //Determines which bound is active
    int activeBound = 0;
    bounds.at(activeBound).addBound(gridHeight, gridWidth);

    for (size_t step = 0; step <= sizeSteps && !bounds.at(activeBound).empty(); ++step)
    {
        const cv::Point gridSize = CellGrid::calculateGridSize(cells.at(step),
                                                               gridWidth, gridHeight,
                                                               CellGrid::PAD_GRID);

        gridState.push_back(CellGrid::stepBestFit(static_cast<size_t>(gridSize.y),
                            std::vector<CellGrid::cellBestFit>(static_cast<size_t>(gridSize.x))));

        //Clear previous bounds
        bounds.at(!activeBound).clear();
        //Create all cells in grid
        for (int y = -CellGrid::PAD_GRID; y < gridSize.y - CellGrid::PAD_GRID; ++y)
        {
            for (int x = -CellGrid::PAD_GRID; x < gridSize.x - CellGrid::PAD_GRID; ++x)
            {
                //Find cell state
                const auto [bestFit, entropy] = findCellState(x, y, bounds.at(activeBound), step);

                gridState.at(step).at(static_cast<size_t>(y + CellGrid::PAD_GRID)).
                        at(static_cast<size_t>(x + CellGrid::PAD_GRID)) = bestFit;

                //If cell entropy exceeded
                if (entropy)
                {
                    //Get cell bounds
                    cv::Rect cellBounds = CellGrid::getRectAt(cells.at(step), x, y);

                    //Bound cell within grid dimensions
                    int yStart = std::clamp(cellBounds.y, 0, gridHeight);
                    int yEnd = std::clamp(cellBounds.br().y, 0, gridHeight);
                    int xStart = std::clamp(cellBounds.x, 0, gridWidth);
                    int xEnd = std::clamp(cellBounds.br().x, 0, gridWidth);

                    //Bound not in grid, just skip
                    if (yStart == yEnd || xStart == xEnd)
                        continue;

                    //Update cell bounds
                    cellBounds.y = yStart;
                    cellBounds.x = xStart;
                    cellBounds.height = yEnd - yStart;
                    cellBounds.width = xEnd - xStart;

                    //Add to inactive bounds
                    bounds.at(!activeBound).addBound(cellBounds);
                }
            }
        }

        //Swap active and inactive bounds
        activeBound = !activeBound;

        //New bounds
        if (!bounds.at(activeBound).empty())
            bounds.at(activeBound).mergeBounds();
    }

    createGrid(gridState, gridHeight, gridWidth);
    update();
}

//Sets the cell shape
void GridViewer::setCellShape(const CellShape &t_cellShape)
{
    cells.at(0) = t_cellShape;

    if (!cells.at(0).getCellMask(0, 0).empty())
    {
        //Create edge cell
        cv::Mat cellMask;
        UtilityFuncs::edgeDetect(cells.at(0).getCellMask(0, 0), cellMask);

        //Make black pixels transparent
        UtilityFuncs::matMakeTransparent(cellMask, getEdgeCell(0, false, false), 0);

        //Create flipped edge cell
        cv::flip(getEdgeCell(0, false, false), getEdgeCell(0, true, false), 1);
        cv::flip(getEdgeCell(0, false, false), getEdgeCell(0, false, true), 0);
        cv::flip(getEdgeCell(0, false, false), getEdgeCell(0, true, true), -1);

        //Update detail cell
        setDetail(detail, true);
    }
    else
    {
        cells.resize(1);
        detailCells.resize(1);
        edgeCells.resize(1);
    }
}

//Returns reference to an edge cell mask
cv::Mat &GridViewer::getEdgeCell(size_t t_sizeStep, bool t_flipHorizontal, bool t_flipVertical)
{
    return edgeCells.at(t_sizeStep).at(t_flipHorizontal + t_flipVertical * 2);
}

//Returns a reference to the cell shape
CellShape &GridViewer::getCellShape()
{
    return cells.at(0);
}

//Sets the minimum cell size
void GridViewer::setSizeSteps(const size_t t_steps, const bool t_reset)
{
    //Resize vectors
    cells.resize(t_steps + 1);
    detailCells.resize(t_steps + 1);
    edgeCells.resize(t_steps + 1, std::vector<cv::Mat>(4));

    //Create cell masks for each step size
    int cellSize = cells.at(0).getCellMask(0, 0).rows;
    for (size_t step = 1; step <= t_steps; ++step)
    {
        cellSize /= 2;

        //Only create if reset or is a new step size
        if (t_reset || step > sizeSteps)
        {
            //Create normal cell mask
            cells.at(step) = cells.at(step - 1).resized(cellSize, cellSize);

            //Create detail cell mask
            detailCells.at(step) = cells.at(step).resized(cellSize * detail, cellSize * detail);

            //Create edge cell mask
            cv::Mat cellMask;
            UtilityFuncs::edgeDetect(cells.at(step).getCellMask(0, 0), cellMask);
            //Make black pixels transparent
            UtilityFuncs::matMakeTransparent(cellMask, getEdgeCell(step, false, false), 0);

            cv::flip(getEdgeCell(0, false, false), getEdgeCell(step, true, false), 1);
            cv::flip(getEdgeCell(0, false, false), getEdgeCell(step, false, true), 0);
            cv::flip(getEdgeCell(0, false, false), getEdgeCell(step, true, true), -1);
        }
    }
    sizeSteps = t_steps;
}

//Sets the background image in grid
void GridViewer::setBackground(const cv::Mat &t_background)
{
    backImage = t_background;
    if (t_background.empty())
    {
        background = QImage();
    }
    else
    {
        cv::Mat inverted(t_background.rows, t_background.cols, t_background.type());
        cv::cvtColor(t_background, inverted, cv::COLOR_BGR2RGB);

        background = QImage(inverted.data, inverted.cols, inverted.rows,
                            static_cast<int>(inverted.step), QImage::Format_RGB888).copy();
    }
}

//Sets the detail level
void GridViewer::setDetail(const int t_detail, const bool t_reset)
{
    double newDetail = detail;
    if (!t_reset)
        newDetail = (t_detail < 1) ? 0.01 : t_detail / 100.0;

    if (newDetail != detail || t_reset)
    {
        detail = newDetail;

        //Create top level detail cell
        if (!cells.at(0).getCellMask(0, 0).empty())
        {
            const int cellSize = cells.at(0).getCellMask(0, 0).rows;
            detailCells.at(0) = cells.at(0).resized(cellSize * detail, cellSize * detail);
        }

        //Update size steps for all cells
        setSizeSteps(sizeSteps, true);
    }
}

//Returns state of current grid
CellGrid::mosaicBestFit GridViewer::getGridState() const
{
    return gridState;
}

//Called when the spinbox value is changed, updates grid zoom
void GridViewer::zoomChanged(double t_value)
{
    zoom = t_value / 100.0;
    update();
}

//Called when the checkbox state changes
void GridViewer::edgeDetectChanged(int /*t_state*/)
{
    update();
}

//Displays grid
void GridViewer::paintEvent(QPaintEvent * /*event*/)
{
    QPainter painter(this);

    double ratio;
    QRectF sourceRect(0, 0, 0, 0);

    //Draw background
    if (!background.isNull())
    {
        //Calculate ratio between background and GridViewer size
        ratio = background.width() / static_cast<double>(width());
        if (height() * ratio < background.height())
            ratio = background.height() / static_cast<double>(height());

        sourceRect.setSize(QSizeF((ratio * width()) / zoom, (ratio * height()) / zoom));

        painter.drawImage(QRectF(QPointF(0,0), QSizeF(width(), height())), background, sourceRect);
    }
    else
        sourceRect.setSize(QSizeF(width() / zoom, height() / zoom));

    //Draw grid
    if (checkEdgeDetect->isChecked())
    {
        if (!edgeGrid.isNull())
        {
            QImage croppedGrid = edgeGrid.copy(0, 0, background.width(), background.height());
            painter.drawImage(QRectF(QPointF(0,0), QSizeF(width(), height())),
                              croppedGrid, sourceRect);
        }
    }
    else if (!grid.isNull())
    {
        QImage croppedGrid = grid.copy(0, 0, background.width(), background.height());
        painter.drawImage(QRectF(QPointF(0,0), QSizeF(width(), height())), croppedGrid, sourceRect);
    }
}

//Generate new grid with new size
void GridViewer::resizeEvent(QResizeEvent * /*event*/)
{
    if (background.isNull())
        updateGrid();
    else
        update();
}

//Change zoom of grid preview based on mouse scrollwheel movement
//Ctrl is a modifier key that allows for faster zooming (x10)
void GridViewer::wheelEvent(QWheelEvent *event)
{
    zoom += event->delta() / ((event->modifiers().testFlag(Qt::ControlModifier)) ? 1200.0 : 12000.0);
    zoom = std::clamp(zoom, MIN_ZOOM, MAX_ZOOM);

    spinZoom->blockSignals(true);
    spinZoom->setValue(zoom * 100);
    spinZoom->blockSignals(false);

    update();
}
