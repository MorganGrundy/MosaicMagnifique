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

#include <QPainter>
#include <QDebug>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <QWheelEvent>

#include "gridutility.h"
#include "imageutility.h"
#include "gridgenerator.h"

GridViewer::GridViewer(QWidget *parent)
    : QWidget(parent), m_cells{}, MIN_ZOOM{0.5}, MAX_ZOOM{10}, zoom{1}
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

//Gets grid state of current options and creates grid
void GridViewer::updateGrid()
{
    //Height & Width to use when no back image
    const int viewerHeight = height() / MIN_ZOOM;
    const int viewerWidth = width() / MIN_ZOOM;
    gridState = GridGenerator::getGridState(m_cells, backImage, viewerHeight, viewerWidth);

    //Calculate grid size
    const int gridHeight = (backImage.empty()) ? viewerHeight : backImage.rows;
    const int gridWidth = (backImage.empty()) ? viewerWidth : backImage.cols;

    createGrid(gridHeight, gridWidth);
    update();
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

//Returns state of current grid
GridUtility::mosaicBestFit GridViewer::getGridState() const
{
    return gridState;
}

//Called when the spinbox value is changed, updates grid zoom
void GridViewer::zoomChanged(double t_value)
{
    zoom = t_value / 100.0;
    update();
}

//Changes if grid preview shows edge detected or normal cells
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

//Updates display of grid
//If no back image then creates new grid
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
    zoom += event->angleDelta().y() /
            ((event->modifiers().testFlag(Qt::ControlModifier)) ? 1200.0 : 12000.0);
    zoom = std::clamp(zoom, MIN_ZOOM, MAX_ZOOM);

    spinZoom->blockSignals(true);
    spinZoom->setValue(zoom * 100);
    spinZoom->blockSignals(false);

    update();
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
                    auto [flipHorizontal, flipVertical] = GridUtility::getFlipStateAt(
                        m_cells.getCell(step), x, y, GridUtility::PAD_GRID);

                    //Create bounded mask
                    const cv::Mat mask(m_cells.getCell(step).getCellMask(flipHorizontal,
                                                                         flipVertical),
                                       cv::Range(yStart - unboundedRect.y, yEnd - unboundedRect.y),
                                       cv::Range(xStart - unboundedRect.x, xEnd - unboundedRect.x));

                    //Create bounded edge mask
                    const cv::Mat edgeMask(m_cells.getEdgeCell(step, flipHorizontal, flipVertical),
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

    grid = QImage(newGrid.at(0).data, gridWidth, gridHeight, static_cast<int>(newGrid.at(0).step),
                  QImage::Format_RGBA8888).copy();
    edgeGrid = QImage(newEdgeGrid.at(0).data, gridWidth, gridHeight,
                      static_cast<int>(newEdgeGrid.at(0).step), QImage::Format_RGBA8888).copy();
}
