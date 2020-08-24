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

#include "grideditviewer.h"

#include <QDebug>

GridEditViewer::GridEditViewer(QWidget *parent) : GridViewer(parent), m_sizeStep{0}
{}

//Sets current size step for editor
void GridEditViewer::setSizeStep(const size_t t_sizeStep)
{
    m_sizeStep = t_sizeStep;
}

//Inverts state of clicked cell
void GridEditViewer::mousePressEvent(QMouseEvent *event)
{
    //If left mouse button pressed
    if (event->buttons() == Qt::MouseButton::LeftButton)
    {
        //Transform click position to grid position
        QPoint gridPos = mapToScene(event->pos()).toPoint();

        //Stores all cells (x,y) at click position
        std::vector<std::pair<int, int>> cellsAtClick;

        //For all cells (at top size step)
        for (int y = -GridUtility::PAD_GRID;
             y < static_cast<int>(gridState.at(m_sizeStep).size()) - GridUtility::PAD_GRID; ++y)
        {
            for (int x = -GridUtility::PAD_GRID;
                 x < static_cast<int>(gridState.at(m_sizeStep).at(y + GridUtility::PAD_GRID).size())
                         - GridUtility::PAD_GRID; ++x)
            {
                //Get cell bounds
                cv::Rect cellBound = GridUtility::getRectAt(m_cells.getCell(m_sizeStep), x, y);

                //Check if click in bounds
                if ((gridPos.x() >= cellBound.x && gridPos.x() < cellBound.br().x) &&
                    (gridPos.y() >= cellBound.y && gridPos.y() < cellBound.br().y))
                {
                    //Transform grid position to cell position
                    QPoint cellPos(gridPos.x() - cellBound.x, gridPos.y() - cellBound.y);

                    //Check if click is in active cell area
                    auto cellFlip = GridUtility::getFlipStateAt(m_cells.getCell(m_sizeStep), x, y,
                                                                GridUtility::PAD_GRID);
                    if (m_cells.getCell(m_sizeStep).getCellMask(cellFlip.first, cellFlip.second).
                        at<uchar>(cellPos.x(), cellPos.y()) != 0)
                    {
                        //Add to vector
                        cellsAtClick.push_back({x, y});
                    }
                }
            }
        }

        //If a cell was clicked
        if (!cellsAtClick.empty())
        {
            //Sort clicked cells in decreasing y, decreasing x
            std::sort(cellsAtClick.begin(), cellsAtClick.end(),
                      [](const std::pair<int, int> &cell1, const std::pair<int, int> &cell2)
                      {
                          if (cell1.second == cell2.second)
                              return (cell1.first > cell2.first);
                          else
                              return (cell1.second > cell2.second);
                      });

            //Get cell state of cell with highest y,x
            GridUtility::cellBestFit &cellState =
                gridState.at(m_sizeStep).at(cellsAtClick.front().second + GridUtility::PAD_GRID).
                at(cellsAtClick.front().first + GridUtility::PAD_GRID);

            //Toggle cell state
            if (cellState.has_value())
                cellState = std::nullopt;
            else
                cellState = 0;

            //Update grid view
            createGrid(backImage.rows, backImage.cols);
            updateView();
        }
    }
}
