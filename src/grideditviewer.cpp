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

GridEditViewer::GridEditViewer(QWidget *parent)
    : GridViewer(parent), m_sizeStep{0}, m_quadtree{}
{}

//Sets current size step for editor
void GridEditViewer::setSizeStep(const size_t t_sizeStep)
{
    m_sizeStep = t_sizeStep;
}

//Gets grid state of current options and creates grid
//Creates quadtree of grid
void GridEditViewer::updateGrid()
{
    //Call grid viewer update grid
    GridViewer::updateGrid();

    //Create a quadtree for each size step
    m_quadtree = std::vector<Quadtree>(gridState.size(),
                                       Quadtree(cv::Rect(0, 0, backImage.cols, backImage.rows)));

    //Insert all cells into quadtrees
    for (size_t step = 0; step < gridState.size(); ++step)
    {
        for (int y = -GridUtility::PAD_GRID;
             y < static_cast<int>(gridState.at(step).size()) - GridUtility::PAD_GRID; ++y)
        {
            for (int x = -GridUtility::PAD_GRID;
                 x < static_cast<int>(gridState.at(step).at(y + GridUtility::PAD_GRID).size())
                         - GridUtility::PAD_GRID; ++x)
            {
                //Get cell bounds
                const cv::Rect cellBound = GridUtility::getRectAt(m_cells.getCell(step), x, y);

                //Insert cell into quadtree
                m_quadtree.at(step).insert({cellBound, cv::Point(x, y)});
            }
        }
    }
}

//Inverts state of clicked cell
void GridEditViewer::mousePressEvent(QMouseEvent *event)
{
    //If left mouse button pressed
    if (event->buttons() == Qt::MouseButton::LeftButton)
    {
        //Transform click position to grid position
        const QPoint qGridPos = mapToScene(event->pos()).toPoint();
        const cv::Point gridPos(qGridPos.x(), qGridPos.y());

        //Get all cells with bounds at click position
        const std::vector<Quadtree::elementType> cellBoundsAtClick =
            m_quadtree.at(m_sizeStep).query(gridPos);

        //Stores all cells at click position
        std::vector<cv::Point> cellsAtClick;

        //For all clicked bounds check if cell area clicked
        for (const auto cell: cellBoundsAtClick)
        {
            //Transform grid position to cell position
            const cv::Point cellPos = gridPos - cell.first.tl();

            //Check if click is in active cell area
            const auto cellFlip = GridUtility::getFlipStateAt(m_cells.getCell(m_sizeStep),
                                                              cell.second.x, cell.second.y,
                                                              GridUtility::PAD_GRID);
            if (m_cells.getCell(m_sizeStep).getCellMask(cellFlip.first, cellFlip.second).
                at<uchar>(cellPos) != 0)
            {
                //Add to vector
                cellsAtClick.push_back(cell.second);
            }
        }

        //If a cell was clicked
        if (!cellsAtClick.empty())
        {
            //Sort clicked cells in decreasing y, decreasing x
            std::sort(cellsAtClick.begin(), cellsAtClick.end(),
                      [](const cv::Point &cell1, const cv::Point &cell2)
                      {
                          if (cell1.y == cell2.y)
                              return (cell1.x > cell2.x);
                          else
                              return (cell1.y > cell2.y);
                      });

            //Get cell state of cell with highest y,x
            GridUtility::cellBestFit &cellState =
                gridState.at(m_sizeStep).at(cellsAtClick.front().y + GridUtility::PAD_GRID).
                at(cellsAtClick.front().x + GridUtility::PAD_GRID);

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
