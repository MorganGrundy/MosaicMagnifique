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

#include "GridEditViewer.h"

GridEditViewer::GridEditViewer(QWidget *parent)
    : GridViewer(parent), m_sizeStep{0}, m_quadtree{}, m_tool{Tool::Single}
{
    //Create scene item for selection rect
    selectionItem = new QGraphicsRectItem(QRectF(0, 0, 0, 0));
    selectionItem->setPen(QColor(0, 200, 255, 250));
    selectionItem->setBrush(QColor(0, 200, 255, 50));
    scene->addItem(selectionItem);
}

GridEditViewer::~GridEditViewer()
{
    delete selectionItem;
}

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

//Sets active tool
void GridEditViewer::setTool(const Tool t_tool)
{
    m_tool = t_tool;
}

//Gets intial click position for selection
void GridEditViewer::mousePressEvent(QMouseEvent *event)
{
    //If selection tool active and left mouse button pressed
    if (m_tool == Tool::Selection && event->button() == Qt::MouseButton::LeftButton)
    {
        //Transform click position to grid position
        m_selectionStart = mapToScene(event->pos()).toPoint();
    }
}

//Displays selection
void GridEditViewer::mouseMoveEvent(QMouseEvent *event)
{
    //If selection tool active and left mouse button pressed
    if (m_tool == Tool::Selection && (event->buttons() & Qt::MouseButton::LeftButton))
    {
        //Update selection rect
        selectionItem->setRect(QRectF(m_selectionStart, mapToScene(event->pos()).toPoint()));
    }
}

//Inverts state of clicked/selected cell
void GridEditViewer::mouseReleaseEvent(QMouseEvent *event)
{
    //Remove selection
    if (selectionItem->rect().width() != 0 && selectionItem->rect().height() != 0)
    {
        selectionItem->setRect(QRectF(0, 0, 0, 0));
    }

    //If left mouse button released
    if (event->button() == Qt::MouseButton::LeftButton)
    {
        //Transform click position to grid position
        const QPoint qGridPos = mapToScene(event->pos()).toPoint();
        const cv::Point gridPos(qGridPos.x(), qGridPos.y());

        //Editing for single tool
        if (m_tool == Tool::Single)
        {
            editSingle(gridPos);
        }
        //Editing for selection tool
        else if (m_tool == Tool::Selection)
        {
            const cv::Point startGridPos(m_selectionStart.x(), m_selectionStart.y());
            cv::Rect selection(startGridPos, gridPos);
            editSelection(selection);
        }
    }
}

//Toggles state of cell at grid position
void GridEditViewer::editSingle(const cv::Point t_gridPos)
{
    //Get all cells with bounds at click position
    const std::vector<Quadtree::elementType> cellBoundsAtClick =
        m_quadtree.at(m_sizeStep).query(t_gridPos);

    //Stores all cells at click position
    std::vector<cv::Point> cellsAtClick;

    //For all clicked bounds check if cell area clicked
    for (const auto &cell: cellBoundsAtClick)
    {
        //Transform grid position to cell position
        const cv::Point cellPos = t_gridPos - cell.first.tl();

        //Check if click is in active cell area
        const auto cellFlip = GridUtility::getFlipStateAt(m_cells.getCell(m_sizeStep),
                                                          cell.second.x, cell.second.y);
        if (m_cells.getCell(m_sizeStep).getCellMask(cellFlip.horizontal, cellFlip.vertical).
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
        GridUtility::CellBestFit &cellState =
            gridState.at(m_sizeStep).at(cellsAtClick.front().y + GridUtility::PAD_GRID).
            at(cellsAtClick.front().x + GridUtility::PAD_GRID);

        //Toggle cell state
        if (cellState.has_value())
            cellState = std::nullopt;
        else
            cellState = 0;

        //Update grid view
        createGrid(backImage.rows, backImage.cols);
        updateView(false);
    }
}

//Toggles state of all cells that intersect rect
void GridEditViewer::editSelection(const cv::Rect t_selectionRect)
{
    //Get all cells with bounds that intersect selection
    const std::vector<Quadtree::elementType> cellBoundsAtSelect =
        m_quadtree.at(m_sizeStep).query(t_selectionRect);

    //Stores all cells that intersect selection
    std::vector<cv::Point> cellsAtSelect;

    //For all selected bounds check if cell area intersects selection
    for (const auto &cell: cellBoundsAtSelect)
    {
        //Get rect of intersection between selection and cell bound
        const cv::Rect selectIntersect = (t_selectionRect & cell.first) - cell.first.tl();

        //Get cell mask
        const auto cellFlip = GridUtility::getFlipStateAt(m_cells.getCell(m_sizeStep),
                                                          cell.second.x, cell.second.y);
        const cv::Mat &cellMask = m_cells.getCell(m_sizeStep).getCellMask(cellFlip.horizontal,
                                                                          cellFlip.vertical);

        //Check that cell mask active area intersects with selection
        bool intersectFound = (selectIntersect.width == cellMask.cols &&
                               selectIntersect.height == cellMask.rows);
        const uchar *p_mask;
        for (int row = selectIntersect.y; row < selectIntersect.br().y && !intersectFound; ++row)
        {
            p_mask = cellMask.ptr<uchar>(row);
            for (int col = selectIntersect.x;
                 col < selectIntersect.br().x && !intersectFound; ++col)
            {
                if (p_mask[col] != 0)
                {
                    intersectFound = true;
                }
            }
        }

        //Add to vector
        if (intersectFound)
            cellsAtSelect.push_back(cell.second);
    }

    //If a cell intersects with selection
    if (!cellsAtSelect.empty())
    {
        //Toggle state of selected cells
        for (const auto &cell: cellsAtSelect)
        {
            //Get cell state
            GridUtility::CellBestFit &cellState =
                gridState.at(m_sizeStep).at(cell.y + GridUtility::PAD_GRID).
                at(cell.x + GridUtility::PAD_GRID);

            //Toggle cell state
            if (cellState.has_value())
                cellState = std::nullopt;
            else
                cellState = 0;
        }

        //Update grid view
        createGrid(backImage.rows, backImage.cols);
        updateView(false);
    }
}
