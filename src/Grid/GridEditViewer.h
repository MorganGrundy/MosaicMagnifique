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

#pragma once

#include "GridViewer.h"

#include <QMouseEvent>

#include "QuadTree.h"

class GridEditViewer : public GridViewer
{
    Q_OBJECT
public:
    explicit GridEditViewer(QWidget *parent = nullptr);
    ~GridEditViewer();

    //Sets current size step for editor
    void setSizeStep(const size_t t_sizeStep);

    //Gets grid state of current options and creates grid
    //Creates quadtree of grid
    void updateGrid();

    //Represents the different tools for grid editing
    //Single - interact with single cell
    //Selection - interact with all cells in selection
    enum class Tool {Single, Selection};
    //Sets active tool
    void setTool(const Tool t_tool);

protected:
    //Gets intial click position for selection
    void mousePressEvent(QMouseEvent *event) override;

    //Displays selection
    void mouseMoveEvent(QMouseEvent *event) override;

    //Inverts state of clicked/selected cell
    void mouseReleaseEvent(QMouseEvent *event) override;

private:
    //Toggles state of cell at grid position
    void editSingle(const cv::Point t_gridPos);
    //Toggles state of all cells that intersect rect
    void editSelection(const cv::Rect t_selectionRect);

    //Current size step of grid to edit
    size_t m_sizeStep;

    //Stores a quadtree of grid cells for each size step
    std::vector<Quadtree> m_quadtree;

    //Stores active tool
    Tool m_tool;

    //Position of selection start
    QPoint m_selectionStart;

    //Rect for displaying selection
    QGraphicsRectItem *selectionItem;
};