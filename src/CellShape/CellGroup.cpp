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

#include "CellGroup.h"

#include "..\Other\ImageUtility.h"

CellGroup::CellGroup() : cells{1}, detailCells{1}, edgeCells{1, std::vector<cv::Mat>(4)},
    detail{1}, sizeSteps{0}
{}

//Sets the cell shape
void CellGroup::setCellShape(const CellShape &t_cellShape)
{
    cells.at(0) = t_cellShape;

    if (!cells.at(0).getCellMask(0, 0).empty())
    {
        //Create edge cell
        cv::Mat cellMask;
        ImageUtility::edgeDetect(cells.at(0).getCellMask(0, 0), cellMask);

        //Make black pixels transparent
        ImageUtility::matMakeTransparent(cellMask, getEdgeCell(0, false, false), 0);

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

//Returns size of top level cell
int CellGroup::getCellSize(const size_t t_sizeStep, const bool t_detail) const
{
    return getCell(t_sizeStep, t_detail).getSize();
}

//Sets the detail level
void CellGroup::setDetail(const int t_detail, const bool t_reset)
{
    double newDetail = detail;
    if (!t_reset)
        newDetail = t_detail / 100.0;

    if (newDetail != detail || t_reset)
    {
        detail = newDetail;

        //Create top level detail cell
        if (!cells.at(0).getCellMask(0, 0).empty())
        {
            const int cellSize = std::max(static_cast<int>(cells.at(0).getSize() * detail), 1);
            detailCells.at(0) = cells.at(0).resized(cellSize);
        }

        //Update size steps for all cells
        setSizeSteps(sizeSteps, true);
    }
}

//Returns the detail level
double CellGroup::getDetail() const
{
    return detail;
}

//Sets the number of size steps, for each step cell is halved in size
void CellGroup::setSizeSteps(const size_t t_steps, const bool t_reset)
{
    //Resize vectors
    cells.resize(t_steps + 1);
    detailCells.resize(t_steps + 1);
    edgeCells.resize(t_steps + 1, std::vector<cv::Mat>(4));

    //Create cell masks for each step size
    int cellSize = cells.at(0).getSize();
    for (size_t step = 1; step <= t_steps; ++step)
    {
        cellSize /= 2;

        //Only create if reset or is a new step size
        if (t_reset || step > sizeSteps)
        {
            //Create normal cell mask
            cells.at(step) = cells.at(step - 1).resized(cellSize);

            //Create detail cell mask
            detailCells.at(step) = cells.at(step).resized(std::max(static_cast<int>(cellSize * detail), 1));

            //Create edge cell mask
            cv::Mat cellMask;
            ImageUtility::edgeDetect(cells.at(step).getCellMask(0, 0), cellMask);
            //Make black pixels transparent
            ImageUtility::matMakeTransparent(cellMask, getEdgeCell(step, false, false), 0);

            cv::flip(getEdgeCell(0, false, false), getEdgeCell(step, true, false), 1);
            cv::flip(getEdgeCell(0, false, false), getEdgeCell(step, false, true), 0);
            cv::flip(getEdgeCell(0, false, false), getEdgeCell(step, true, true), -1);
        }
    }
    sizeSteps = t_steps;
}

//Returns the size steps
size_t CellGroup::getSizeSteps() const
{
    return sizeSteps;
}

//Returns reference to normal and detail cell shapes
CellShape &CellGroup::getCell(const size_t t_sizeStep, const bool t_detail)
{
    return t_detail ? detailCells.at(t_sizeStep) : cells.at(t_sizeStep);
}

//Returns const reference to normal and detail cell shapes
const CellShape &CellGroup::getCell(const size_t t_sizeStep, const bool t_detail) const
{
    return t_detail ? detailCells.at(t_sizeStep) : cells.at(t_sizeStep);
}

//Returns reference to an edge cell mask
cv::Mat &CellGroup::getEdgeCell(const size_t t_sizeStep, const bool t_flipHorizontal,
                                const bool t_flipVertical)
{
    return edgeCells.at(t_sizeStep).at(t_flipHorizontal + t_flipVertical * 2);
}

//Returns const reference to an edge cell mask
const cv::Mat &CellGroup::getEdgeCell(const size_t t_sizeStep, const bool t_flipHorizontal,
                                      const bool t_flipVertical) const
{
    return edgeCells.at(t_sizeStep).at(t_flipHorizontal + t_flipVertical * 2);
}
