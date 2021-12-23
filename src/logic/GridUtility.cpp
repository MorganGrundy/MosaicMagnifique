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

#include "GridUtility.h"

#include <opencv2/imgproc.hpp>

//Calculates the number of given cells needed to fill an image of given size
cv::Point GridUtility::calculateGridSize(const CellShape &t_cellShape,
                                         const int t_imageWidth, const int t_imageHeight,
                                         const int t_pad)
{
    cv::Point gridSize;
    //Calculates number of cells across x-axis
    if (t_cellShape.getColSpacing() != t_cellShape.getAlternateColSpacing())
        gridSize.x = 2 * ((t_imageWidth + t_cellShape.getColSpacing()
                           + t_cellShape.getAlternateColSpacing() - 1)
                          / (t_cellShape.getColSpacing() + t_cellShape.getAlternateColSpacing()));
    else
        gridSize.x = (t_imageWidth + t_cellShape.getColSpacing() - 1)
                / t_cellShape.getColSpacing();

    //Calculates number of cells across y-axis
    if (t_cellShape.getRowSpacing() != t_cellShape.getAlternateRowSpacing())
        gridSize.y = 2 * ((t_imageHeight + t_cellShape.getRowSpacing()
                           + t_cellShape.getAlternateRowSpacing() - 1)
                          / (t_cellShape.getRowSpacing() + t_cellShape.getAlternateRowSpacing()));
    else
        gridSize.y = (t_imageHeight + t_cellShape.getRowSpacing() - 1)
                / t_cellShape.getRowSpacing();

    gridSize.x += t_pad;
    gridSize.y += t_pad;

    return gridSize;
}


//Calculates the minimum image size containing target number of cells
cv::Point GridUtility::calculateImageSize(const CellShape &t_cellShape,
                                          const int t_cellX, const int t_cellY,
                                          const int t_pad)
{
    cv::Point imageSize;

    //Remove padding
    int cellX = t_cellX - t_pad;
    int cellY = t_cellY - t_pad;

    //Calculate image width
    if (t_cellShape.getColSpacing() != t_cellShape.getAlternateColSpacing())
        imageSize.x = (cellX / 2)
                      * (t_cellShape.getColSpacing() + t_cellShape.getAlternateColSpacing())
                      - t_cellShape.getAlternateColSpacing() - t_cellShape.getColSpacing() + 1;
    else
        imageSize.x = (cellX * t_cellShape.getColSpacing()) - t_cellShape.getColSpacing() + 1;

    //Calculate image height
    if (t_cellShape.getRowSpacing() != t_cellShape.getAlternateRowSpacing())
        imageSize.y = (cellY / 2)
                      * (t_cellShape.getRowSpacing() + t_cellShape.getAlternateRowSpacing())
                      - t_cellShape.getAlternateRowSpacing() - t_cellShape.getRowSpacing() + 1;
    else
        imageSize.y = (cellY * t_cellShape.getRowSpacing()) - t_cellShape.getRowSpacing() + 1;

    return imageSize;
}

//Returns rect of cell shape at the given grid position
cv::Rect GridUtility::getRectAt(const CellShape &t_cellShape, const int t_x, const int t_y)
{
    const int cellsX = t_x / 2;
    const int alternateCellsX = t_x - cellsX;
    const int cellsY = t_y / 2;
    const int alternateCellsY = t_y - cellsY;

    cv::Rect result;
    //Calculate cell x start position
    if (t_x < 0)
        result.x = alternateCellsX * t_cellShape.getColSpacing()
                + cellsX * t_cellShape.getAlternateColSpacing();
    else
        result.x = cellsX * t_cellShape.getColSpacing()
                + alternateCellsX * t_cellShape.getAlternateColSpacing();
    result.x += (t_y % 2 != 0) ? t_cellShape.getAlternateRowOffset() : 0;

    //Calculate cell y start position
    if (t_y < 0)
        result.y = alternateCellsY * t_cellShape.getRowSpacing()
                + cellsY * t_cellShape.getAlternateRowSpacing();
    else
        result.y = cellsY * t_cellShape.getRowSpacing()
                + alternateCellsY * t_cellShape.getAlternateRowSpacing();
    result.y += (t_x % 2 != 0) ? t_cellShape.getAlternateColOffset() : 0;

    result.width = t_cellShape.getSize();
    result.height = t_cellShape.getSize();
    return result;
}

//Returns the flip state of the given cell at given grid position
GridUtility::FlipState GridUtility::getFlipStateAt(const CellShape &t_cellShape,
                                                   const int t_x, const int t_y)
{
    FlipState flipState;
    if (t_cellShape.getAlternateColFlipHorizontal() && t_x % 2 != 0)
        flipState.horizontal = !flipState.horizontal;
    if (t_cellShape.getAlternateRowFlipHorizontal() && t_y % 2 != 0)
        flipState.horizontal = !flipState.horizontal;
    if (t_cellShape.getAlternateColFlipVertical() && t_x % 2 != 0)
        flipState.vertical = !flipState.vertical;
    if (t_cellShape.getAlternateRowFlipVertical() && t_y % 2 != 0)
        flipState.vertical = !flipState.vertical;

    return flipState;
}
