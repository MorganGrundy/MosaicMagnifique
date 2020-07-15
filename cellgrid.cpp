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

#include "cellgrid.h"

#include <opencv2/imgproc.hpp>

//Calculates the number of given cells needed to fill an image of given size
cv::Point CellGrid::calculateGridSize(const CellShape &t_cellShape,
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

//Returns rect of cell shape at the given grid position
cv::Rect CellGrid::getRectAt(const CellShape &t_cellShape, const int t_x, const int t_y)
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

    result.width = t_cellShape.getCellMask(0,0).cols;
    result.height = t_cellShape.getCellMask(0,0).rows;
    return result;
}

//Returns the flip state of the given cell at given grid position
std::pair<bool, bool> CellGrid::getFlipStateAt(const CellShape &t_cellShape,
                                               const int t_x, const int t_y, const int t_pad)
{
    bool flipHorizontal = false, flipVertical = false;
    if (t_cellShape.getColFlipHorizontal() && (t_x + t_pad) % 2 == 1)
        flipHorizontal = !flipHorizontal;
    if (t_cellShape.getRowFlipHorizontal() && (t_y + t_pad) % 2 == 1)
        flipHorizontal = !flipHorizontal;
    if (t_cellShape.getColFlipVertical() && (t_x + t_pad) % 2 == 1)
        flipVertical = !flipVertical;
    if (t_cellShape.getRowFlipVertical() && (t_y + t_pad) % 2 == 1)
        flipVertical = !flipVertical;

    return {flipHorizontal, flipVertical};
}

//Returns the entropy of the given image in the given mask
double CellGrid::calculateEntropy(const cv::Mat &t_mask, const cv::Mat &t_image)
{
    if (t_image.empty() || t_mask.empty())
        return 0;

    //Convert image to grayscale
    cv::Mat grayImage;
    cv::cvtColor(t_image, grayImage, cv::COLOR_BGR2GRAY);

    //Calculate histogram in cell shape
    size_t pixelCount = 0;
    std::vector<size_t> histogram(256, 0);

    const uchar *p_im, *p_mask;
    for (int row = 0; row < grayImage.rows; ++row)
    {
        p_im = grayImage.ptr<uchar>(row);
        p_mask = t_mask.ptr<uchar>(row);
        for (int col = 0; col < grayImage.cols; ++col)
        {
            if (p_mask[col] != 0)
            {
                ++histogram.at(p_im[col]);
                ++pixelCount;
            }
        }
    }

    //Calculate entropy
    double entropy = 0;
    for (auto value: histogram)
    {
        const double probability = value / static_cast<double>(pixelCount);
        if (probability > 0)
            entropy -= probability * std::log2(probability);
    }

    return entropy;
}
