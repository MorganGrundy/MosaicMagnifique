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

#include "gridgenerator.h"

#include "imageutility.h"
#include "cellgroup.h"

GridGenerator::GridGenerator()
{}

//Generates grid preview
GridUtility::MosaicBestFit GridGenerator::getGridState(const CellGroup &t_cells,
                                                    const cv::Mat &t_mainImage,
                                                    const int height, const int width)
{
    GridUtility::MosaicBestFit gridState;
    //No cell mask, no grid
    if (t_cells.getCell(0).getCellMask(0, 0).empty())
        return gridState;

    //Calculate grid size
    const int gridHeight = (t_mainImage.empty()) ? height : t_mainImage.rows;
    const int gridWidth = (t_mainImage.empty()) ? width : t_mainImage.cols;

    //Stores grid bounds starting with full grid size
    std::vector<GridBounds> bounds(2);
    //Determines which bound is active
    int activeBound = 0;
    bounds.at(activeBound).addBound(gridHeight, gridWidth);

    for (size_t step = 0; step <= t_cells.getSizeSteps()
                          && !bounds.at(activeBound).empty(); ++step)
    {
        const cv::Point gridSize = GridUtility::calculateGridSize(t_cells.getCell(step),
                                                               gridWidth, gridHeight,
                                                               GridUtility::PAD_GRID);

        gridState.push_back(GridUtility::StepBestFit(static_cast<size_t>(gridSize.y),
                            std::vector<GridUtility::CellBestFit>(static_cast<size_t>(gridSize.x))));

        //Clear previous bounds
        bounds.at(!activeBound).clear();
        //Create all cells in grid
        for (int y = -GridUtility::PAD_GRID; y < gridSize.y - GridUtility::PAD_GRID; ++y)
        {
            for (int x = -GridUtility::PAD_GRID; x < gridSize.x - GridUtility::PAD_GRID; ++x)
            {
                //Find cell state
                const auto [bestFit, entropy] = findCellState(t_cells, t_mainImage, x, y,
                                                              bounds.at(activeBound), step);

                gridState.at(step).at(static_cast<size_t>(y + GridUtility::PAD_GRID)).
                    at(static_cast<size_t>(x + GridUtility::PAD_GRID)) = bestFit;

                //If cell entropy exceeded
                if (entropy)
                {
                    //Get cell bounds
                    cv::Rect cellBounds = GridUtility::getRectAt(t_cells.getCell(step),
                                                              x, y);

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

    return gridState;
}

//Finds state of cell at current position and step in detail image
std::pair<GridUtility::CellBestFit, bool>
GridGenerator::findCellState(const CellGroup &t_cells, const cv::Mat &t_mainImage,
                             const int x, const int y,
                             const GridBounds &t_bounds, const size_t t_step)
{
    const cv::Rect unboundedRect = GridUtility::getRectAt(t_cells.getCell(t_step), x, y);

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
    if (!t_mainImage.empty() && t_step < t_cells.getSizeSteps())
    {
        //Cell bounded positions (in image)
        yStart = std::clamp(unboundedRect.tl().y, 0, t_mainImage.rows);
        yEnd = std::clamp(unboundedRect.br().y, 0, t_mainImage.rows);
        xStart = std::clamp(unboundedRect.tl().x, 0, t_mainImage.cols);
        xEnd = std::clamp(unboundedRect.br().x, 0, t_mainImage.cols);

        const cv::Rect boundedRect(xStart - unboundedRect.x, yStart - unboundedRect.y,
                                   xEnd - xStart, yEnd - yStart);

        //Copies visible part of image to cell
        cv::Mat cell(t_mainImage, cv::Range(yStart, yEnd), cv::Range(xStart, xEnd));

        //Calculate if and how current cell is flipped
        const auto flipState = GridUtility::getFlipStateAt(t_cells.getCell(t_step), x, y,
                                                           GridUtility::PAD_GRID);

        //Resizes bounded rect for detail cells
        const cv::Rect boundedDetailRect(boundedRect.x * t_cells.getDetail(),
                                         boundedRect.y * t_cells.getDetail(),
                                         boundedRect.width * t_cells.getDetail(),
                                         boundedRect.height * t_cells.getDetail());

        //Create bounded mask of detail cell
        const cv::Mat mask(t_cells.getCell(t_step, true)
                               .getCellMask(flipState.horizontal, flipState.vertical),
                           boundedDetailRect);

        //Resize image cell to size of mask
        cell = ImageUtility::resizeImage(cell, mask.rows, mask.cols,
                                         ImageUtility::ResizeType::EXCLUSIVE);

        //If cell entropy exceeds threshold return true
        if (ImageUtility::calculateEntropy(cell, mask) >= ImageUtility::MAX_ENTROPY * 0.7)
            return {std::nullopt, true};
    }

    //Cell is valid
    return {0, false};
}
