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

#include "cpuphotomosaicgenerator.h"

#include <QDebug>

#include "colourdifference.h"

CPUPhotomosaicGenerator::CPUPhotomosaicGenerator()
{}

//Generate best fits for Photomosaic cells
//Returns true if successful
bool CPUPhotomosaicGenerator::generateBestFits()
{
    //Converts colour space of main image and library images
    //Resizes library based on detail level
    auto [mainImage, resizedLib] = resizeAndCvtColor();

    //For all size steps, stop if no bounds for step
    for (size_t step = 0; step < m_bestFits.size(); ++step)
    {
        const int progressStep = std::pow(4, (m_bestFits.size() - 1) - step);

        //Reference to cell shapes
        const CellShape &normalCellShape = m_cells.getCell(step);
        const CellShape &detailCellShape = m_cells.getCell(step, true);

        //Find best match for each cell in grid
        for (int y = -GridUtility::PAD_GRID;
             y < static_cast<int>(m_bestFits.at(step).size()) - GridUtility::PAD_GRID; ++y)
        {
            for (int x = -GridUtility::PAD_GRID;
                 x < static_cast<int>(m_bestFits.at(step).at(y + GridUtility::PAD_GRID).size())
                         - GridUtility::PAD_GRID; ++x)
            {
                //If user hits cancel in QProgressDialog then return empty best fit
                if (m_wasCanceled)
                    return false;

                //If cell is valid
                if (m_bestFits.at(step).at(y + GridUtility::PAD_GRID).
                    at(x + GridUtility::PAD_GRID).has_value())
                {
                    //Find cell best fit
                    m_bestFits.at(step).at(static_cast<size_t>(y + GridUtility::PAD_GRID)).
                        at(static_cast<size_t>(x + GridUtility::PAD_GRID)) =
                            findCellBestFit(normalCellShape, detailCellShape, x, y,
                                            GridUtility::PAD_GRID, mainImage, resizedLib,
                                            m_bestFits.at(step));
                }

                //Increment progress bar
                m_progress += progressStep;
                emit progress(m_progress);
            }
        }

        //Resize for next step
        if ((step + 1) < m_bestFits.size())
        {
            //Halve cell size
            ImageUtility::batchResizeMat(resizedLib);
        }
    }

    return true;
}

//Returns best fit index for cell if it is the grid
std::optional<size_t>
CPUPhotomosaicGenerator::findCellBestFit(const CellShape &t_cellShape,
                                         const CellShape &t_detailCellShape,
                                         const int x, const int y, const bool t_pad,
                                         const cv::Mat &t_image, const std::vector<cv::Mat> &t_lib,
                                         const GridUtility::StepBestFit &t_grid) const
{
    auto [cell, cellBounds] = getCellAt(t_cellShape, t_detailCellShape, x, y, t_pad, t_image);

    //Calculate if and how current cell is flipped
    const auto flipState = GridUtility::getFlipStateAt(t_cellShape, x, y, t_pad);

    const cv::Mat &cellMask = t_detailCellShape.getCellMask(flipState.horizontal,
                                                            flipState.vertical);

    //Calculate repeat value of each library image in repeat range
    const std::map<size_t, int> repeats = calculateRepeats(t_grid, x + t_pad, y + t_pad);

    //Find library image most similar to cell
    int bestFit = -1;
    long double bestVariant = std::numeric_limits<long double>::max();
    for (size_t i = 0; i < t_lib.size(); ++i)
    {
        //Calculate difference between cell and library image
        //Increase difference based on repeating library images in range
        const auto it = repeats.find(i);
        double variant = (it != repeats.end()) ? it->second : 0;

        //Sum difference of corresponding pixels in cell and library image
        const cv::Vec3b *p_main, *p_im;
        const uchar *p_mask;
        for (int row = cellBounds.y; row < cellBounds.br().y && variant < bestVariant; ++row)
        {
            p_main = cell.ptr<cv::Vec3b>(row);
            p_im = t_lib.at(i).ptr<cv::Vec3b>(row);
            p_mask = cellMask.ptr<uchar>(row);
            for (int col = cellBounds.x; col < cellBounds.br().x && variant < bestVariant; ++col)
            {
                //Check pixel active in mask
                if (p_mask[col] != 0)
                {
                    if (m_mode == Mode::RGB_EUCLIDEAN)
                        variant += ColourDifference::calculateRGBEuclidean(p_main[col], p_im[col]);
                    else if (m_mode == Mode::CIE76)
                        variant += ColourDifference::calculateCIE76(p_main[col], p_im[col]);
                    else if (m_mode == Mode::CIEDE2000)
                        variant += ColourDifference::calculateCIEDE2000(p_main[col], p_im[col]);
                    else
                        qDebug() << Q_FUNC_INFO << "Unsupported mode";
                }
            }
        }

        //If image difference is less than current lowest then replace
        if (variant < bestVariant)
        {
            bestVariant = variant;
            bestFit = static_cast<int>(i);
        }
    }

    //Invalid index, should never happen
    if (bestFit < 0 || bestFit >= static_cast<int>(t_lib.size()))
    {
        qDebug() << Q_FUNC_INFO << "Failed to find a best fit";
        return std::nullopt;
    }

    return bestFit;
}

//Calculates the repeat value of each library image in repeat range around x,y
//Only needs to look at first half of cells as the latter half are not yet used
std::map<size_t, int> CPUPhotomosaicGenerator::calculateRepeats(
    const GridUtility::StepBestFit &grid, const int x, const int y) const
{
    std::map<size_t, int> repeats;
    const int repeatStartY = std::clamp(y - m_repeatRange, 0, static_cast<int>(grid.size()));
    const int repeatStartX = std::clamp(x - m_repeatRange, 0, static_cast<int>(grid.at(y).size()));
    const int repeatEndX = std::clamp(x + m_repeatRange, 0,
                                      static_cast<int>(grid.at(y).size()) - 1);

    //Looks at cells above the current cell
    for (int repeatY = repeatStartY; repeatY < y; ++repeatY)
    {
        for (int repeatX = repeatStartX; repeatX <= repeatEndX; ++repeatX)
        {
            const std::optional<size_t> cell = grid.at(static_cast<size_t>(repeatY)).
                                               at(static_cast<size_t>(repeatX));
            if (cell.has_value())
            {
                const auto it = repeats.find(cell.value());
                if (it != repeats.end())
                    it->second += m_repeatAddition;
                else
                    repeats.emplace(cell.value(), m_repeatAddition);
            }
        }
    }

    //Looks at cells directly to the left of current cell
    for (int repeatX = repeatStartX; repeatX < x; ++repeatX)
    {
        const std::optional<size_t> cell = grid.at(static_cast<size_t>(y)).
                                           at(static_cast<size_t>(repeatX));
        if (cell.has_value())
        {
            const auto it = repeats.find(cell.value());
            if (it != repeats.end())
                it->second += m_repeatAddition;
            else
                repeats.emplace(cell.value(), m_repeatAddition);
        }
    }
    return repeats;
}
