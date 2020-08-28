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

CPUPhotomosaicGenerator::CPUPhotomosaicGenerator(QWidget *t_parent)
    : PhotomosaicGeneratorBase{t_parent} {}

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
        //Initialise progress bar
        if (step == 0)
        {
            setMaximum(m_bestFits.at(0).at(0).size() * m_bestFits.at(0).size()
                       * std::pow(4, m_bestFits.size() - 1) * (m_bestFits.size()));
            setValue(0);
            setLabelText("Finding best fits...");
            show();

        }
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
                if (wasCanceled())
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
                setValue(value() + progressStep);
            }
        }

        //Resize for next step
        if ((step + 1) < m_bestFits.size())
        {
            //Halve cell size
            resizedLib = ImageUtility::batchResizeMat(resizedLib);
        }
    }

    close();
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

    //Find cell best fit
    int index = -1;
    if (m_mode == Mode::CIEDE2000)
        index = findBestFitCIEDE2000(cell, cellMask, t_lib, repeats, cellBounds);
    else
        index = findBestFitEuclidean(cell, cellMask, t_lib, repeats, cellBounds);

    //Invalid index, should never happen
    if (index < 0 || index >= static_cast<int>(t_lib.size()))
    {
        qDebug() << "Failed to find a best fit";
        return std::nullopt;
    }

    return index;
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

//Compares pixels in the cell against the library images
//Returns the index of the library image with the smallest difference
//Used for mode RGB_EUCLIDEAN and CIE76
//(CIE76 is just a euclidean formulae in a different colour space)
int CPUPhotomosaicGenerator::findBestFitEuclidean(const cv::Mat &cell, const cv::Mat &mask,
                                                  const std::vector<cv::Mat> &library,
                                                  const std::map<size_t, int> &repeats,
                                                  const cv::Rect &t_bounds) const
{
    int bestFit = -1;
    long double bestVariant = std::numeric_limits<long double>::max();

    for (size_t i = 0; i < library.size(); ++i)
    {
        const auto it = repeats.find(i);
        long double variant = (it != repeats.end()) ? it->second : 0;

        //For cell and library image compare the corresponding pixels
        //Sum all pixel differences for total image difference
        const uchar *p_main, *p_im, *p_mask;
        for (int row = t_bounds.y;
             row < t_bounds.br().y && row < cell.rows && variant < bestVariant; ++row)
        {
            p_main = cell.ptr<uchar>(row);
            p_im = library.at(i).ptr<uchar>(row);
            p_mask = mask.ptr<uchar>(row);
            for (int col = t_bounds.x * cell.channels();
                 col < t_bounds.br().x * cell.channels() && variant < bestVariant;
                 col += cell.channels())
            {
                if (p_mask[col / cell.channels()] != 0)
                    variant += static_cast<long double>(
                        sqrt(pow(p_main[col] - p_im[col], 2) +
                             pow(p_main[col + 1] - p_im[col + 1], 2) +
                             pow(p_main[col + 2] - p_im[col + 2], 2)));
            }
        }

        //If image difference is less than current lowest then replace
        if (variant < bestVariant)
        {
            bestVariant = variant;
            bestFit = static_cast<int>(i);
        }
    }
    return bestFit;
}

//Compares pixels in the cell against the library images
//Returns the index of the library image with the smallest difference
//Used for mode CIEDE2000
int CPUPhotomosaicGenerator::findBestFitCIEDE2000(const cv::Mat &cell, const cv::Mat &mask,
                                                  const std::vector<cv::Mat> &library,
                                                  const std::map<size_t, int> &repeats,
                                                  const cv::Rect &t_bounds) const
{
    int bestFit = -1;
    long double bestVariant = std::numeric_limits<long double>::max();

    for (size_t i = 0; i < library.size(); ++i)
    {
        const auto it = repeats.find(i);
        long double variant = (it != repeats.end()) ? it->second : 0;

        //For cell and library image compare the corresponding pixels
        //Sum all pixel differences for total image difference
        const uchar *p_main, *p_im, *p_mask;
        for (int row = t_bounds.y;
             row < t_bounds.br().y && row < cell.rows && variant < bestVariant; ++row)
        {
            p_main = cell.ptr<uchar>(row);
            p_im = library.at(i).ptr<uchar>(row);
            p_mask = mask.ptr<uchar>(row);
            for (int col = t_bounds.x * cell.channels();
                 col < t_bounds.br().x * cell.channels() && variant < bestVariant;
                 col += cell.channels())
            {
                if (p_mask[col / cell.channels()] == 0)
                    continue;

                const double k_L = 1.0, k_C = 1.0, k_H = 1.0;
                const double deg360InRad = degToRad(360.0);
                const double deg180InRad = degToRad(180.0);
                const double pow25To7 = 6103515625.0; //pow(25, 7)

                const double C1 = sqrt((p_main[col + 1] * p_main[col + 1]) +
                                       (p_main[col + 2] * p_main[col + 2]));
                const double C2 = sqrt((p_im[col + 1] * p_im[col + 1]) +
                                       (p_im[col + 2] * p_im[col + 2]));
                const double barC = (C1 + C2) / 2.0;

                const double G = 0.5 * (1 - sqrt(pow(barC, 7) / (pow(barC, 7) + pow25To7)));

                const double a1Prime = (1.0 + G) * p_main[col + 1];
                const double a2Prime = (1.0 + G) * p_im[col + 1];

                const double CPrime1 = sqrt((a1Prime * a1Prime) +
                                            (p_main[col + 2] * p_main[col + 2]));
                const double CPrime2 = sqrt((a2Prime * a2Prime) +(p_im[col + 2] * p_im[col + 2]));

                double hPrime1;
                if (p_main[col + 2] == 0 && a1Prime == 0.0)
                    hPrime1 = 0.0;
                else
                {
                    hPrime1 = atan2(p_main[col + 2], a1Prime);
                    //This must be converted to a hue angle in degrees between 0 and 360 by
                    //addition of 2 pi to negative hue angles.
                    if (hPrime1 < 0)
                        hPrime1 += deg360InRad;
                }

                double hPrime2;
                if (p_im[col + 2] == 0 && a2Prime == 0.0)
                    hPrime2 = 0.0;
                else
                {
                    hPrime2 = atan2(p_im[col + 2], a2Prime);
                    //This must be converted to a hue angle in degrees between 0 and 360 by
                    //addition of 2pi to negative hue angles.
                    if (hPrime2 < 0)
                        hPrime2 += deg360InRad;
                }

                const double deltaLPrime = p_im[col] - p_main[col];
                const double deltaCPrime = CPrime2 - CPrime1;

                double deltahPrime;
                const double CPrimeProduct = CPrime1 * CPrime2;
                if (CPrimeProduct == 0.0)
                    deltahPrime = 0;
                else
                {
                    //Avoid the fabs() call
                    deltahPrime = hPrime2 - hPrime1;
                    if (deltahPrime < -deg180InRad)
                        deltahPrime += deg360InRad;
                    else if (deltahPrime > deg180InRad)
                        deltahPrime -= deg360InRad;
                }

                const double deltaHPrime = 2.0 * sqrt(CPrimeProduct) * sin(deltahPrime / 2.0);

                const double barLPrime = (p_main[col] + p_im[col]) / 2.0;
                const double barCPrime = (CPrime1 + CPrime2) / 2.0;

                double barhPrime;
                const double hPrimeSum = hPrime1 + hPrime2;
                if (CPrime1 * CPrime2 == 0.0)
                    barhPrime = hPrimeSum;
                else
                {
                    if (fabs(hPrime1 - hPrime2) <= deg180InRad)
                        barhPrime = hPrimeSum / 2.0;
                    else
                    {
                        if (hPrimeSum < deg360InRad)
                            barhPrime = (hPrimeSum + deg360InRad) / 2.0;
                        else
                            barhPrime = (hPrimeSum - deg360InRad) / 2.0;
                    }
                }

                const double T = 1.0 - (0.17 * cos(barhPrime - degToRad(30.0))) +
                                 (0.24 * cos(2.0 * barhPrime)) +
                                 (0.32 * cos((3.0 * barhPrime) + degToRad(6.0))) -
                                 (0.20 * cos((4.0 * barhPrime) - degToRad(63.0)));

                const double deltaTheta = degToRad(30.0) *
                                          exp(-pow((barhPrime - degToRad(275.0)) / degToRad(25.0), 2.0));

                const double R_C = 2.0 * sqrt(pow(barCPrime, 7.0) /
                                              (pow(barCPrime, 7.0) + pow25To7));

                const double S_L = 1 + ((0.015 * pow(barLPrime - 50.0, 2.0)) /
                                        sqrt(20 + pow(barLPrime - 50.0, 2.0)));
                const double S_C = 1 + (0.045 * barCPrime);
                const double S_H = 1 + (0.015 * barCPrime * T);

                const double R_T = (-sin(2.0 * deltaTheta)) * R_C;


                variant += static_cast<long double>(sqrt(pow(deltaLPrime / (k_L * S_L), 2.0) +
                                                         pow(deltaCPrime / (k_C * S_C), 2.0) +
                                                         pow(deltaHPrime / (k_H * S_H), 2.0) +
                                                         (R_T * (deltaCPrime / (k_C * S_C)) *
                                                          (deltaHPrime / (k_H * S_H)))));

            }
        }

        //If image difference is less than current lowest then replace
        if (variant < bestVariant)
        {
            bestVariant = variant;
            bestFit = static_cast<int>(i);
        }
    }
    return bestFit;
}

//Converts degrees to radians
double CPUPhotomosaicGenerator::degToRad(const double deg) const
{
    return (deg * M_PI) / 180;
}
