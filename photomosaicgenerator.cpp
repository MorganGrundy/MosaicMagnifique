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
    along with Foobar.  If not, see <https://www.gnu.org/licenses/>.
*/

#include "photomosaicgenerator.h"

#include <cmath>
#include <vector>
#include <climits>
#include <algorithm>
#include <opencv2/core/mat.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <QDebug>

#ifdef OPENCV_W_CUDA
#include <opencv2/cudawarping.hpp>
#include <opencv2/cudaimgproc.hpp>
#endif

#include "cellgrid.h"
#include "gridbounds.h"

#ifdef CUDA
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <math_constants.h>

#include "cudaphotomosaicdata.h"
#endif

PhotomosaicGenerator::PhotomosaicGenerator(QWidget *t_parent)
    : QProgressDialog{t_parent}, m_img{}, m_lib{}, m_detail{100}, m_mode{Mode::RGB_EUCLIDEAN},
      m_repeatRange{0}, m_repeatAddition{0}
{
    setWindowModality(Qt::WindowModal);
}

PhotomosaicGenerator::~PhotomosaicGenerator() {}

void PhotomosaicGenerator::setMainImage(const cv::Mat &t_img)
{
    m_img = t_img;
}

void PhotomosaicGenerator::setLibrary(const std::vector<cv::Mat> &t_lib)
{
    m_lib = t_lib;
}

void PhotomosaicGenerator::setDetail(const int t_detail)
{
    m_detail = (t_detail < 1) ? 0.01 : t_detail / 100.0;
}

void PhotomosaicGenerator::setMode(const Mode t_mode)
{
    m_mode = t_mode;
}

void PhotomosaicGenerator::setCellShape(const CellShape &t_cellShape)
{
    m_cellShape = t_cellShape;
}

void PhotomosaicGenerator::setGridState(const CellGrid::mosaicBestFit &t_gridState)
{
    m_gridState = t_gridState;
}

void PhotomosaicGenerator::setSizeSteps(const size_t t_steps)
{
    sizeSteps = t_steps;
}

void PhotomosaicGenerator::setRepeat(int t_repeatRange, int t_repeatAddition)
{
    m_repeatRange = t_repeatRange;
    m_repeatAddition = t_repeatAddition;
}

//Converts colour space of main image and library images
//Resizes library based on detail level
//Returns results
std::pair<cv::Mat, std::vector<cv::Mat>> PhotomosaicGenerator::resizeAndCvtColor()
{
    cv::Mat resultMain;
    std::vector<cv::Mat> result(m_lib.size(), cv::Mat());

    //Use INTER_AREA for decreasing, INTER_CUBIC for increasing
    cv::InterpolationFlags flags = (m_detail < 1) ? cv::INTER_AREA : cv::INTER_CUBIC;

#ifdef OPENCV_W_CUDA
    cv::cuda::Stream stream;
    //Main image
    cv::cuda::GpuMat main;
    main.upload(m_img, stream);
    if (m_mode == Mode::CIE76 || m_mode == Mode::CIEDE2000)
        cv::cuda::cvtColor(main, main, cv::COLOR_BGR2Lab, 0, stream);
    main.download(resultMain, stream);

    //Library image
    std::vector<cv::cuda::GpuMat> src(m_lib.size()), dst(m_lib.size());
    for (size_t i = 0; i < m_lib.size(); ++i)
    {
        //Resize image
        src.at(i).upload(m_lib.at(i), stream);
        cv::cuda::resize(src.at(i), dst.at(i),
                         cv::Size(static_cast<int>(m_detail * src.at(i).cols),
                                  static_cast<int>(m_detail * src.at(i).rows)),
                         0, 0, flags, stream);
        if (m_mode == Mode::CIE76 || m_mode == Mode::CIEDE2000)
            cv::cuda::cvtColor(dst.at(i), dst.at(i), cv::COLOR_BGR2Lab, 0, stream);
        dst.at(i).download(result.at(i), stream);
    }
    stream.waitForCompletion();
#else
    //Main image
    cv::cvtColor(m_img, resultMain, cv::COLOR_BGR2Lab);
    //Library image
    for (size_t i = 0; i < m_lib.size(); ++i)
    {
        cv::resize(m_lib.at(i), result.at(i),
                   cv::Size(static_cast<int>(m_detail * m_lib.at(i).cols),
                            static_cast<int>(m_detail * m_lib.at(i).rows)), 0, 0, flags);
        cv::cvtColor(result.at(i), result.at(i), cv::COLOR_BGR2Lab);
    }


#endif

    return {resultMain, result};
}

//Resizes vector of images based on ratio
void PhotomosaicGenerator::resizeImages(std::vector<cv::Mat> &t_images, const double t_ratio)
{
    //Use INTER_AREA for decreasing, INTER_CUBIC for increasing
    cv::InterpolationFlags flags = (t_ratio < 1) ? cv::INTER_AREA : cv::INTER_CUBIC;
#ifdef OPENCV_W_CUDA
    cv::cuda::Stream stream;

    //Library image
    std::vector<cv::cuda::GpuMat> src(t_images.size()), dst(t_images.size());
    for (size_t i = 0; i < t_images.size(); ++i)
    {
        //Resize image
        src.at(i).upload(t_images.at(i), stream);
        cv::cuda::resize(src.at(i), dst.at(i),
                         cv::Size(static_cast<int>(t_ratio * src.at(i).cols),
                                  static_cast<int>(t_ratio * src.at(i).rows)),
                         0, 0, flags, stream);
        dst.at(i).download(t_images.at(i), stream);
    }
    stream.waitForCompletion();
#else
    //Library image
    for (size_t i = 0; i < t_images.size(); ++i)
    {
        cv::resize(t_images.at(i), t_images.at(i),
                   cv::Size(static_cast<int>(t_ratio * t_images.at(i).cols),
                            static_cast<int>(t_ratio * t_images.at(i).rows)), 0, 0, flags);
    }


#endif
}

//Returns the cell image at given position and it's local bounds
std::pair<cv::Mat, cv::Rect> PhotomosaicGenerator::getCellAt(
    const CellShape &t_cellShape, const CellShape &t_detailCellShape,
    const int x, const int y, const bool t_pad,
    const cv::Mat &t_image) const
{
    //Gets bounds of cell in global space
    const cv::Rect cellGlobalBound = CellGrid::getRectAt(t_cellShape, x, y);

    //Bound cell in image area
    const int yStart = std::clamp(cellGlobalBound.y, 0, t_image.rows);
    const int yEnd = std::clamp(cellGlobalBound.br().y, 0, t_image.rows);
    const int xStart = std::clamp(cellGlobalBound.x, 0, t_image.cols);
    const int xEnd = std::clamp(cellGlobalBound.br().x, 0, t_image.cols);

    //Bounds of cell in local space
    const cv::Rect cellLocalBound(xStart - cellGlobalBound.x, yStart - cellGlobalBound.y,
                                  xEnd - xStart, yEnd - yStart);

    //Calculate if and how current cell is flipped
    auto [flipHorizontal, flipVertical] = CellGrid::getFlipStateAt(t_cellShape, x, y, t_pad);

    //Copies visible part of main image to cell
    cv::Mat cell(cellGlobalBound.height, cellGlobalBound.width, t_image.type(), cv::Scalar(0));
    t_image(cv::Range(yStart, yEnd), cv::Range(xStart, xEnd)).copyTo(cell(cellLocalBound));

    const cv::Mat &cellMask = t_detailCellShape.getCellMask(flipHorizontal, flipVertical);

    //Resize image cell to detail level
    cell = UtilityFuncs::resizeImage(cell, cellMask.rows, cellMask.cols,
                                     UtilityFuncs::ResizeType::EXACT);

    //Resizes bounds of cell in local space to detail level
    const cv::Rect detailCellLocalBound(cellLocalBound.x * m_detail, cellLocalBound.y * m_detail,
                                        cellLocalBound.width * m_detail,
                                        cellLocalBound.height * m_detail);

    return {cell, detailCellLocalBound};
}

#ifdef CUDA

size_t differenceGPU(CUDAPhotomosaicData &photomosaicData);

//Returns a Photomosaic of the main image made of the library images
//Generates using CUDA
cv::Mat PhotomosaicGenerator::cudaGenerate()
{
    //Converts colour space of main image and library images
    //Resizes library based on detail level
    auto [mainImage, resizedLib] = resizeAndCvtColor();

    //Stores cell shape, halved at each size step
    CellShape normalCellShape(m_cellShape);

    for (size_t step = 0; step < m_gridState.size(); ++step)
    {
        //Initialise progress bar
        if (step == 0)
        {
            setMaximum(m_gridState.at(0).at(0).size() * m_gridState.at(0).size()
                       * std::pow(4, m_gridState.size() - 1) * (m_gridState.size()));
            setValue(0);
            setLabelText("Moving data to CUDA device...");
        }
        const int progressStep = std::pow(4, (m_gridState.size() - 1) - step);

        //Create detail cell shape
        const int detailCellSize = normalCellShape.getCellMask(0, 0).rows * m_detail;
        CellShape detailCellShape(normalCellShape.resized(detailCellSize, detailCellSize));

        //Stores grid size
        const int gridHeight = static_cast<int>(m_gridState.at(step).size());
        const int gridWidth = static_cast<int>(m_gridState.at(step).at(0).size());

        //Count number of valid cells
        size_t validCells = 0;
        for (auto row: m_gridState.at(step))
            validCells += std::count_if(row.begin(), row.end(),
                                        [](const CellGrid::cellBestFit &bestFit) {
                                            return bestFit.has_value();
                                        });

        //Allocate memory on GPU and copy data from CPU to GPU
        CUDAPhotomosaicData photomosaicData(detailCellSize, resizedLib.front().channels(),
                                            gridWidth, gridHeight, validCells, resizedLib.size(),
                                            m_mode != PhotomosaicGenerator::Mode::CIEDE2000,
                                            m_repeatRange, m_repeatAddition);
        if (!photomosaicData.mallocData())
            return cv::Mat();

        //Move library images to GPU
        photomosaicData.setLibraryImages(resizedLib);

        //Move mask images to GPU
        photomosaicData.setMaskImage(detailCellShape.getCellMask(0, 0), 0, 0);
        photomosaicData.setMaskImage(detailCellShape.getCellMask(1, 0), 1, 0);
        photomosaicData.setMaskImage(detailCellShape.getCellMask(0, 1), 0, 1);
        photomosaicData.setMaskImage(detailCellShape.getCellMask(1, 1), 1, 1);
        photomosaicData.setFlipStates(detailCellShape.getColFlipHorizontal(),
                                      detailCellShape.getColFlipVertical(),
                                      detailCellShape.getRowFlipHorizontal(),
                                      detailCellShape.getRowFlipVertical());

        //Stores cell image
        cv::Mat cell(detailCellSize, detailCellSize, m_img.type(), cv::Scalar(0));

        //Stores next data index
        size_t dataIndex = 0;

        //Copy input from host to CUDA device
        for (int y = -CellGrid::PAD_GRID; y < gridHeight - CellGrid::PAD_GRID; ++y)
        {
            for (int x = -CellGrid::PAD_GRID; x < gridWidth - CellGrid::PAD_GRID; ++x)
            {
                //If user hits cancel in QProgressDialog then return empty mat
                if (wasCanceled())
                    return cv::Mat();

                const CellGrid::cellBestFit &cellState = m_gridState.at(step).
                                                         at(y + CellGrid::PAD_GRID).
                                                         at(x + CellGrid::PAD_GRID);

                //Set cell state on host
                photomosaicData.setCellState(x + CellGrid::PAD_GRID, y + CellGrid::PAD_GRID,
                                             cellState.has_value());

                //If cell valid
                if (cellState.has_value())
                {
                    //Sets cell position
                    photomosaicData.setCellPosition(x + CellGrid::PAD_GRID, y + CellGrid::PAD_GRID,
                                                    dataIndex);

                    //Move cell image to GPU
                    auto [cell, cellBounds] = getCellAt(normalCellShape, detailCellShape,
                                                        x, y, CellGrid::PAD_GRID, mainImage);
                    photomosaicData.setCellImage(cell, dataIndex);

                    //Move cell bounds to GPU
                    const size_t targetArea[4]{static_cast<size_t>(cellBounds.y),
                                               static_cast<size_t>(cellBounds.br().y),
                                               static_cast<size_t>(cellBounds.x),
                                               static_cast<size_t>(cellBounds.br().x)};
                    photomosaicData.setTargetArea(targetArea, dataIndex);

                    ++dataIndex;
                }

                setValue(value() + progressStep);
            }
        }

        //Copy cell states to GPU
        photomosaicData.copyCellState();

        //Calculate differences
        differenceGPU(photomosaicData);

        //Copy results from CUDA device to host
        size_t *resultFlat = photomosaicData.getResults();
        for (int y = -CellGrid::PAD_GRID; y < gridHeight - CellGrid::PAD_GRID; ++y)
        {
            for (int x = -CellGrid::PAD_GRID; x < gridWidth - CellGrid::PAD_GRID; ++x)
            {
                CellGrid::cellBestFit &cellState = m_gridState.at(step).at(y + CellGrid::PAD_GRID).
                                                   at(x + CellGrid::PAD_GRID);
                //Skip if cell invalid
                if (!cellState.has_value())
                    continue;

                const size_t index = (y + CellGrid::PAD_GRID) * gridWidth + x + CellGrid::PAD_GRID;
                if (resultFlat[index] >= resizedLib.size())
                {
                    qDebug() << "Error: Failed to find a best fit";
                    continue;
                }

                cellState = resultFlat[index];
            }
        }

        //Deallocate memory on GPU and CPU
        photomosaicData.freeData();

        //Resize for next step
        if ((step + 1) < m_gridState.size())
        {
            //Halve cell size
            normalCellShape = normalCellShape.resized(normalCellShape.getCellMask(0, 0).cols / 2,
                                                      normalCellShape.getCellMask(0, 0).rows / 2);
            resizeImages(resizedLib);
        }
    }

    //Combines all results into single image (mosaic)
    return combineResults(m_gridState);
}
#endif

//Returns best fit index for cell if it is the grid
std::optional<size_t>
PhotomosaicGenerator::findCellBestFit(const CellShape &t_cellShape,
                                      const CellShape &t_detailCellShape,
                                      const int x, const int y, const bool t_pad,
                                      const cv::Mat &t_image, const std::vector<cv::Mat> &t_lib,
                                      const CellGrid::stepBestFit &t_grid) const
{
    auto [cell, cellBounds] = getCellAt(t_cellShape, t_detailCellShape, x, y, t_pad, t_image);

    //Calculate if and how current cell is flipped
    const auto [flipHorizontal, flipVertical] = CellGrid::getFlipStateAt(t_cellShape, x, y, t_pad);

    const cv::Mat &cellMask = t_detailCellShape.getCellMask(flipHorizontal, flipVertical);

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

//Returns a Photomosaic of the main image made of the library images
cv::Mat PhotomosaicGenerator::generate()
{
    //Converts colour space of main image and library images
    //Resizes library based on detail level
    auto [mainImage, resizedLib] = resizeAndCvtColor();

    //Stores cell shape, halved at each size step
    CellShape normalCellShape(m_cellShape);

    //For all size steps, stop if no bounds for step
    for (size_t step = 0; step < m_gridState.size(); ++step)
    {
        //Initialise progress bar
        if (step == 0)
        {
            setMaximum(m_gridState.at(0).at(0).size() * m_gridState.at(0).size()
                       * std::pow(4, m_gridState.size() - 1) * (m_gridState.size()));
            setValue(0);
            setLabelText("Finding best fits...");
        }
        const int progressStep = std::pow(4, (m_gridState.size() - 1) - step);

        //Create detail cell shape
        const int detailCellSize = normalCellShape.getCellMask(0, 0).rows * m_detail;
        CellShape detailCellShape(normalCellShape.resized(detailCellSize, detailCellSize));

        //Find best match for each cell in grid
        for (int y = -CellGrid::PAD_GRID;
             y < static_cast<int>(m_gridState.at(step).size()) - CellGrid::PAD_GRID; ++y)
        {
            for (int x = -CellGrid::PAD_GRID;
                 x < static_cast<int>(m_gridState.at(step).at(y + CellGrid::PAD_GRID).size())
                         - CellGrid::PAD_GRID; ++x)
            {
                //If user hits cancel in QProgressDialog then return empty mat
                if (wasCanceled())
                    return cv::Mat();

                //If cell is valid
                if (m_gridState.at(step).at(y + CellGrid::PAD_GRID).
                    at(x + CellGrid::PAD_GRID).has_value())
                {
                    //Find cell best fit
                    m_gridState.at(step).at(static_cast<size_t>(y + CellGrid::PAD_GRID)).
                            at(static_cast<size_t>(x + CellGrid::PAD_GRID)) =
                            findCellBestFit(normalCellShape, detailCellShape, x, y,
                                            CellGrid::PAD_GRID, mainImage, resizedLib,
                                            m_gridState.at(step));
                }

                //Increment progress bar
                setValue(value() + progressStep);
            }
        }

        //Resize for next step
        if ((step + 1) < m_gridState.size())
        {
            //Halve cell size
            normalCellShape = normalCellShape.resized(normalCellShape.getCellMask(0, 0).cols / 2,
                                                      normalCellShape.getCellMask(0, 0).rows / 2);
            resizeImages(resizedLib);
        }
    }

    //Combines all results into single image (mosaic)
    return combineResults(m_gridState);
}

//Calculates the repeat value of each library image in repeat range around x,y
//Only needs to look at first half of cells as the latter half are not yet used
std::map<size_t, int> PhotomosaicGenerator::calculateRepeats(
        const CellGrid::stepBestFit &grid, const int x, const int y) const
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
int PhotomosaicGenerator::findBestFitEuclidean(const cv::Mat &cell, const cv::Mat &mask,
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
        for (int row = t_bounds.y; row < t_bounds.br().y && row < cell.rows && variant < bestVariant; ++row)
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
int PhotomosaicGenerator::findBestFitCIEDE2000(const cv::Mat &cell, const cv::Mat &mask,
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
        for (int row = t_bounds.y; row < t_bounds.br().y && row < cell.rows && variant < bestVariant; ++row)
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
double PhotomosaicGenerator::degToRad(const double deg) const
{
    return (deg * M_PI) / 180;
}

//Combines results into a Photomosaic
cv::Mat PhotomosaicGenerator::combineResults(const CellGrid::mosaicBestFit &results)
{
    //Update progress bar
    setLabelText("Building Photomosaic...");
    setValue(0);

    cv::Mat mosaic = cv::Mat::zeros(m_img.rows, m_img.cols, m_img.type());
    cv::Mat mosaicStep;

    cv::Mat mosaicMask = cv::Mat::zeros(m_img.rows, m_img.cols, CV_8UC1);
    cv::Mat mosaicMaskStep;

    //Stores library images, halved at each size step
    std::vector<cv::Mat> libImg(m_lib);

    //Stores cell shape, halved at each size step
    CellShape normalCellShape(m_cellShape);

    //For all size steps in results
    for (size_t step = 0; step < results.size(); ++step)
    {
        const int progressStep = std::pow(4, (m_gridState.size() - 1) - step);

        //Halve cell size
        if (step != 0)
        {
            normalCellShape = normalCellShape.resized(normalCellShape.getCellMask(0, 0).cols / 2,
                                                      normalCellShape.getCellMask(0, 0).rows / 2);
            resizeImages(libImg);
        }

        mosaicStep = cv::Mat::zeros(m_img.rows, m_img.cols, m_img.type());
        mosaicMaskStep = cv::Mat::zeros(m_img.rows, m_img.cols, CV_8UC1);

        //For all cells
        for (int y = -CellGrid::PAD_GRID; y < static_cast<int>(results.at(step).size())
                                                  - CellGrid::PAD_GRID; ++y)
        {
            for (int x = -CellGrid::PAD_GRID;
                 x < static_cast<int>(results.at(step).at(y + CellGrid::PAD_GRID).size())
                         - CellGrid::PAD_GRID; ++x)
            {
                const CellGrid::cellBestFit &cellData = results.at(step).
                        at(static_cast<size_t>(y + CellGrid::PAD_GRID)).
                        at(static_cast<size_t>(x + CellGrid::PAD_GRID));
                //Skip invalid cells
                if (!cellData.has_value())
                {
                    //Increment progress bar
                    setValue(value() + progressStep);
                    continue;
                }

                //Gets bounds of cell in global space
                const cv::Rect cellGlobalBound = CellGrid::getRectAt(normalCellShape, x, y);

                //Bound cell in image area
                const int yStart = std::clamp(cellGlobalBound.tl().y, 0, m_img.rows);
                const int yEnd = std::clamp(cellGlobalBound.br().y, 0, m_img.rows);
                const int xStart = std::clamp(cellGlobalBound.tl().x, 0, m_img.cols);
                const int xEnd = std::clamp(cellGlobalBound.br().x, 0, m_img.cols);

                //Bounds of cell in local space
                const cv::Rect cellLocalBound(xStart - cellGlobalBound.x,
                                              yStart - cellGlobalBound.y,
                                              xEnd - xStart, yEnd - yStart);

                //Calculate if and how current cell is flipped
                const auto [flipHorizontal, flipVertical] =
                    CellGrid::getFlipStateAt(normalCellShape, x, y, CellGrid::PAD_GRID);

                //Creates mask bounded
                const cv::Mat maskBounded(normalCellShape.getCellMask(flipHorizontal, flipVertical),
                                          cellLocalBound);

                //Adds cells to mosaic step
                const cv::Mat libBounded(libImg.at(cellData.value()), cellLocalBound);
                libBounded.copyTo(mosaicStep(cv::Range(yStart, yEnd), cv::Range(xStart, xEnd)),
                                  maskBounded);

                //Adds cell mask to mosaic mask step
                cv::Mat mosaicMaskPart(mosaicMaskStep, cv::Range(yStart, yEnd),
                                       cv::Range(xStart, xEnd));
                cv::bitwise_or(mosaicMaskPart, maskBounded, mosaicMaskPart);

                //Increment progress bar
                setValue(value() + progressStep);
            }
        }

        //Combine mosaic step into mosaic
        if (step != 0)
        {
            cv::Mat mask;
            cv::bitwise_not(mosaicMask, mask);
            mosaicStep.copyTo(mosaic, mask);
            mosaicMaskStep.copyTo(mosaicMask, mask);
            //CopyTo is a shallow copy, clone to make a deep copy
            mosaic = mosaic.clone();
            mosaicMask = mosaicMask.clone();
        }
        else
        {
            mosaic = mosaicStep.clone();
            mosaicMask = mosaicMaskStep.clone();
        }
    }

    return mosaic;
}
