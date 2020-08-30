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

#include "cudaphotomosaicgenerator.h"

#include <QDebug>

#include "cudaphotomosaicdata.h"

CUDAPhotomosaicGenerator::CUDAPhotomosaicGenerator(QWidget *t_parent)
    : PhotomosaicGeneratorBase{t_parent} {}

size_t differenceGPU(CUDAPhotomosaicData &photomosaicData);

//Generate best fits for Photomosaic cells
//Returns true if successful
bool CUDAPhotomosaicGenerator::generateBestFits()
{
    //Initialise progress bar
    if (!m_bestFits.empty())
    {
        setMaximum(2 * m_bestFits.at(0).at(0).size() * m_bestFits.at(0).size()
                   * std::pow(4, m_bestFits.size() - 1) * (m_bestFits.size()));
        setValue(0);
        setLabelText("Finding best fits (CUDA)...");
    }

    //Converts colour space of main image and library images
    //Resizes library based on detail level
    auto [mainImage, resizedLib] = resizeAndCvtColor();

    for (size_t step = 0; step < m_bestFits.size(); ++step)
    {
        const int progressStep = std::pow(4, (m_bestFits.size() - 1) - step);

        //Reference to cell shapes
        const CellShape &normalCellShape = m_cells.getCell(step);
        const CellShape &detailCellShape = m_cells.getCell(step, true);
        const int detailCellSize = m_cells.getCellSize(step, true);

        //Stores grid size
        const int gridHeight = static_cast<int>(m_bestFits.at(step).size());
        const int gridWidth = static_cast<int>(m_bestFits.at(step).at(0).size());

        //Count number of valid cells
        size_t validCells = 0;
        for (auto row: m_bestFits.at(step))
            validCells += std::count_if(row.begin(), row.end(),
                                        [](const GridUtility::CellBestFit &bestFit) {
                                            return bestFit.has_value();
                                        });

        //Allocate memory on GPU and copy data from CPU to GPU
        CUDAPhotomosaicData photomosaicData(detailCellSize, resizedLib.front().channels(),
                                            gridWidth, gridHeight, validCells, resizedLib.size(),
                                            m_mode != PhotomosaicGeneratorBase::Mode::CIEDE2000,
                                            m_repeatRange, m_repeatAddition);
        if (!photomosaicData.mallocData())
            return false;

        //Move library images to GPU
        photomosaicData.setLibraryImages(resizedLib);

        //Move mask images to GPU
        photomosaicData.setMaskImage(detailCellShape.getCellMask(0, 0), 0, 0);
        photomosaicData.setMaskImage(detailCellShape.getCellMask(1, 0), 1, 0);
        photomosaicData.setMaskImage(detailCellShape.getCellMask(0, 1), 0, 1);
        photomosaicData.setMaskImage(detailCellShape.getCellMask(1, 1), 1, 1);
        photomosaicData.setFlipStates(detailCellShape.getAlternateColFlipHorizontal(),
                                      detailCellShape.getAlternateColFlipVertical(),
                                      detailCellShape.getAlternateRowFlipHorizontal(),
                                      detailCellShape.getAlternateRowFlipVertical());

        //Stores cell image
        cv::Mat cell(detailCellSize, detailCellSize, m_img.type(), cv::Scalar(0));

        //Stores next data index
        size_t dataIndex = 0;

        //Copy input from host to CUDA device
        for (int y = -GridUtility::PAD_GRID; y < gridHeight - GridUtility::PAD_GRID; ++y)
        {
            for (int x = -GridUtility::PAD_GRID; x < gridWidth - GridUtility::PAD_GRID; ++x)
            {
                //If user hits cancel in QProgressDialog then return empty best fit
                if (wasCanceled())
                    return false;

                const GridUtility::CellBestFit &cellState = m_bestFits.at(step).
                                                            at(y + GridUtility::PAD_GRID).
                                                            at(x + GridUtility::PAD_GRID);

                //Set cell state on host
                photomosaicData.setCellState(x + GridUtility::PAD_GRID, y + GridUtility::PAD_GRID,
                                             cellState.has_value());

                //If cell valid
                if (cellState.has_value())
                {
                    //Sets cell position
                    photomosaicData.setCellPosition(x + GridUtility::PAD_GRID,
                                                    y + GridUtility::PAD_GRID,
                                                    dataIndex);

                    //Move cell image to GPU
                    auto [cell, cellBounds] = getCellAt(normalCellShape, detailCellShape,
                                                        x, y, GridUtility::PAD_GRID, mainImage);
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
        for (int y = -GridUtility::PAD_GRID; y < gridHeight - GridUtility::PAD_GRID; ++y)
        {
            for (int x = -GridUtility::PAD_GRID; x < gridWidth - GridUtility::PAD_GRID; ++x)
            {
                GridUtility::CellBestFit &cellState = m_bestFits.at(step).at(y + GridUtility::PAD_GRID).
                                                      at(x + GridUtility::PAD_GRID);
                //Skip if cell invalid
                if (!cellState.has_value())
                    continue;

                const size_t index = (y + GridUtility::PAD_GRID) * gridWidth + x + GridUtility::PAD_GRID;
                if (resultFlat[index] >= resizedLib.size())
                {
                    qDebug() << "Error: Failed to find a best fit";
                    continue;
                }

                cellState = resultFlat[index];

                setValue(value() + progressStep);
            }
        }

        //Deallocate memory on GPU and CPU
        photomosaicData.freeData();

        //Resize for next step
        if ((step + 1) < m_bestFits.size())
        {
            //Halve cell size
            ImageUtility::batchResizeMat(resizedLib);
        }
    }

    close();
    return true;
}
