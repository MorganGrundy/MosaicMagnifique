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
#ifdef CUDA

#include "cudaphotomosaicgenerator.h"

#include <QDebug>

#include "cudautility.h"
#include "photomosaicgenerator.cuh"
#include "reduction.cuh"
#include "..\..\Other\TimingLogger.h"

CUDAPhotomosaicGenerator::CUDAPhotomosaicGenerator()
{}

//Generate best fits for Photomosaic cells
//Returns true if successful
bool CUDAPhotomosaicGenerator::generateBestFits()
{
    TimingLogger timingLogger;
    timingLogger.StartTiming("generateBestFits");

    timingLogger.StartTiming("Preprocess");
    //Converts colour space of main image and library images
    //Resizes library based on detail level
    auto mainImages = preprocessMainImage();
    auto libImages = preprocessLibraryImages();
    timingLogger.StopTiming("Preprocess");

    timingLogger.StartTiming("cudaMalloc");
    //Get CUDA block size
    cudaDeviceProp deviceProp;
    gpuErrchk(cudaGetDeviceProperties(&deviceProp, 0));
    //Cannot use full blockSize in debug, so half it
#ifndef NDEBUG
    const size_t blockSize = deviceProp.maxThreadsPerBlock / 2;
#else
    const size_t blockSize = deviceProp.maxThreadsPerBlock;
#endif

    //Create streams
    const size_t streamCount = 16;
    cudaStream_t streams[streamCount];
    for (size_t i = 0; i < streamCount; ++i)
        gpuErrchk(cudaStreamCreate(&streams[i]));

    //The total pixels/area of the largest cell (where size step is 0)
    const size_t maxCellSize = std::pow(m_cells.getCellSize(0, true), 2);

    //Device memory for library images
    std::vector<float *> h_dLibIm(libImages.size(), nullptr);
    for (size_t libI = 0; libI < libImages.size(); ++libI)
        gpuErrchk(cudaMalloc((void **)&h_dLibIm.at(libI), maxCellSize * 3 * sizeof(float)));

    //Device memory for mask images
    std::vector<uchar *> h_dMaskImages(4);
    for (size_t i = 0; i < h_dMaskImages.size(); ++i)
        gpuErrchk(cudaMalloc((void **)&h_dMaskImages.at(i), maxCellSize * sizeof(uchar)));

    //Device memory for variants
    std::vector<double *> h_dVariants(mainImages.size(), nullptr);
    double **d_dVariants;
    for (size_t i = 0; i < mainImages.size(); ++i)
        gpuErrchk(cudaMalloc((void **)&h_dVariants.at(i), libImages.size() * maxCellSize * sizeof(double)));
    gpuErrchk(cudaMalloc((void **)&d_dVariants, mainImages.size() * sizeof(double *)));
    gpuErrchk(cudaMemcpy(d_dVariants, h_dVariants.data(), mainImages.size() * sizeof(double *), cudaMemcpyHostToDevice));

    //Device memory for cell image
    std::vector<float *> h_dcellImages(mainImages.size(), nullptr);
    for (size_t i = 0; i < mainImages.size(); ++i)
        gpuErrchk(cudaMalloc((void **)&h_dcellImages.at(i), maxCellSize * 3 * sizeof(float)));

    //Device memory for target area
    size_t *d_targetArea;
    gpuErrchk(cudaMalloc((void **)&d_targetArea, 4 * sizeof(size_t)));

    //The maximum size of memory needed for add reduction
    const size_t maxReductionMemSize = (((maxCellSize + 1) / 2 + blockSize - 1) / blockSize + 1) / 2;
    //Device memory for reduction memory
    std::vector<double *> h_dReductionMems(streamCount);
    for (size_t streamI = 0; streamI < streamCount; ++streamI)
        gpuErrchk(cudaMalloc((void **)&h_dReductionMems.at(streamI), libImages.size() * maxReductionMemSize * sizeof(double)));

    //Device memory for best fits
    size_t *d_bestFit;
    //Device memory for lowest variant
    double *d_lowestVariant;
    gpuErrchk(cudaMalloc((void **)&d_lowestVariant, sizeof(double)));
    constexpr double maxVariant = std::numeric_limits<double>::max();

    //The maximum number of cells in grid at a single size step (the smallest size step has the most cells)
    const size_t maxNoOfCells = m_bestFits.back().size() * m_bestFits.back().back().size();
    //Device memory for best fits
    gpuErrchk(cudaMalloc((void **)&d_bestFit, maxNoOfCells * sizeof(size_t)));
    timingLogger.StopTiming("cudaMalloc");

    timingLogger.StartTiming("StepLoop");
    //For all size steps, stop if no bounds for step
    for (size_t step = 0; step < m_bestFits.size(); ++step)
    {
        //If user hits cancel in QProgressDialog then return empty best fit
        if (m_wasCanceled)
            break;

        const int progressStep = std::pow(4, (m_bestFits.size() - 1) - step);

        //Reference to cell shapes
        const CellShape &normalCellShape = m_cells.getCell(step);
        const CellShape &detailCellShape = m_cells.getCell(step, true);

        //The total pixels/area of cells at the current size step
        const size_t cellSize = std::pow(m_cells.getCellSize(step, true), 2);
        //Size of memory needed for add reduction at the current size step
        const size_t reductionMemSize = (((cellSize + 1) / 2 + blockSize - 1) / blockSize + 1) / 2;

        timingLogger.StartTiming("cudaMemcpy");
        //Copy library images to device memory
        for (size_t libI = 0; libI < libImages.size(); ++libI)
            copyMatToDevice<float>(libImages.at(libI), h_dLibIm.at(libI));

        //Copy cell masks to device
        copyMatToDevice<uchar>(detailCellShape.getCellMask(0, 0), h_dMaskImages.at(0));
        copyMatToDevice<uchar>(detailCellShape.getCellMask(1, 0), h_dMaskImages.at(1));
        copyMatToDevice<uchar>(detailCellShape.getCellMask(0, 1), h_dMaskImages.at(2));
        copyMatToDevice<uchar>(detailCellShape.getCellMask(1, 1), h_dMaskImages.at(3));

        //Total number of cells in grid at current size step
        const size_t noOfCells = m_bestFits.at(step).size() * m_bestFits.at(step).at(0).size();
        //Set best fits to max value
        const size_t maxBestFit = libImages.size();
        gpuErrchk(cudaMemcpy(d_bestFit, &maxBestFit, sizeof(size_t), cudaMemcpyHostToDevice));
        for (size_t cellIndex = 1; cellIndex < noOfCells; ++cellIndex)
            gpuErrchk(cudaMemcpy(d_bestFit + cellIndex, d_bestFit, sizeof(size_t), cudaMemcpyDeviceToDevice));
        timingLogger.StopTiming("cudaMemcpy");

        timingLogger.StartTiming("YLoop");
        //Find best match for each cell in grid
        for (int y = -GridUtility::PAD_GRID;
             y < static_cast<int>(m_bestFits.at(step).size()) - GridUtility::PAD_GRID; ++y)
        {
            //If user hits cancel in QProgressDialog then return empty best fit
            if (m_wasCanceled)
                break;

            timingLogger.StartTiming("XLoop");
            for (int x = -GridUtility::PAD_GRID;
                 x < static_cast<int>(m_bestFits.at(step).at(y + GridUtility::PAD_GRID).size())
                         - GridUtility::PAD_GRID; ++x)
            {
                //If user hits cancel in QProgressDialog then return empty best fit
                if (m_wasCanceled)
                    break;

                //If cell is valid
                if (m_bestFits.at(step).at(y + GridUtility::PAD_GRID).at(x + GridUtility::PAD_GRID).has_value())
                {
                    auto [cells, cellBounds] = getCellAt(normalCellShape, detailCellShape, x, y, mainImages);

                    timingLogger.StartTiming("cudaMemcpy");
                    //Copy cell images to device
                    for (size_t i = 0; i < cells.size(); ++i)
                        copyMatToDevice<float>(cells.at(i), h_dcellImages.at(i));

                    const bool cellIsClipped = cellBounds.y > 0 || cellBounds.x > 0 || cellBounds.br().y < cells.front().cols || cellBounds.br().x < cells.front().rows;

                    if (cellIsClipped)
                    {
                        timingLogger.StartTiming("ClippedCell");
                        //Copy target area to device
                        const size_t targetArea[4] = {static_cast<size_t>(cellBounds.y),
                                                      static_cast<size_t>(cellBounds.br().y),
                                                      static_cast<size_t>(cellBounds.x),
                                                      static_cast<size_t>(cellBounds.br().x)};
                        gpuErrchk(cudaMemcpy(d_targetArea, targetArea, 4 * sizeof(size_t), cudaMemcpyHostToDevice));
                        timingLogger.StopTiming("ClippedCell");
                    }


                    //Calculate if and how current cell is flipped
                    const auto flipState = GridUtility::getFlipStateAt(detailCellShape, x, y);
                    //Select correct device mask image
                    uchar *d_mask = h_dMaskImages.at(flipState.horizontal + flipState.vertical * 2);

                    //Clear variants
                    for (size_t i = 0; i < cells.size(); ++i)
                        gpuErrchk(cudaMemset(h_dVariants.at(i), 0, libImages.size() * cellSize * sizeof(double)));

                    //Reset lowest variant
                    gpuErrchk(cudaMemcpy(d_lowestVariant, &maxVariant, sizeof(double), cudaMemcpyHostToDevice));

                    gpuErrchk(cudaStreamSynchronize(0));
                    timingLogger.StopTiming("cudaMemcpy");

                    timingLogger.StartTiming("DiffReduce");
                    size_t streamIndex = streamCount;
                    //Calculate difference
                    for (size_t libI = 0; libI < libImages.size(); ++libI)
                    {
                        for (size_t cellI = 0; cellI < cells.size(); ++cellI)
                        {
                            streamIndex = (streamIndex + 1) % streamCount;
                            ColourDifference::getCUDAFunction(m_colourDiffType, cellIsClipped)(h_dcellImages.at(cellI), h_dLibIm.at(libI), d_mask,
                                cells.front().rows, d_targetArea, h_dVariants.at(cellI) + libI * cellSize,
                                blockSize, streams[streamIndex]);

                            reduceAddKernelWrapper(blockSize, cellSize, h_dVariants.at(cellI) + libI * cellSize, h_dReductionMems.at(streamIndex), streams[streamIndex]);

                            gpuErrchk(cudaPeekAtLastError());
                        }
                    }

                    for (size_t streamI = 0; streamI < streamCount; ++streamI)
                        gpuErrchk(cudaStreamSynchronize(streams[streamI]));
                    timingLogger.StopTiming("DiffReduce");

                    //Flatten variants (the reduced variants have large gaps between them, we want them continuous)
                    timingLogger.StartTiming("Flatten");
                    flattenKernelWrapper(d_dVariants, cells.size(), libImages.size(), cellSize, blockSize);
                    timingLogger.StopTiming("Flatten");

                    //Calculate repeats and add to variants
                    timingLogger.StartTiming("Repeats");
                    const size_t gridWidth = m_bestFits.at(step).at(0).size();
                    calculateRepeatsKernelWrapper(d_dVariants, cells.size(),
                                                  d_bestFit, libImages.size(),
                                                  gridWidth, x, y,
                                                  GridUtility::PAD_GRID,
                                                  m_repeatRange, m_repeatAddition);
                    gpuErrchk(cudaPeekAtLastError());
                    timingLogger.StopTiming("Repeats");

                    //Find lowest variant
                    timingLogger.StartTiming("FindLowest");
                    const size_t cellPosition = (y + GridUtility::PAD_GRID) * gridWidth + (x + GridUtility::PAD_GRID);
                    findLowestKernelWrapper(d_lowestVariant, d_bestFit + cellPosition, d_dVariants, libImages.size(), cells.size());
                    gpuErrchk(cudaPeekAtLastError());
                    timingLogger.StopTiming("FindLowest");
                }

                //Increment progress bar
                m_progress += progressStep;
                emit progress(m_progress);
            }
            timingLogger.StopTiming("XLoop");
        }
        timingLogger.StopTiming("YLoop");

        //Copy best fit to host
        timingLogger.StartTiming("BestFit");
        std::vector<size_t> bestFit(noOfCells, 0);
        gpuErrchk(cudaMemcpy(bestFit.data(), d_bestFit, noOfCells * sizeof(size_t), cudaMemcpyDeviceToHost)); //Unknown error
        for (int y = -GridUtility::PAD_GRID;
            y < static_cast<int>(m_bestFits.at(step).size()) - GridUtility::PAD_GRID; ++y)
        {
            for (int x = -GridUtility::PAD_GRID;
                x < static_cast<int>(m_bestFits.at(step).at(y + GridUtility::PAD_GRID).size())
                - GridUtility::PAD_GRID; ++x)
            {
                //If cell is valid
                if (m_bestFits.at(step).at(y + GridUtility::PAD_GRID).at(x + GridUtility::PAD_GRID).has_value())
                {
                    const size_t cellPosition = (y + GridUtility::PAD_GRID) * m_bestFits.at(step).at(0).size() + (x + GridUtility::PAD_GRID);
                    m_bestFits.at(step).at(y + GridUtility::PAD_GRID).at(x + GridUtility::PAD_GRID) = bestFit.at(cellPosition);
                }
            }
        }
        timingLogger.StopTiming("BestFit");

        //Resize for next step
        if ((step + 1) < m_bestFits.size())
        {
            //Halve cell size
            ImageUtility::batchResizeMat(libImages);
        }
    }
    timingLogger.StopTiming("StepLoop");

    timingLogger.StartTiming("cudaFree");
    for (size_t i = 0; i < streamCount; ++i)
        gpuErrchk(cudaStreamDestroy(streams[i]));

    for (size_t i = 0; i < h_dMaskImages.size(); ++i)
        gpuErrchk(cudaFree(h_dMaskImages.at(i)));
    gpuErrchk(cudaFree(d_dVariants));
    for (size_t i = 0; i < mainImages.size(); ++i)
        gpuErrchk(cudaFree(h_dVariants.at(i)));
    for (size_t i = 0; i < mainImages.size(); ++i)
        gpuErrchk(cudaFree(h_dcellImages.at(i)));
    for (size_t i = 0; i < streamCount; ++i)
        gpuErrchk(cudaFree(h_dReductionMems.at(i)));

    for (auto im : h_dLibIm)
        gpuErrchk(cudaFree(im));

    gpuErrchk(cudaFree(d_targetArea));
    gpuErrchk(cudaFree(d_lowestVariant));
    gpuErrchk(cudaFree(d_bestFit));
    timingLogger.StopTiming("cudaFree");
    timingLogger.StopTiming("generateBestFits");
    timingLogger.StopAllTiming();
    timingLogger.LogTiming();

    return !m_wasCanceled;
}

//Copies mat to device pointer
template <typename T>
void CUDAPhotomosaicGenerator::copyMatToDevice(const cv::Mat &t_mat, T *t_device) const
{
    if (t_mat.isContinuous())
    {
        gpuErrchk(cudaMemcpy(t_device, t_mat.data, t_mat.rows * t_mat.cols * t_mat.channels() * sizeof(T),
            cudaMemcpyHostToDevice));
    }
    else
    {
        const T *p_im;
        for (int row = 0; row < t_mat.rows; ++row)
        {
            p_im = t_mat.ptr<T>(row);
            gpuErrchk(cudaMemcpy(t_device + row * t_mat.cols * t_mat.channels(), p_im,
                t_mat.cols * t_mat.channels() * sizeof(T),
                cudaMemcpyHostToDevice));
        }
    }
}

#endif // CUDA