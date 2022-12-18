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
#include "..\..\Other\Logger.h"

CUDAPhotomosaicGenerator::CUDAPhotomosaicGenerator(const int device) : PhotomosaicGeneratorBase(), m_device(device)
{
    m_d_d_variants = nullptr;
    m_d_targetArea = nullptr;
    m_d_bestFit = nullptr;
    m_d_lowestVariant = nullptr;
}

//Generate best fits for Photomosaic cells
//Returns true if successful
bool CUDAPhotomosaicGenerator::generateBestFits()
{
    LogInfo("Started generating Photomosaic on GPU with CUDA.");

    TimingLogger timingLogger;
    timingLogger.StartTiming("generateBestFits");

    timingLogger.StartTiming("Preprocess");
    //Converts colour space of main image and library images
    //Resizes library based on detail level
    auto mainImages = preprocessMainImage();
    auto libImages = preprocessLibraryImages();
    timingLogger.StopTiming("Preprocess");

    //Get CUDA block size
    cudaDeviceProp deviceProp;
    cudaError cudaErrCode = cudaGetDeviceProperties(&deviceProp, m_device);
    CUDAUtility::cudaErrorType cudaErrType = CUDAUtility::CUDAErrMessageBox(nullptr, QString("Failed to get properties for device %1. Was it disconnected?").arg(m_device), cudaErrCode, { cudaErrorInvalidDevice }, __FILE__, __LINE__);
    if (cudaErrType != CUDAUtility::cudaErrorType::SUCCESS)
        return false;

    //Cannot use full blockSize in debug, so half it
#ifndef NDEBUG
    m_blockSize = deviceProp.maxThreadsPerBlock / 2;
#else
    m_blockSize = deviceProp.maxThreadsPerBlock;
#endif

    //Create streams
    cudaStream_t streams[streamCount];
    size_t streamsCreated = 0;
    for (size_t i = 0; i < streamCount && cudaErrType == CUDAUtility::cudaErrorType::SUCCESS; ++i)
    {
        cudaErrCode = cudaStreamCreate(&streams[i]);
        //This should never fail
        cudaErrType = CUDAUtility::CUDAErrMessageBox(nullptr, QString("Failed to create stream %1/%2").arg(i+1).arg(streamCount), cudaErrCode, {}, __FILE__, __LINE__);
        if (cudaErrType == CUDAUtility::cudaErrorType::SUCCESS)
            ++streamsCreated;
    }

    bool memWasAllocated = false;
    if (cudaErrType == CUDAUtility::cudaErrorType::SUCCESS)
        memWasAllocated = allocateDeviceMemory(timingLogger, mainImages, libImages);

    if (memWasAllocated)
    {
        cudaErrCode = cudaMemcpy(m_d_d_variants, m_h_d_variants.data(), mainImages.size() * sizeof(double *), cudaMemcpyHostToDevice);
        cudaErrType = CUDAUtility::CUDAErrMessageBox(nullptr, "Failed to copy variant device pointers to device array", cudaErrCode, {}, __FILE__, __LINE__);
    }

    if (cudaErrType == CUDAUtility::cudaErrorType::SUCCESS && memWasAllocated)
    {
        timingLogger.StartTiming("StepLoop");
        //For all size steps, stop if no bounds for step
        for (size_t step = 0; step < m_bestFits.size() && cudaErrType == CUDAUtility::cudaErrorType::SUCCESS; ++step)
        {
            //If user hits cancel in QProgressDialog then break out of loop and return after clean up
            if (m_wasCanceled)
            {
                LogInfo("Photomosaic generation cancelled.");
                break;
            }

            const int progressStep = std::pow(4, (m_bestFits.size() - 1) - step);

            //Reference to cell shapes
            const CellShape &normalCellShape = m_cells.getCell(step);
            const CellShape &detailCellShape = m_cells.getCell(step, true);

            //The total pixels/area of cells at the current size step
            const size_t cellSize = std::pow(m_cells.getCellSize(step, true), 2);
            //Size of memory needed for add reduction at the current size step
            const size_t reductionMemSize = (((cellSize + 1) / 2 + m_blockSize - 1) / m_blockSize + 1) / 2;

            timingLogger.StartTiming("cudaMemcpy");
            //Copy library images to device memory
            for (size_t libI = 0; libI < libImages.size() && cudaErrType == CUDAUtility::cudaErrorType::SUCCESS; ++libI)
            {
                cudaErrCode = copyMatToDevice<float>(libImages.at(libI), m_h_d_libIm.at(libI));
                cudaErrType = CUDAUtility::CUDAErrMessageBox(nullptr, QString("Failed to copy library image to device %1/%2").arg(libI+1).arg(libImages.size()), cudaErrCode, {}, __FILE__, __LINE__);
            }
            if (cudaErrType != CUDAUtility::cudaErrorType::SUCCESS)
                break;

            //Copy cell masks to device
            cudaErrCode = copyMatToDevice<uchar>(detailCellShape.getCellMask(0, 0), m_h_d_maskImages.at(0));
            cudaErrType = CUDAUtility::CUDAErrMessageBox(nullptr, "Failed to copy cell mask to device 1/4", cudaErrCode, {}, __FILE__, __LINE__);
            if (cudaErrType != CUDAUtility::cudaErrorType::SUCCESS)
                break;
            cudaErrCode = copyMatToDevice<uchar>(detailCellShape.getCellMask(1, 0), m_h_d_maskImages.at(1));
            cudaErrType = CUDAUtility::CUDAErrMessageBox(nullptr, "Failed to copy cell mask to device 2/4", cudaErrCode, {}, __FILE__, __LINE__);
            if (cudaErrType != CUDAUtility::cudaErrorType::SUCCESS)
                break;
            cudaErrCode = copyMatToDevice<uchar>(detailCellShape.getCellMask(0, 1), m_h_d_maskImages.at(2));
            cudaErrType = CUDAUtility::CUDAErrMessageBox(nullptr, "Failed to copy cell mask to device 3/4", cudaErrCode, {}, __FILE__, __LINE__);
            if (cudaErrType != CUDAUtility::cudaErrorType::SUCCESS)
                break;
            cudaErrCode = copyMatToDevice<uchar>(detailCellShape.getCellMask(1, 1), m_h_d_maskImages.at(3));
            cudaErrType = CUDAUtility::CUDAErrMessageBox(nullptr, "Failed to copy cell mask to device 4/4", cudaErrCode, {}, __FILE__, __LINE__);
            if (cudaErrType != CUDAUtility::cudaErrorType::SUCCESS)
                break;

            //Total number of cells in grid at current size step
            const size_t noOfCells = m_bestFits.at(step).size() * m_bestFits.at(step).at(0).size();
            //Set best fits to max value
            const size_t maxBestFit = libImages.size();
            cudaErrCode = cudaMemcpy(m_d_bestFit, &maxBestFit, sizeof(size_t), cudaMemcpyHostToDevice);
            cudaErrType = CUDAUtility::CUDAErrMessageBox(nullptr, "Failed to copy max best fit from host to device", cudaErrCode, {}, __FILE__, __LINE__);
            for (size_t cellIndex = 1; cellIndex < noOfCells && cudaErrType == CUDAUtility::cudaErrorType::SUCCESS; ++cellIndex)
            {
                cudaErrCode = cudaMemcpy(m_d_bestFit + cellIndex, m_d_bestFit, sizeof(size_t), cudaMemcpyDeviceToDevice);
                cudaErrType = CUDAUtility::CUDAErrMessageBox(nullptr, "Failed to copy max best fit from device to device", cudaErrCode, {}, __FILE__, __LINE__);
            }
            if (cudaErrType != CUDAUtility::cudaErrorType::SUCCESS)
                break;
            timingLogger.StopTiming("cudaMemcpy");

            timingLogger.StartTiming("YLoop");
            //Find best match for each cell in grid
            for (int y = -GridUtility::PAD_GRID;
                y < static_cast<int>(m_bestFits.at(step).size()) - GridUtility::PAD_GRID
                && cudaErrType == CUDAUtility::cudaErrorType::SUCCESS; ++y)
            {
                //If user hits cancel in QProgressDialog then break out of loop and return after clean up
                if (m_wasCanceled)
                {
                    LogInfo("Photomosaic generation cancelled.");
                    break;
                }

                timingLogger.StartTiming("XLoop");
                for (int x = -GridUtility::PAD_GRID;
                    x < static_cast<int>(m_bestFits.at(step).at(y + GridUtility::PAD_GRID).size())
                    - GridUtility::PAD_GRID && cudaErrType == CUDAUtility::cudaErrorType::SUCCESS; ++x)
                {
                    //If user hits cancel in QProgressDialog then break out of loop and return after clean up
                    if (m_wasCanceled)
                    {
                        LogInfo("Photomosaic generation cancelled.");
                        break;
                    }

                    //If cell is valid
                    if (m_bestFits.at(step).at(y + GridUtility::PAD_GRID).at(x + GridUtility::PAD_GRID).has_value())
                    {
                        auto [cells, cellBounds] = getCellAt(normalCellShape, detailCellShape, x, y, mainImages);

                        timingLogger.StartTiming("cudaMemcpy");
                        //Copy cell images to device
                        for (size_t i = 0; i < cells.size() && cudaErrType == CUDAUtility::cudaErrorType::SUCCESS; ++i)
                        {
                            cudaErrCode = copyMatToDevice<float>(cells.at(i), m_h_d_cellImages.at(i));
                            cudaErrType = CUDAUtility::CUDAErrMessageBox(nullptr, QString("Failed to copy cell image to device %1/%2").arg(i+1).arg(cells.size()), cudaErrCode, {}, __FILE__, __LINE__);
                        }
                        if (cudaErrType != CUDAUtility::cudaErrorType::SUCCESS)
                            break;

                        const bool cellIsClipped = cellBounds.y > 0 || cellBounds.x > 0 || cellBounds.br().y < cells.front().cols || cellBounds.br().x < cells.front().rows;

                        if (cellIsClipped)
                        {
                            timingLogger.StartTiming("ClippedCell");
                            //Copy target area to device
                            const size_t targetArea[4] = { static_cast<size_t>(cellBounds.y),
                                                          static_cast<size_t>(cellBounds.br().y),
                                                          static_cast<size_t>(cellBounds.x),
                                                          static_cast<size_t>(cellBounds.br().x) };
                            cudaErrCode = cudaMemcpy(m_d_targetArea, targetArea, 4 * sizeof(size_t), cudaMemcpyHostToDevice);
                            cudaErrType = CUDAUtility::CUDAErrMessageBox(nullptr, "Failed to copy target area to device", cudaErrCode, {}, __FILE__, __LINE__);
                            if (cudaErrType != CUDAUtility::cudaErrorType::SUCCESS)
                                break;
                            timingLogger.StopTiming("ClippedCell");
                        }

                        //Calculate if and how current cell is flipped
                        const auto flipState = GridUtility::getFlipStateAt(detailCellShape, x, y);
                        //Select correct device mask image
                        uchar *d_mask = m_h_d_maskImages.at(flipState.horizontal + flipState.vertical * 2);

                        //Clear variants
                        for (size_t i = 0; i < cells.size() && cudaErrType == CUDAUtility::cudaErrorType::SUCCESS; ++i)
                        {
                            cudaErrCode = cudaMemset(m_h_d_variants.at(i), 0, libImages.size() * cellSize * sizeof(double));
                            cudaErrType = CUDAUtility::CUDAErrMessageBox(nullptr, QString("Failed to clear variant %1/%2").arg(i+1).arg(cells.size()), cudaErrCode, {}, __FILE__, __LINE__);
                        }
                        if (cudaErrType != CUDAUtility::cudaErrorType::SUCCESS)
                            break;

                        //Reset lowest variant
                        cudaErrCode = cudaMemcpy(m_d_lowestVariant, &maxVariant, sizeof(double), cudaMemcpyHostToDevice);
                        cudaErrType = CUDAUtility::CUDAErrMessageBox(nullptr, "Failed to reset lowest variant", cudaErrCode, {}, __FILE__, __LINE__);
                        if (cudaErrType != CUDAUtility::cudaErrorType::SUCCESS)
                            break;

                        cudaErrCode = cudaStreamSynchronize(0);
                        cudaErrType = CUDAUtility::CUDAErrMessageBox(nullptr, "Failed to synchronise main stream", cudaErrCode, {}, __FILE__, __LINE__);
                        if (cudaErrType != CUDAUtility::cudaErrorType::SUCCESS)
                            break;
                        timingLogger.StopTiming("cudaMemcpy");

                        timingLogger.StartTiming("DiffReduce");
                        size_t streamIndex = streamCount;
                        //Calculate difference
                        for (size_t libI = 0; libI < libImages.size() && cudaErrType == CUDAUtility::cudaErrorType::SUCCESS; ++libI)
                        {
                            for (size_t cellI = 0; cellI < cells.size() && cudaErrType == CUDAUtility::cudaErrorType::SUCCESS; ++cellI)
                            {
                                streamIndex = (streamIndex + 1) % streamCount;
                                ColourDifference::getCUDAFunction(m_colourDiffType, cellIsClipped)(m_h_d_cellImages.at(cellI), m_h_d_libIm.at(libI), d_mask,
                                    cells.front().rows, m_d_targetArea, m_h_d_variants.at(cellI) + libI * cellSize,
                                    m_blockSize, streams[streamIndex]);

                                reduceAddKernelWrapper(m_blockSize, cellSize, m_h_d_variants.at(cellI) + libI * cellSize, m_h_d_reductionMems.at(streamIndex), streams[streamIndex]);

                                cudaErrCode = cudaPeekAtLastError();
                                cudaErrType = CUDAUtility::CUDAErrMessageBox(nullptr, "Error during colour difference or add reduction kernels.", cudaErrCode, {}, __FILE__, __LINE__);
                                if (cudaErrType != CUDAUtility::cudaErrorType::SUCCESS)
                                    break;
                            }
                        }

                        for (size_t streamI = 0; streamI < streamCount && cudaErrType == CUDAUtility::cudaErrorType::SUCCESS; ++streamI)
                        {
                            cudaErrCode = cudaStreamSynchronize(streams[streamI]);
                            cudaErrType = CUDAUtility::CUDAErrMessageBox(nullptr, QString("Failed to synchronise stream %1/%2").arg(streamI).arg(streamCount), cudaErrCode, {}, __FILE__, __LINE__);
                        }
                        if (cudaErrType != CUDAUtility::cudaErrorType::SUCCESS)
                            break;
                        timingLogger.StopTiming("DiffReduce");

                        //Flatten variants (the reduced variants have large gaps between them, we want them continuous)
                        timingLogger.StartTiming("Flatten");
                        flattenKernelWrapper(m_d_d_variants, cells.size(), libImages.size(), cellSize, m_blockSize);
                        cudaErrCode = cudaPeekAtLastError();
                        cudaErrType = CUDAUtility::CUDAErrMessageBox(nullptr, "Error during flatten kernel", cudaErrCode, {}, __FILE__, __LINE__);
                        if (cudaErrType != CUDAUtility::cudaErrorType::SUCCESS)
                            break;
                        timingLogger.StopTiming("Flatten");

                        //Calculate repeats and add to variants
                        timingLogger.StartTiming("Repeats");
                        const size_t gridWidth = m_bestFits.at(step).at(0).size();
                        calculateRepeatsKernelWrapper(m_d_d_variants, cells.size(),
                            m_d_bestFit, libImages.size(),
                            gridWidth, x, y,
                            GridUtility::PAD_GRID,
                            m_repeatRange, m_repeatAddition);
                        cudaErrCode = cudaPeekAtLastError();
                        cudaErrType = CUDAUtility::CUDAErrMessageBox(nullptr, "Error during calculate repeats kernel", cudaErrCode, {}, __FILE__, __LINE__);
                        if (cudaErrType != CUDAUtility::cudaErrorType::SUCCESS)
                            break;
                        timingLogger.StopTiming("Repeats");

                        //Find lowest variant
                        timingLogger.StartTiming("FindLowest");
                        const size_t cellPosition = (y + GridUtility::PAD_GRID) * gridWidth + (x + GridUtility::PAD_GRID);
                        findLowestKernelWrapper(m_d_lowestVariant, m_d_bestFit + cellPosition, m_d_d_variants, libImages.size(), cells.size());
                        cudaErrCode = cudaPeekAtLastError();
                        cudaErrType = CUDAUtility::CUDAErrMessageBox(nullptr, "Error during find lowest kernel", cudaErrCode, {}, __FILE__, __LINE__);
                        if (cudaErrType != CUDAUtility::cudaErrorType::SUCCESS)
                            break;
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
            cudaErrCode = cudaMemcpy(bestFit.data(), m_d_bestFit, noOfCells * sizeof(size_t), cudaMemcpyDeviceToHost); //Unknown error
            cudaErrType = CUDAUtility::CUDAErrMessageBox(nullptr, "Error copying best fits to host", cudaErrCode, {}, __FILE__, __LINE__);
            if (cudaErrType != CUDAUtility::cudaErrorType::SUCCESS)
                break;
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
    }

    size_t streamsDestroyed = 0;
    for (size_t i = 0; i < streamsCreated; ++i)
    {
        cudaErrCode = cudaStreamDestroy(streams[i]);
        if (cudaErrCode == cudaSuccess)
            ++streamsDestroyed;
        else
            LogCritical(QString("Failed to destroy CUDA stream %1/%2\n").arg(i+1).arg(streamCount) + CUDAUtility::createCUDAErrStr(cudaErrCode, __FILE__, __LINE__));
    }
    if (streamsDestroyed != streamsCreated)
    {
        QMessageBox::StandardButton clickedButton = MessageBox::critical(nullptr, "CUDA failed to destroy streams",
            "We failed to destroy some of the CUDA streams. I recommend restarting Mosaic Magnifique.\nPlease report this with the logs at: https://github.com/MorganGrundy/MosaicMagnifique/issues \n",
            QMessageBox::StandardButton::Ok | QMessageBox::Open);

        if (clickedButton == QMessageBox::Open)
            QDesktopServices::openUrl(QUrl("https://github.com/MorganGrundy/MosaicMagnifique/issues"));
    }

    freeDeviceMemory(timingLogger);

    timingLogger.StopTiming("generateBestFits");
    timingLogger.StopAllTiming();
    timingLogger.LogTiming();

    return !m_wasCanceled && memWasAllocated && cudaErrType == CUDAUtility::cudaErrorType::SUCCESS;
}

//Attempts to allocate all of the device memory
//Returns whether it was all allocated
bool CUDAPhotomosaicGenerator::allocateDeviceMemory(TimingLogger &timingLogger,
    const std::vector<cv::Mat> &mainImages, std::vector<cv::Mat> &libImages)
{
    LogInfo("CUDA Photomosaic Generator - Allocating CUDA memory.");
    timingLogger.StartTiming("cudaMalloc");

    //The total pixels/area of the largest cell (where size step is 0)
    const size_t maxCellSize = std::pow(m_cells.getCellSize(0, true), 2);

    //The maximum size of memory needed for add reduction
    const size_t maxReductionMemSize = (((maxCellSize + 1) / 2 + m_blockSize - 1) / m_blockSize + 1) / 2;

    //The maximum number of cells in grid at a single size step (the smallest size step has the most cells)
    const size_t maxNoOfCells = m_bestFits.back().size() * m_bestFits.back().back().size();

    //Sizes of the individual cudaMallocs
    const size_t libraryMallocSize = maxCellSize * 3 * sizeof(float);
    const size_t maskMallocSize = maxCellSize * sizeof(uchar);
    const size_t variantMallocSize = libImages.size() * maxCellSize * sizeof(double);
    const size_t variantArrayMallocSize = mainImages.size() * sizeof(double *);
    const size_t cellMallocSize = maxCellSize * 3 * sizeof(float);
    const size_t targetMallocSize = 4 * sizeof(size_t);
    const size_t reductionMallocSize = libImages.size() * maxReductionMemSize * sizeof(double);
    const size_t lowestVariantMallocSize = sizeof(double);
    const size_t bestFitMallocSize = maxNoOfCells * sizeof(size_t);

    //Calculate the total size of all cudaMallocs
    const size_t totalMallocSize = libraryMallocSize * libImages.size()
        + maskMallocSize * 4
        + variantMallocSize * mainImages.size()
        + variantArrayMallocSize
        + cellMallocSize * mainImages.size()
        + targetMallocSize
        + reductionMallocSize * streamCount
        + lowestVariantMallocSize
        + bestFitMallocSize;

    //Get the device free and total memory
    size_t freeMem = 0, totalMem = 0;
    cudaError cudaErrCode = cudaMemGetInfo(&freeMem, &totalMem);
    //Just log if this fails but continue as it's not essential (though if this fails the cudaMalloc probably will too)
    if (cudaErrCode != cudaSuccess)
        LogWarn("Failed to get device memory info\n" + CUDAUtility::createCUDAErrStr(cudaErrCode, __FILE__, __LINE__));
    else
    {
        LogInfo(QString("CUDA device has %1 free out of a total %2.").arg(Utility::FormatBytesAsString(freeMem)).arg(Utility::FormatBytesAsString(totalMem)));

        //Error if we don't have enough memory
        if (totalMem < totalMallocSize)
        {
            MessageBox::critical(nullptr, "CUDA not enough total memory", QString("CUDA device does not have enough total memory to generate this Photomosaic. Try reducing the number of library images, the detail, or cell size?\nIt has %1, needs over %2.").arg(Utility::FormatBytesAsString(totalMem)).arg(Utility::FormatBytesAsString(totalMallocSize)));
            return false;
        }
        else if (freeMem < totalMallocSize)
        {
            MessageBox::critical(nullptr, "CUDA not enough free memory", QString("CUDA device does not have enough free memory to generate this Photomosaic. Try closing other apps using the CUDA device.\nIt has %1 free out of %2, needs over %3.").arg(Utility::FormatBytesAsString(freeMem)).arg(Utility::FormatBytesAsString(totalMem)).arg(Utility::FormatBytesAsString(totalMallocSize)));
            return false;
        }
    }

    CUDAUtility::cudaErrorType cudaErrType;

    //Device memory for library images
    m_h_d_libIm = std::vector<float *>(libImages.size(), nullptr);
    for (size_t libI = 0; libI < libImages.size(); ++libI)
    {
        cudaErrCode = cudaMalloc((void **)&m_h_d_libIm.at(libI), libraryMallocSize);
        cudaErrType = CUDAUtility::CUDAErrMessageBox(nullptr, QString("Failed to allocate device memory for library image %1/%2").arg(libI+1).arg(libImages.size()), cudaErrCode, { cudaErrorMemoryAllocation }, __FILE__, __LINE__);
        if (cudaErrType != CUDAUtility::cudaErrorType::SUCCESS)
            return false;
    }

    //Device memory for mask images
    m_h_d_maskImages = std::vector<uchar *>(4);
    for (size_t i = 0; i < m_h_d_maskImages.size(); ++i)
    {
        cudaErrCode = cudaMalloc((void **)&m_h_d_maskImages.at(i), maskMallocSize);
        cudaErrType = CUDAUtility::CUDAErrMessageBox(nullptr, QString("Failed to allocate device memory for mask images %1/%2").arg(i+1).arg(m_h_d_maskImages.size()), cudaErrCode, { cudaErrorMemoryAllocation }, __FILE__, __LINE__);
        if (cudaErrType != CUDAUtility::cudaErrorType::SUCCESS)
            return false;
    }

    //Device memory for variants
    m_h_d_variants = std::vector<double *>(mainImages.size(), nullptr);
    for (size_t i = 0; i < mainImages.size(); ++i)
    {
        cudaErrCode = cudaMalloc((void **)&m_h_d_variants.at(i), variantMallocSize);
        cudaErrType = CUDAUtility::CUDAErrMessageBox(nullptr, QString("Failed to allocate device memory for variants %1/%2").arg(i+1).arg(mainImages.size()), cudaErrCode, { cudaErrorMemoryAllocation }, __FILE__, __LINE__);
        if (cudaErrType != CUDAUtility::cudaErrorType::SUCCESS)
            return false;
    }
    cudaErrCode = cudaMalloc((void **)&m_d_d_variants, variantArrayMallocSize);
    cudaErrType = CUDAUtility::CUDAErrMessageBox(nullptr, "Failed to allocate device memory for variants array", cudaErrCode, { cudaErrorMemoryAllocation }, __FILE__, __LINE__);
    if (cudaErrType != CUDAUtility::cudaErrorType::SUCCESS)
        return false;

    //Device memory for cell image
    m_h_d_cellImages = std::vector<float *>(mainImages.size(), nullptr);
    for (size_t i = 0; i < mainImages.size(); ++i)
    {
        cudaErrCode = cudaMalloc((void **)&m_h_d_cellImages.at(i), cellMallocSize);
        cudaErrType = CUDAUtility::CUDAErrMessageBox(nullptr, QString("Failed to allocate device memory for cell images %1/%2").arg(i+1).arg(mainImages.size()), cudaErrCode, { cudaErrorMemoryAllocation }, __FILE__, __LINE__);
        if (cudaErrType != CUDAUtility::cudaErrorType::SUCCESS)
            return false;
    }

    //Device memory for target area
    cudaErrCode = cudaMalloc((void **)&m_d_targetArea, targetMallocSize);
    cudaErrType = CUDAUtility::CUDAErrMessageBox(nullptr, "Failed to allocate device memory for target area", cudaErrCode, { cudaErrorMemoryAllocation }, __FILE__, __LINE__);
    if (cudaErrType != CUDAUtility::cudaErrorType::SUCCESS)
        return false;

    //Device memory for reduction memory
    m_h_d_reductionMems = std::vector<double *>(streamCount);
    for (size_t streamI = 0; streamI < streamCount; ++streamI)
    {
        cudaErrCode = cudaMalloc((void **)&m_h_d_reductionMems.at(streamI), reductionMallocSize);
        cudaErrType = CUDAUtility::CUDAErrMessageBox(nullptr, QString("Failed to allocate device memory for reduction %1/%2").arg(streamI+1).arg(streamCount), cudaErrCode, { cudaErrorMemoryAllocation }, __FILE__, __LINE__);
        if (cudaErrType != CUDAUtility::cudaErrorType::SUCCESS)
            return false;
    }

    //Device memory for lowest variant
    cudaErrCode = cudaMalloc((void **)&m_d_lowestVariant, lowestVariantMallocSize);
    cudaErrType = CUDAUtility::CUDAErrMessageBox(nullptr, "Failed to allocate device memory for lowest variant", cudaErrCode, { cudaErrorMemoryAllocation }, __FILE__, __LINE__);
    if (cudaErrType != CUDAUtility::cudaErrorType::SUCCESS)
        return false;

    //Device memory for best fits
    cudaErrCode = cudaMalloc((void **)&m_d_bestFit, bestFitMallocSize);
    cudaErrType = CUDAUtility::CUDAErrMessageBox(nullptr, "Failed to allocate device memory for best fits", cudaErrCode, { cudaErrorMemoryAllocation }, __FILE__, __LINE__);
    if (cudaErrType != CUDAUtility::cudaErrorType::SUCCESS)
        return false;

    timingLogger.StopTiming("cudaMalloc");
    LogInfo("CUDA Photomosaic Generator - Allocated CUDA memory.");

    return true;
}

//Attempts to free all of the device memory
//Returns whether it was all freed
void CUDAPhotomosaicGenerator::freeDeviceMemory(TimingLogger &timingLogger)
{
    LogInfo("CUDA Photomosaic Generator - Freeing CUDA memory.");
    timingLogger.StartTiming("cudaFree");

    //For freeing we will just log the errors and continue trying to free all the memory
    //Then at the end we will display a message box if any failed
    cudaError cudaErrCode;
    bool allFreed = true;

    for (size_t i = 0; i < m_h_d_libIm.size(); ++i)
    {
        cudaErrCode = cudaFree(m_h_d_libIm.at(i));
        if (cudaErrCode != cudaSuccess)
        {
            LogCritical(QString("Failed to free device memory for library image %1/%2\n").arg(i + 1).arg(m_h_d_libIm.size()) + CUDAUtility::createCUDAErrStr(cudaErrCode, __FILE__, __LINE__));
            allFreed = false;
        }
    }
    m_h_d_libIm.clear();

    for (size_t i = 0; i < m_h_d_maskImages.size(); ++i)
    {
        cudaErrCode = cudaFree(m_h_d_maskImages.at(i));
        if (cudaErrCode != cudaSuccess)
        {
            LogCritical(QString("Failed to free device memory for mask image %1/%2\n").arg(i+1).arg(m_h_d_maskImages.size()) + CUDAUtility::createCUDAErrStr(cudaErrCode, __FILE__, __LINE__));
            allFreed = false;
        }
    }
    m_h_d_maskImages.clear();

    cudaErrCode = cudaFree(m_d_d_variants);
    if (cudaErrCode != cudaSuccess)
    {
        LogCritical("Failed to free device memory for variant array\n" + CUDAUtility::createCUDAErrStr(cudaErrCode, __FILE__, __LINE__));
        allFreed = false;
    }
    m_d_d_variants = nullptr;
    for (size_t i = 0; i < m_h_d_variants.size(); ++i)
    {
        cudaErrCode = cudaFree(m_h_d_variants.at(i));
        if (cudaErrCode != cudaSuccess)
        {
            LogCritical(QString("Failed to free device memory for variant %1/%2\n").arg(i+1).arg(m_h_d_variants.size()) + CUDAUtility::createCUDAErrStr(cudaErrCode, __FILE__, __LINE__));
            allFreed = false;
        }
    }
    m_h_d_variants.clear();

    for (size_t i = 0; i < m_h_d_cellImages.size(); ++i)
    {
        cudaErrCode = cudaFree(m_h_d_cellImages.at(i));
        if (cudaErrCode != cudaSuccess)
        {
            LogCritical(QString("Failed to free device memory for cell image %1/%2\n").arg(i+1).arg(m_h_d_cellImages.size()) + CUDAUtility::createCUDAErrStr(cudaErrCode, __FILE__, __LINE__));
            allFreed = false;
        }
    }
    m_h_d_cellImages.clear();

    cudaErrCode = cudaFree(m_d_targetArea);
    if (cudaErrCode != cudaSuccess)
    {
        LogCritical("Failed to free device memory for target area\n" + CUDAUtility::createCUDAErrStr(cudaErrCode, __FILE__, __LINE__));
        allFreed = false;
    }
    m_d_targetArea = nullptr;

    for (size_t i = 0; i < m_h_d_reductionMems.size(); ++i)
    {
        cudaErrCode = cudaFree(m_h_d_reductionMems.at(i));
        if (cudaErrCode != cudaSuccess)
        {
            LogCritical(QString("Failed to free device memory for reduction %1/%2\n").arg(i+1).arg(m_h_d_reductionMems.size()) + CUDAUtility::createCUDAErrStr(cudaErrCode, __FILE__, __LINE__));
            allFreed = false;
        }
    }
    m_h_d_reductionMems.clear();

    cudaErrCode = cudaFree(m_d_lowestVariant);
    if (cudaErrCode != cudaSuccess)
    {
        LogCritical("Failed to free device memory for lowest variant\n" + CUDAUtility::createCUDAErrStr(cudaErrCode, __FILE__, __LINE__));
        allFreed = false;
    }
    m_d_lowestVariant = nullptr;

    cudaErrCode = cudaFree(m_d_bestFit);
    if (cudaErrCode != cudaSuccess)
    {
        LogCritical("Failed to free device memory for best fit\n" + CUDAUtility::createCUDAErrStr(cudaErrCode, __FILE__, __LINE__));
        allFreed = false;
    }
    m_d_bestFit = nullptr;

    if (!allFreed)
    {
        QMessageBox::StandardButton clickedButton = MessageBox::critical(nullptr, "CUDA failed to free memory",
            "We failed to free some of the CUDA memory. I recommend restarting Mosaic Magnifique.\nPlease report this with the logs at: https://github.com/MorganGrundy/MosaicMagnifique/issues \n",
            QMessageBox::StandardButton::Ok | QMessageBox::Open);

        if (clickedButton == QMessageBox::Open)
            QDesktopServices::openUrl(QUrl("https://github.com/MorganGrundy/MosaicMagnifique/issues"));
    }

    timingLogger.StopTiming("cudaFree");
    LogInfo("CUDA Photomosaic Generator - Freed CUDA memory.");
}

//Copies mat to device pointer
template <typename T>
cudaError CUDAPhotomosaicGenerator::copyMatToDevice(const cv::Mat &t_mat, T *t_device) const
{
    cudaError cudaErrCode = cudaSuccess;

    if (t_mat.isContinuous())
    {
        cudaErrCode = cudaMemcpy(t_device, t_mat.data, t_mat.rows * t_mat.cols * t_mat.channels() * sizeof(T),
            cudaMemcpyHostToDevice);
    }
    else
    {
        const T *p_im;
        for (int row = 0; row < t_mat.rows && cudaErrCode == cudaSuccess; ++row)
        {
            p_im = t_mat.ptr<T>(row);
            cudaErrCode = cudaMemcpy(t_device + row * t_mat.cols * t_mat.channels(), p_im,
                t_mat.cols * t_mat.channels() * sizeof(T), cudaMemcpyHostToDevice);
        }
    }

    return cudaErrCode;
}

#endif // CUDA