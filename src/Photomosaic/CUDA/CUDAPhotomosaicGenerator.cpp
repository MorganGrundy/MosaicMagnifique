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
#include <opencv2/cudawarping.hpp>

#include "cudautility.h"
#include "photomosaicgenerator.cuh"
#include "reduction.cuh"

CUDAPhotomosaicGenerator::CUDAPhotomosaicGenerator()
{}

//Sets CUDA library images
void CUDAPhotomosaicGenerator::setCUDALibrary(const std::vector<cv::cuda::GpuMat> &t_lib)
{
    m_cudaLib = t_lib;
}

//Generate best fits for Photomosaic cells
//Returns true if successful
bool CUDAPhotomosaicGenerator::generateBestFits()
{
    //Converts colour space of main image and library images
    //Resizes library based on detail level
    auto mainImages = preprocessMainImage();
    auto libImages = preprocessCUDALibraryImages();

    //Get CUDA block size
    cudaDeviceProp deviceProp;
    gpuErrchk(cudaGetDeviceProperties(&deviceProp, 0));
    //Cannot use full blockSize in debug, so half it
#ifndef NDEBUG
    const size_t blockSize = deviceProp.maxThreadsPerBlock / 2;
#else
    const size_t blockSize = deviceProp.maxThreadsPerBlock;
#endif

    const size_t streamCount = 16;
    cudaStream_t streams[streamCount];
    for (size_t i = 0; i < streamCount; ++i)
        gpuErrchk(cudaStreamCreate(&streams[i]));

    std::vector<float *> libIm_continuousD(libImages.size(), nullptr);

    //Device memory for mask images
    std::vector<uchar *> d_maskImages(4);

    //Device memory for variants
    double *d_variants;
    //Device memory for cell image
    float *d_cellImage;
    //Device memory for target area
    size_t *d_targetArea;
    gpuErrchk(cudaMalloc((void **)&d_targetArea, 4 * sizeof(size_t)));

    //Device memory for reduction memory
    std::vector<double *> d_reductionMems(streamCount);

    //Device memory for best fits
    size_t *d_bestFit;
    //Device memory for lowest variant
    double *d_lowestVariant;
    gpuErrchk(cudaMalloc((void **)&d_lowestVariant, sizeof(double)));
    constexpr double maxVariant = std::numeric_limits<double>::max();

    //For all size steps, stop if no bounds for step
    for (size_t step = 0; step < m_bestFits.size(); ++step)
    {
        const int progressStep = std::pow(4, (m_bestFits.size() - 1) - step);

        //Reference to cell shapes
        const CellShape &normalCellShape = m_cells.getCell(step);
        const CellShape &detailCellShape = m_cells.getCell(step, true);
        const size_t cellSize = std::pow(m_cells.getCellSize(step, true), 2);

        //Size of memory needed for add reduction
        const size_t reductionMemSize = (((cellSize + 1) / 2 + blockSize - 1) / blockSize + 1) / 2;

        //Allocate memory for device pointer
        if (step == 0)
        {
            //Cell masks
            for (size_t i = 0; i < d_maskImages.size(); ++i)
                gpuErrchk(cudaMalloc((void **)&d_maskImages.at(i), cellSize * sizeof(uchar)));

            //Cell image
            gpuErrchk(cudaMalloc((void **)&d_cellImage, cellSize * 3 * sizeof(float)));

            //Variants
            gpuErrchk(cudaMalloc((void **)&d_variants, libImages.size() * cellSize * sizeof(double)));

            //Reduction memory
            for (size_t streamI = 0; streamI < streamCount; ++streamI)
                gpuErrchk(cudaMalloc((void **)&d_reductionMems.at(streamI), libImages.size() * reductionMemSize * sizeof(double)));
        }

        size_t incontinuousCount = 0;
        //Check that all the CUDA library images are continuous, I can't figure out what causes them not to be
        //In the test project they seem to always be incontinuous but the actual CUDA project always continuous???
        //So if a library image is incontinuous then we need to copy it to continuous memory
        for (size_t libI = 0; libI < libImages.size(); ++libI)
        {
            if (!libImages.at(libI).isContinuous())
            {
                ++incontinuousCount;
                if (libIm_continuousD.at(libI) == nullptr)
                {
                    gpuErrchk(cudaMalloc((void **)&libIm_continuousD.at(libI), cellSize * 3 * sizeof(float)));
                }
                copyMatToDevice<float>(libImages.at(libI), libIm_continuousD.at(libI));
            }
        }
        if (incontinuousCount > 0)
            qInfo() << "Not all library images were continuous (" << incontinuousCount << "/" << libImages.size() << "), this will have some performance impact.";

        //Copy cell masks to device
        copyMatToDevice<uchar>(detailCellShape.getCellMask(0, 0), d_maskImages.at(0));
        copyMatToDevice<uchar>(detailCellShape.getCellMask(1, 0), d_maskImages.at(1));
        copyMatToDevice<uchar>(detailCellShape.getCellMask(0, 1), d_maskImages.at(2));
        copyMatToDevice<uchar>(detailCellShape.getCellMask(1, 1), d_maskImages.at(3));

        //Total number of cells in grid
        const size_t noOfCells = m_bestFits.at(step).size() * m_bestFits.at(step).at(0).size();
        //Create best fits and set to max value
        gpuErrchk(cudaMalloc((void **)&d_bestFit, noOfCells * sizeof(size_t)));
        const size_t maxBestFit = libImages.size();
        gpuErrchk(cudaMemcpy(d_bestFit, &maxBestFit, sizeof(size_t), cudaMemcpyHostToDevice));
        for (size_t cellIndex = 1; cellIndex < noOfCells; ++cellIndex)
            gpuErrchk(cudaMemcpy(d_bestFit + cellIndex, d_bestFit, sizeof(size_t),
                                 cudaMemcpyDeviceToDevice));

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
                    auto [cell, cellBounds] = getCellAt(normalCellShape, detailCellShape, x, y, mainImages);

                    //Copy cell image to device
                    copyMatToDevice<float>(cell.front(), d_cellImage);

                    //Copy target area to device
                    const size_t targetArea[4] = {static_cast<size_t>(cellBounds.y),
                                                  static_cast<size_t>(cellBounds.br().y),
                                                  static_cast<size_t>(cellBounds.x),
                                                  static_cast<size_t>(cellBounds.br().x)};
                    gpuErrchk(cudaMemcpy(d_targetArea, targetArea, 4 * sizeof(size_t),
                                         cudaMemcpyHostToDevice));


                    //Calculate if and how current cell is flipped
                    const auto flipState = GridUtility::getFlipStateAt(detailCellShape, x, y);
                    //Select correct device mask image
                    uchar *d_mask = d_maskImages.at(flipState.horizontal + flipState.vertical * 2);

                    //Clear variants
                    gpuErrchk(cudaMemset(d_variants, 0, libImages.size() * cellSize * sizeof(double)));

                    //Clear reduction memory
                    for (size_t streamI = 0; streamI < streamCount; ++streamI)
                        gpuErrchk(cudaMemsetAsync(d_reductionMems.at(streamI), 0, libImages.size() * reductionMemSize * sizeof(double), streams[streamI]));

                    //Reset lowest variant
                    gpuErrchk(cudaMemcpy(d_lowestVariant, &maxVariant, sizeof(double), cudaMemcpyHostToDevice));

                    gpuErrchk(cudaDeviceSynchronize());

                    //Calculate difference
                    for (size_t libI = 0; libI < libImages.size(); ++libI)
                    {
                        const size_t streamIndex = libI % streamCount;

                        float *d_libIm = libIm_continuousD.at(libI);
                        if (d_libIm == nullptr)
                            d_libIm = (float *)libImages.at(libI).cudaPtr();
                        ColourDifference::getCUDAFunction(m_colourDiffType)(d_cellImage, d_libIm, d_mask,
                            cell.front().rows, cell.front().channels(), d_targetArea, d_variants + libI * cellSize, blockSize, streams[streamIndex]);

                        reduceAddKernelWrapper(blockSize, cellSize, d_variants + libI * cellSize, d_reductionMems.at(streamIndex), streams[streamIndex]);

                        gpuErrchk(cudaPeekAtLastError());
                    }

                    for (size_t streamI = 0; streamI < streamCount; ++streamI)
                        gpuErrchk(cudaStreamSynchronize(streams[streamI]));

                    //Reduce variants
                    for (size_t libI = 0; libI < libImages.size(); ++libI)
                    {
                        const size_t streamIndex = libI % streamCount;
                        //Shift variant so that all variants are continuous
                        gpuErrchk(cudaMemcpyAsync(d_variants + libI, d_variants + libI * cellSize,
                                             sizeof(double), cudaMemcpyDeviceToDevice, streams[streamIndex]));
                    }

                    for (size_t streamI = 0; streamI < streamCount; ++streamI)
                        gpuErrchk(cudaStreamSynchronize(streams[streamI]));

                    //Calculate repeats and add to variants
                    const size_t gridWidth = m_bestFits.at(step).at(0).size();
                    calculateRepeatsKernelWrapper(d_variants,
                                                  d_bestFit, libImages.size(),
                                                  gridWidth, x, y,
                                                  GridUtility::PAD_GRID,
                                                  m_repeatRange, m_repeatAddition);
                    gpuErrchk(cudaPeekAtLastError());

                    //Find lowest variant
                    const size_t cellPosition = (y + GridUtility::PAD_GRID) * gridWidth + (x + GridUtility::PAD_GRID);
                    findLowestKernelWrapper(d_lowestVariant, d_bestFit + cellPosition, d_variants, libImages.size());
                    gpuErrchk(cudaPeekAtLastError());

                    //Copy best fit to host
                    size_t bestFit = 0;
                    gpuErrchk(cudaMemcpy(&bestFit, d_bestFit + cellPosition, sizeof(size_t), cudaMemcpyDeviceToHost)); //Unknown error

                    m_bestFits.at(step).at(y + GridUtility::PAD_GRID).at(x + GridUtility::PAD_GRID) = bestFit;
                }

                //Increment progress bar
                m_progress += progressStep;
                emit progress(m_progress);
            }
        }

        //Free best fits
        gpuErrchk(cudaFree(d_bestFit));

        //Resize for next step
        if ((step + 1) < m_bestFits.size())
        {
            //Halve cell size
            ImageUtility::batchResizeMat(libImages);
        }
    }

    for (size_t i = 0; i < streamCount; ++i)
        gpuErrchk(cudaStreamDestroy(streams[i]));

    for (size_t i = 0; i < d_maskImages.size(); ++i)
        gpuErrchk(cudaFree(d_maskImages.at(i)));
    gpuErrchk(cudaFree(d_variants));
    gpuErrchk(cudaFree(d_cellImage));
    for (size_t i = 0; i < streamCount; ++i)
        gpuErrchk(cudaFree(d_reductionMems.at(i)));

    for (auto d_libIm : libIm_continuousD)
    {
        if (d_libIm != nullptr)
            gpuErrchk(cudaFree(d_libIm));
    }

    gpuErrchk(cudaFree(d_targetArea));
    gpuErrchk(cudaFree(d_lowestVariant));

    return true;
}

//Performs preprocessing steps on library images: resize, convert colour space
std::vector<cv::cuda::GpuMat> CUDAPhotomosaicGenerator::preprocessCUDALibraryImages()
{
    //Resize
    std::vector<cv::cuda::GpuMat> result(m_cudaLib.size(), cv::cuda::GpuMat());
    //Use INTER_AREA for decreasing, INTER_CUBIC for increasing
    cv::InterpolationFlags flags = (m_cells.getDetail() < 1) ? cv::INTER_AREA : cv::INTER_CUBIC;
    int cellSize = std::round(m_cells.getCellSize(0, true));

    cv::cuda::Stream stream;
    for (size_t i = 0; i < m_cudaLib.size(); ++i)
    {
        if (m_cells.getDetail() != 1)
            cv::cuda::resize(m_cudaLib.at(i), result.at(i), cv::Size(cellSize, cellSize), 0, 0, flags, stream);
        else
            result.at(i) = m_cudaLib.at(i).clone();

        if (m_colourDiffType == ColourDifference::Type::CIE76 || m_colourDiffType == ColourDifference::Type::CIEDE2000)
        {
            //Convert 8U [0..255] to 32F [0..1]
            result.at(i).convertTo(result.at(i), CV_32F, 1 / 255.0, stream);
            //Convert BGR to Lab
            cv::cuda::cvtColor(result.at(i), result.at(i), cv::COLOR_BGR2Lab, 0, stream);
        }
        else
            //Convert 8U [0..255] to 32F [0..255]
            result.at(i).convertTo(result.at(i), CV_32F, stream);
    }
    stream.waitForCompletion();

    return result;
}

//Copies mat to device pointer
template <typename T>
void CUDAPhotomosaicGenerator::copyMatToDevice(const cv::Mat &t_mat, T *t_device) const
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

//Copies gpumat to device pointer
template <typename T>
void CUDAPhotomosaicGenerator::copyMatToDevice(const cv::cuda::GpuMat &t_mat, T *t_device) const
{
    const T *p_im;
    for (int row = 0; row < t_mat.rows; ++row)
    {
        p_im = t_mat.ptr<T>(row);
        gpuErrchk(cudaMemcpy(t_device + row * t_mat.cols * t_mat.channels(), p_im,
            t_mat.cols * t_mat.channels() * sizeof(T),
            cudaMemcpyDeviceToDevice));
    }
}

#endif // CUDA