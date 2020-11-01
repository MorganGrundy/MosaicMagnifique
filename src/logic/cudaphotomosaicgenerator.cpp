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

#include "cudautility.h"
#include "photomosaicgenerator.cuh"
#include "reduction.cuh"

CUDAPhotomosaicGenerator::CUDAPhotomosaicGenerator()
{}

//Generate best fits for Photomosaic cells
//Returns true if successful
bool CUDAPhotomosaicGenerator::generateBestFits()
{
    //Converts colour space of main image and library images
    //Resizes library based on detail level
    auto [mainImage, resizedLib] = resizeAndCvtColor();

    //Get CUDA block size
    cudaDeviceProp deviceProp;
    gpuErrchk(cudaGetDeviceProperties(&deviceProp, 0));
    const size_t blockSize = deviceProp.maxThreadsPerBlock;

    //Device memory for mask images
    std::vector<uchar *> d_maskImages(4);
    //Device memory for library images
    float *d_libraryImage;

    //Device memory for variants
    double *d_variants;
    //Device memory for cell image
    float *d_cellImage;
    //Device memory for target area
    size_t *d_targetArea;
    //Device memory for reduction memory
    double *d_reductionMem;

    //Device memory for best fits
    size_t *d_bestFit;
    //Device memory for repeats
    size_t *d_repeats;
    //Device memory for lowest variant
    double *d_lowestVariant;

    //For all size steps, stop if no bounds for step
    for (size_t step = 0; step < m_bestFits.size(); ++step)
    {
        const int progressStep = std::pow(4, (m_bestFits.size() - 1) - step);

        //Reference to cell shapes
        const CellShape &normalCellShape = m_cells.getCell(step);
        const CellShape &detailCellShape = m_cells.getCell(step, true);
        const size_t cellSize = detailCellShape.getCellMask(0, 0).rows
                                * detailCellShape.getCellMask(0, 0).cols;

        //Copy cell masks to device
        for (size_t i = 0; i < d_maskImages.size(); ++i)
            gpuErrchk(cudaMalloc((void **)&d_maskImages.at(i), cellSize * sizeof(uchar)));
        copyMatToDevice<uchar>(detailCellShape.getCellMask(0, 0), d_maskImages.at(0));
        copyMatToDevice<uchar>(detailCellShape.getCellMask(1, 0), d_maskImages.at(1));
        copyMatToDevice<uchar>(detailCellShape.getCellMask(0, 1), d_maskImages.at(2));
        copyMatToDevice<uchar>(detailCellShape.getCellMask(1, 1), d_maskImages.at(3));

        //Copy image library to device
        gpuErrchk(cudaMalloc((void **)&d_libraryImage,
                             resizedLib.size() * cellSize * 3 * sizeof(float)));
        for (size_t libI = 0; libI < resizedLib.size(); ++libI)
            copyMatToDevice<float>(resizedLib.at(libI), d_libraryImage + libI * cellSize * 3);

        //Create device memory for variants
        gpuErrchk(cudaMalloc((void **)&d_variants, resizedLib.size() * cellSize * sizeof(double)));

        //Create device memory for cell image
        gpuErrchk(cudaMalloc((void **)&d_cellImage, cellSize * 3 * sizeof(float)));

        //Create device memory for target area
        gpuErrchk(cudaMalloc((void **)&d_targetArea, 4 * sizeof(size_t)));

        //Create device memory for reduction
        const size_t reductionMemorySize = (((cellSize + 1) / 2 + blockSize - 1) / blockSize + 1) / 2;
        gpuErrchk(cudaMalloc((void **)&d_reductionMem,
                             resizedLib.size() * reductionMemorySize * sizeof(double)));

        //Create device best fit
        const size_t noOfCells = m_bestFits.at(step).size() * m_bestFits.at(step).at(0).size();
        gpuErrchk(cudaMalloc((void **)&d_bestFit, noOfCells * sizeof(size_t)));
        gpuErrchk(cudaMemcpy(d_bestFit, &noOfCells, sizeof(size_t), cudaMemcpyHostToDevice));
        for (size_t cellIndex = 1; cellIndex < noOfCells; ++cellIndex)
            gpuErrchk(cudaMemcpy(d_bestFit + cellIndex, d_bestFit, sizeof(size_t),
                                 cudaMemcpyDeviceToDevice));

        //Create device repeats
        gpuErrchk(cudaMalloc((void **)&d_repeats, resizedLib.size() * sizeof(size_t)));

        //Create device lowest variant
        const double maxVariant = std::numeric_limits<double>::max();
        gpuErrchk(cudaMalloc((void **)&d_lowestVariant, sizeof(double)));

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
                    auto [cell, cellBounds] = getCellAt(normalCellShape, detailCellShape, x, y,
                                                        GridUtility::PAD_GRID, mainImage);

                    //Calculate if and how current cell is flipped
                    const auto flipState = GridUtility::getFlipStateAt(detailCellShape, x, y,
                                                                       GridUtility::PAD_GRID);
                    //Get mask image
                    uchar *d_mask = d_maskImages.at(flipState.horizontal + flipState.vertical * 2);

                    //Copy cell image to device
                    copyMatToDevice<float>(cell, d_cellImage);

                    //Copy target area to device
                    const size_t targetArea[4] = {static_cast<size_t>(cellBounds.y),
                                                  static_cast<size_t>(cellBounds.br().y),
                                                  static_cast<size_t>(cellBounds.x),
                                                  static_cast<size_t>(cellBounds.br().x)};
                    gpuErrchk(cudaMemcpy(d_targetArea, targetArea, 4 * sizeof(size_t),
                                         cudaMemcpyHostToDevice));

                    //Clear variants
                    gpuErrchk(cudaMemset(d_variants, 0,
                                         resizedLib.size() * cellSize * sizeof(double)));

                    //Clear reduction memory
                    gpuErrchk(cudaMemset(d_reductionMem, 0,
                                         resizedLib.size() * reductionMemorySize * sizeof(double)));

                    //Clear repeats
                    gpuErrchk(cudaMemset(d_repeats, 0,
                                         resizedLib.size() * sizeof(size_t)));

                    //Reset lowest variant
                    gpuErrchk(cudaMemcpy(d_lowestVariant, &maxVariant, sizeof(double),
                                         cudaMemcpyHostToDevice));

                    //Calculate difference
                    if (m_mode == Mode::CIEDE2000)
                        CIEDE2000DifferenceKernelWrapper(d_cellImage, d_libraryImage,
                                                         resizedLib.size(), d_mask, cell.rows,
                                                         cell.channels(), d_targetArea, d_variants,
                                                         blockSize);
                    else if (m_mode == Mode::RGB_EUCLIDEAN || m_mode == Mode::CIE76)
                        euclideanDifferenceKernelWrapper(d_cellImage, d_libraryImage,
                                                         resizedLib.size(), d_mask, cell.rows,
                                                         cell.channels(), d_targetArea, d_variants,
                                                         blockSize);
                    else
                        throw std::invalid_argument(Q_FUNC_INFO " Unsupported mode");

                    //Reduce difference
                    for (size_t libI = 0; libI < resizedLib.size(); ++libI)
                    {
                        reduceAddKernelWrapper(blockSize, cellSize,
                                               d_variants + libI * cellSize, d_reductionMem);
                        gpuErrchk(cudaMemcpy(d_variants + libI, d_variants + libI * cellSize,
                                             sizeof(double), cudaMemcpyDeviceToDevice));
                    }

                    //Calculate repeats
                    const int xMax = static_cast<int>(m_bestFits.at(step).at(0).size());
                    const int shiftedX = x + GridUtility::PAD_GRID;
                    const int shiftedY = y + GridUtility::PAD_GRID;
                    const size_t cellPosition = shiftedY * xMax + shiftedX;

                    const int leftRange = std::min(m_repeatRange, x);
                    const int rightRange = std::min(m_repeatRange, xMax - x - 1);
                    const int upRange = std::min(m_repeatRange, y);
                    calculateRepeatsKernelWrapper(d_bestFit + cellPosition, d_repeats, cell.rows,
                                                  leftRange, rightRange, upRange,
                                                  m_repeatAddition, resizedLib.size());
                    gpuErrchk(cudaPeekAtLastError());
                    gpuErrchk(cudaDeviceSynchronize());

                    //Add repeats to variants
                    addRepeatsKernelWrapper(d_variants, d_repeats, resizedLib.size(), blockSize);
                    gpuErrchk(cudaPeekAtLastError());
                    gpuErrchk(cudaDeviceSynchronize());

                    //Find lowest variant
                    findLowestKernelWrapper(d_lowestVariant, d_bestFit + cellPosition, d_variants,
                                            resizedLib.size());
                    gpuErrchk(cudaPeekAtLastError());
                    gpuErrchk(cudaDeviceSynchronize());

                    //Copy best fit to host
                    size_t bestFit = 0;
                    gpuErrchk(cudaMemcpy(&bestFit, d_bestFit + cellPosition, sizeof(size_t),
                                         cudaMemcpyDeviceToHost));

                    m_bestFits.at(step).at(y + GridUtility::PAD_GRID).
                        at(x + GridUtility::PAD_GRID) = bestFit;
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

    for (size_t i = 0; i < d_maskImages.size(); ++i)
        gpuErrchk(cudaFree(d_maskImages.at(i)));

    gpuErrchk(cudaFree(d_libraryImage));

    gpuErrchk(cudaFree(d_variants));
    gpuErrchk(cudaFree(d_cellImage));
    gpuErrchk(cudaFree(d_targetArea));
    gpuErrchk(cudaFree(d_reductionMem));

    gpuErrchk(cudaFree(d_bestFit));
    gpuErrchk(cudaFree(d_repeats));
    gpuErrchk(cudaFree(d_lowestVariant));

    return true;
}

//Set library batch size
void CUDAPhotomosaicGenerator::setLibraryBatchSize(const size_t t_libraryBatchSize)
{
    m_libraryBatchSize = t_libraryBatchSize;
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
