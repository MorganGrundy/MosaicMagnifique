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

#include "cudaphotomosaicdata.h"

#include <limits>

CUDAPhotomosaicData::CUDAPhotomosaicData(const size_t t_imageSize, const size_t t_imageChannels,
                                         const size_t t_noXCellImages, const size_t t_noYCellImages,
                                         const size_t t_noValidCells,
                                         const size_t t_noLibraryImages,
                                         const size_t t_libraryBatchSize,
                                         const bool t_euclidean,
                                         const size_t t_repeatRange, const size_t t_repeatAddition)
    : imageSize{t_imageSize}, imageChannels{t_imageChannels},
      pixelCount{t_imageSize * t_imageSize}, fullSize{t_imageSize * t_imageSize * t_imageChannels},
      noXCellImages{t_noXCellImages}, noYCellImages{t_noYCellImages},
      noCellImages{t_noXCellImages * t_noYCellImages},
      noValidCells{t_noValidCells},
      noLibraryImages{t_noLibraryImages},
      euclidean{t_euclidean},
      repeatRange{t_repeatRange}, repeatAddition{t_repeatAddition},
      libraryBatchSize{t_libraryBatchSize},
      dataIsAllocated{false}
{
    cudaDeviceProp deviceProp;
    gpuErrchk(cudaGetDeviceProperties(&deviceProp, 0));
    blockSize = deviceProp.maxThreadsPerBlock;
    batchSize = 0;
}

CUDAPhotomosaicData::~CUDAPhotomosaicData()
{
    freeData();
}

//Allocates memory on GPU for Photomosaic data
bool CUDAPhotomosaicData::mallocData()
{
    freeData();

    size_t freeMem, totalMem;
    gpuErrchk(cudaMemGetInfo(&freeMem, &totalMem));

    const size_t cellImageSize = fullSize;
    const size_t libraryImagesSize = fullSize * noLibraryImages;
    const size_t maskImagesSize = pixelCount * 4;

    const size_t variantsSize = pixelCount * noLibraryImages;
    reductionMemorySize = noLibraryImages * ((pixelCount + blockSize - 1) / blockSize + 1) / 2;
    const size_t maxVariantSize = 1;
    const size_t lowestVariantSize = 1;

    const size_t bestFitSize = noCellImages;
    const size_t repeatsSize = noLibraryImages;
    const size_t targetAreaSize = 4;

    const size_t cellStateSize = noCellImages;

    const size_t staticMemory = (maskImagesSize) * sizeof(uchar)
                                + (libraryImagesSize) * sizeof(float)
                                + (maxVariantSize) * sizeof(double)
                                + (bestFitSize + repeatsSize) * sizeof(size_t)
                                + (cellStateSize) * sizeof(bool);

    const size_t batchScalingMemory = (cellImageSize) * sizeof(float)
            + (variantsSize + reductionMemorySize + lowestVariantSize) * sizeof(double)
            + (targetAreaSize) * sizeof(size_t);

    //Calculate max batch size possible with given memory
    batchSize = noValidCells;
    while (staticMemory + batchScalingMemory * batchSize >= 3 * (freeMem / 4) && batchSize != 0)
        --batchSize;
    //If memory needed exceeds available memory then exit
    if (batchSize == 0)
    {
        fprintf(stderr, "Not enough memory available on GPU to generate Photomosaic\n");
        return false;
    }
    //Calculate batch size <= max batch size such that the data load is evenly spread
    noOfBatch = (noValidCells + batchSize - 1) / batchSize;
    batchSize = (noValidCells + noOfBatch - 1) / noOfBatch;

    HOST_cellPositions.resize(noValidCells);

    //Allocate host memory for cell states, cell images, target areas, and best fits
    if (!(HOST_cellStates = static_cast<bool *>(malloc(cellStateSize * sizeof(bool)))))
    {
        fprintf(stderr, "Failed to allocate host memory for cell states\n");
        return false;
    }
    if (!(HOST_cellImages = static_cast<float *>(malloc(cellImageSize * noValidCells
                                                        * sizeof(float)))))
    {
        fprintf(stderr, "Failed to allocate host memory for cell images\n");
        free(HOST_cellStates);
        return false;
    }
    if (!(HOST_targetAreas = static_cast<size_t *>(malloc(targetAreaSize * noValidCells
                                                          * sizeof(size_t)))))
    {
        fprintf(stderr, "Failed to allocate host memory for target areas\n");
        free(HOST_cellStates);
        free(HOST_cellImages);
        return false;
    }
    if (!(HOST_bestFit = static_cast<size_t *>(malloc(bestFitSize * sizeof(size_t)))))
    {
        fprintf(stderr, "Failed to allocate host memory for best fits\n");
        free(HOST_cellStates);
        free(HOST_cellImages);
        free(HOST_targetAreas);
        return false;
    }

    void *mem;
    //Batch allocate memory for cell images, library images, and mask image
    gpuErrchk(cudaMalloc(&mem, maskImagesSize * sizeof(uchar)));
    maskImages = static_cast<uchar *>(mem);

    //Batch allocate memory for cell images, library images, and mask image
    gpuErrchk(cudaMalloc(&mem, (cellImageSize * batchSize + libraryImagesSize + maskImagesSize)
                         * sizeof(float)));
    cellImage = static_cast<float *>(mem);
    libraryImages = cellImage + cellImageSize * batchSize;

    //Batch allocate memory for variants, reduce memory, max variant and lowest variants
    gpuErrchk(cudaMalloc(&mem, (maxVariantSize
                                + (variantsSize + reductionMemorySize + lowestVariantSize)
                                * batchSize) * sizeof(double)));
    variants = static_cast<double *>(mem);
    reductionMemory = variants + variantsSize * batchSize;
    maxVariant = reductionMemory + reductionMemorySize * batchSize;
    lowestVariant = maxVariant + maxVariantSize;

    //Batch allocate memory for best fits, repeats, and target areas
    gpuErrchk(cudaMalloc(&mem, (bestFitSize + repeatsSize + targetAreaSize * batchSize)
                         * sizeof(size_t)));
    bestFit = static_cast<size_t *>(mem);
    repeats = bestFit + bestFitSize;
    targetArea = repeats + repeatsSize;

    //Batch allocate memory for cell states
    gpuErrchk(cudaMalloc(&mem, cellStateSize * sizeof(bool)));
    cellStates = static_cast<bool *>(mem);

    //Set max variant
    const double doubleMax = std::numeric_limits<double>::max();
    gpuErrchk(cudaMemcpy(maxVariant, &doubleMax, sizeof(double), cudaMemcpyHostToDevice));

    dataIsAllocated = true;
    currentBatchIndex = -1;
    return true;
}

//Frees memory on GPU
void CUDAPhotomosaicData::freeData()
{
    if (dataIsAllocated)
    {
        //Free host memory
        free(HOST_cellStates);
        free(HOST_cellImages);
        free(HOST_targetAreas);
        free(HOST_bestFit);

        //Free uchar memory (mask image)
        gpuErrchk(cudaFree(maskImages));

        //Free float memory (cell image, library images)
        gpuErrchk(cudaFree(cellImage));

        //Free double memory (variants, reduction memory, lowest variant)
        gpuErrchk(cudaFree(variants));

        //Free size_t memory (best fit, repeats, target area)
        gpuErrchk(cudaFree(bestFit));

        //Free bool memory (cell states)
        gpuErrchk(cudaFree(cellStates));

        dataIsAllocated = false;
        currentBatchIndex = -1;
    }
}

//Copies next batch of host data to device, returns batch index
int CUDAPhotomosaicData::copyNextBatchToDevice()
{
    if (!dataIsAllocated)
        return -1;

    ++currentBatchIndex;
    if (static_cast<size_t>(currentBatchIndex) == noOfBatch)
    {
        currentBatchIndex = -1;
        return -1;
    }

    if (currentBatchIndex == 0)
        resetBestFit();

    //Copy cell image and target area
    const size_t startOffset = batchSize * currentBatchIndex;
    const size_t offset = std::min(noValidCells - startOffset, batchSize);
    gpuErrchk(cudaMemcpy(cellImage, HOST_cellImages + startOffset * fullSize,
                         offset * fullSize * sizeof(float), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(targetArea, HOST_targetAreas + startOffset * 4,
                         offset * 4 * sizeof(size_t), cudaMemcpyHostToDevice));

    resetLowestVariant();

    return currentBatchIndex;
}

#include <iostream>
size_t continuousIm = 0;
size_t incontinuousIm = 0;

//Copies best fits from device to host and returns pointer
size_t *CUDAPhotomosaicData::getResults()
{
    std::cout << "Continuous images " << continuousIm << " vs " << incontinuousIm << '\n' << std::flush;
    continuousIm = 0;
    incontinuousIm = 0;
    gpuErrchk(cudaMemcpy(HOST_bestFit, bestFit, noCellImages * sizeof(size_t),
                         cudaMemcpyDeviceToHost));
    return HOST_bestFit;
}

//Returns batch size
size_t CUDAPhotomosaicData::getBatchSize()
{
    return batchSize;
}

//Returns current batch index
size_t CUDAPhotomosaicData::getBatchIndex()
{
    return currentBatchIndex;
}

//Returns block size
size_t CUDAPhotomosaicData::getBlockSize()
{
    return blockSize;
}

//Set cell position at given index
void CUDAPhotomosaicData::setCellPosition(const size_t x, const size_t y, const size_t i)
{
    HOST_cellPositions.at(i) = {x, y};
}

//Returns cell position at given index
std::pair<size_t, size_t> CUDAPhotomosaicData::getCellPosition(const size_t i)
{
    return HOST_cellPositions.at(currentBatchIndex * batchSize + i);
}

//Set cell state at given position to given state
void CUDAPhotomosaicData::setCellState(const int x, const int y, const bool t_cellState)
{
    const size_t index = y * noXCellImages + x;
    HOST_cellStates[index] = t_cellState;
}

//Returns cell state
bool CUDAPhotomosaicData::getCellState(const size_t i)
{
    auto [x, y] = getCellPosition(i);
    const size_t offset = y * noXCellImages + x;
    return HOST_cellStates[offset];
}

//Copies cell state to GPU
void CUDAPhotomosaicData::copyCellState()
{
    gpuErrchk(cudaMemcpy(cellStates, HOST_cellStates, noCellImages * sizeof(bool),
                         cudaMemcpyHostToDevice));
}

//Returns pointer to cell state on GPU
bool *CUDAPhotomosaicData::getCellStateGPU(const size_t i)
{
    auto [x, y] = getCellPosition(i);
    const size_t offset = y * noXCellImages + x;
    return cellStates + offset;
}

//Copies cell image to host memory at index i
void CUDAPhotomosaicData::setCellImage(const cv::Mat &t_cellImage, const size_t i)
{
    if (t_cellImage.isContinuous())
        ++continuousIm;
    else
        ++incontinuousIm;
    memcpy(HOST_cellImages + i * fullSize, t_cellImage.data, fullSize * sizeof(float));
}

//Returns pointer to cell image i on GPU
float *CUDAPhotomosaicData::getCellImage(const size_t i)
{
    return cellImage + i * fullSize;
}

//Copies library images to GPU
void CUDAPhotomosaicData::setLibraryImages(const std::vector<cv::Mat> &t_libraryImages)
{
    for (size_t i = 0; i < noLibraryImages; ++i)
    {
        if (t_libraryImages.at(i).isContinuous())
            ++continuousIm;
        else
            ++incontinuousIm;
        gpuErrchk(cudaMemcpy(libraryImages + i * fullSize, t_libraryImages.at(i).data,
                             fullSize * sizeof(float), cudaMemcpyHostToDevice));
    }
}

//Returns pointer to library image on GPU
float *CUDAPhotomosaicData::getLibraryImage(const size_t i)
{
    return libraryImages + i * fullSize;
}

//Copies mask image to GPU
void CUDAPhotomosaicData::setMaskImage(const cv::Mat &t_maskImage, const bool t_flippedHorizontal,
                                       const bool t_flippedVertical)
{
    if (t_maskImage.isContinuous())
        ++continuousIm;
    else
        ++incontinuousIm;
    const size_t offset = pixelCount * (t_flippedHorizontal + t_flippedVertical * 2);
    gpuErrchk(cudaMemcpy(maskImages + offset, t_maskImage.data, pixelCount * sizeof(uchar),
                         cudaMemcpyHostToDevice));
}

//Set if and how cells should flip on alternate rows/columns
void CUDAPhotomosaicData::setFlipStates(const bool t_colFlipHorizontal,
                                        const bool t_colFlipVertical,
                                        const bool t_rowFlipHorizontal,
                                        const bool t_rowFlipVertical)
{
    m_colFlipHorizontal = t_colFlipHorizontal;
    m_colFlipVertical = t_colFlipVertical;
    m_rowFlipHorizontal = t_rowFlipHorizontal;
    m_rowFlipVertical = t_rowFlipVertical;
}

//Returns pointer to mask image on GPU
uchar *CUDAPhotomosaicData::getMaskImage(const int t_gridX, const int t_gridY)
{
    //Calculate if and how current cell is flipped
    bool flipHorizontal = false, flipVertical = false;
    if (m_colFlipHorizontal && t_gridX % 2 == 1)
        flipHorizontal = !flipHorizontal;
    if (m_rowFlipHorizontal && t_gridY % 2 == 1)
        flipHorizontal = !flipHorizontal;
    if (m_colFlipVertical && t_gridX % 2 == 1)
        flipVertical = !flipVertical;
    if (m_rowFlipVertical && t_gridY % 2 == 1)
        flipVertical = !flipVertical;

    const size_t offset = pixelCount * (flipHorizontal + flipVertical * 2);
    return maskImages + offset;
}

//Copies target area to host memory at index i
void CUDAPhotomosaicData::setTargetArea(const size_t (&t_targetArea)[4], const size_t i)
{
    memcpy(HOST_targetAreas + i * 4, t_targetArea, 4 * sizeof(size_t));
}

//Returns pointer to target area i on GPU
size_t *CUDAPhotomosaicData::getTargetArea(const size_t i)
{
    return targetArea + i * 4;
}

//Sets repeats to 0
void CUDAPhotomosaicData::clearRepeats()
{
    gpuErrchk(cudaMemset(repeats, 0, noLibraryImages * sizeof(size_t)));
}

//Returns pointer to repeats on GPU
size_t *CUDAPhotomosaicData::getRepeats()
{
    return repeats;
}

//Returns pointer to variants on GPU
double *CUDAPhotomosaicData::getVariants(const size_t i)
{
    return variants + i * pixelCount * noLibraryImages;
}

//Returns pointer to variants on GPU
double *CUDAPhotomosaicData::getVariants(const size_t t_cellIndex, const size_t t_libIndex)
{
    return variants + t_cellIndex * pixelCount * noLibraryImages + t_libIndex * pixelCount;
}

//Sets best fit to number of library images
void CUDAPhotomosaicData::resetBestFit()
{
    gpuErrchk(cudaMemcpy(bestFit, &noLibraryImages, sizeof(size_t), cudaMemcpyHostToDevice));
    for (size_t i = 1; i < noCellImages; ++i)
        gpuErrchk(cudaMemcpy(bestFit + i, bestFit, sizeof(size_t), cudaMemcpyDeviceToDevice));
}

//Returns pointer to best fit on GPU
size_t *CUDAPhotomosaicData::getBestFit(const size_t i)
{
    auto [x, y] = getCellPosition(i);
    const size_t offset = y * noXCellImages + x;
    return bestFit + offset;
}

//Sets lowest variant to max double
void CUDAPhotomosaicData::resetLowestVariant()
{
    for (size_t i = 0; i < batchSize; ++i)
        gpuErrchk(cudaMemcpy(lowestVariant + i, maxVariant, sizeof(double),
                             cudaMemcpyDeviceToDevice));
}

//Returns pointer to lowest variant i on GPU
double *CUDAPhotomosaicData::getLowestVariant(const size_t i)
{
    return lowestVariant + i;
}

//Returns pointer to reduction memory on GPU
double *CUDAPhotomosaicData::getReductionMemory(const size_t i)
{
    return reductionMemory + i * reductionMemorySize;
}
