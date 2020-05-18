#include "cudaphotomosaicdata.h"

#include <limits>

CUDAPhotomosaicData::CUDAPhotomosaicData(const size_t t_imageSize, const size_t t_imageChannels,
                                         const size_t t_noXCellImages, const size_t t_noYCellImages,
                                         const size_t t_noLibraryImages,
                                         const bool t_euclidean,
                                         const size_t t_repeatRange, const size_t t_repeatAddition)
    : imageSize{t_imageSize}, imageChannels{t_imageChannels},
      pixelCount{t_imageSize * t_imageSize}, fullSize{t_imageSize * t_imageSize * t_imageChannels},
      noXCellImages{t_noXCellImages}, noYCellImages{t_noYCellImages},
      noCellImages{t_noXCellImages * t_noYCellImages},
      noLibraryImages{t_noLibraryImages},
      euclidean{t_euclidean},
      repeatRange{t_repeatRange}, repeatAddition{t_repeatAddition},
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
    const size_t maskImageSize = pixelCount;

    const size_t variantsSize = pixelCount * noLibraryImages;
    reductionMemorySize = noLibraryImages * ((pixelCount + blockSize - 1) / blockSize + 1) / 2;
    const size_t maxVariantSize = 1;
    const size_t lowestVariantSize = 1;

    const size_t bestFitSize = noCellImages;
    const size_t repeatsSize = noLibraryImages;
    const size_t targetAreaSize = 4;

    const size_t staticMemory = (libraryImagesSize + maskImageSize) * sizeof(uchar)
            + (maxVariantSize) * sizeof(double)
            + (bestFitSize + repeatsSize) * sizeof(size_t);

    const size_t batchScalingMemory = (cellImageSize) * sizeof(uchar)
            + (variantsSize + reductionMemorySize + lowestVariantSize) * sizeof(double)
            + (targetAreaSize) * sizeof(size_t);

    //Calculate max batch size possible with given memory
    batchSize = noCellImages;
    while (staticMemory + batchScalingMemory * batchSize >= freeMem && batchSize != 0)
        --batchSize;
    //If memory needed exceeds available memory then exit
    if (batchSize == 0)
    {
        fprintf(stderr, "Not enough memory available on GPU to generate Photomosaic\n");
        return false;
    }

    //Allocate host memory for cell images, target areas, and best fits
    if (!(HOST_cellImages = static_cast<uchar *>(malloc(cellImageSize * noCellImages
                                                        * sizeof(uchar)))))
    {
        fprintf(stderr, "Failed to allocate host memory for cell images\n");
        return false;
    }
    if (!(HOST_targetAreas = static_cast<size_t *>(malloc(targetAreaSize * noCellImages
                                                          * sizeof(size_t)))))
    {
        fprintf(stderr, "Failed to allocate host memory for target areas\n");
        free(HOST_cellImages);
        return false;
    }
    if (!(HOST_bestFit = static_cast<size_t *>(malloc(noCellImages * sizeof(size_t)))))
    {
        fprintf(stderr, "Failed to allocate host memory for best fits\n");
        free(HOST_cellImages);
        free(HOST_targetAreas);
        return false;
    }

    void *mem;
    //Batch allocate memory for cell images, library images, and mask image
    gpuErrchk(cudaMalloc(&mem, (cellImageSize * batchSize + libraryImagesSize + maskImageSize)
                         * sizeof(uchar)));
    cellImage = static_cast<uchar *>(mem);
    libraryImages = cellImage + cellImageSize * batchSize;
    maskImage = libraryImages + libraryImagesSize;

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
        free(HOST_cellImages);
        free(HOST_targetAreas);
        free(HOST_bestFit);

        //Free uchar memory (cell image, library images, and mask image)
        gpuErrchk(cudaFree(cellImage));

        //Free double memory (variants, reduction memory, lowest variant)
        gpuErrchk(cudaFree(variants));

        //Free size_t memory (best fit, repeats, target
        gpuErrchk(cudaFree(bestFit));

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
    if (static_cast<size_t>(currentBatchIndex * batchSize) >= noCellImages)
    {
        currentBatchIndex = -1;
        return -1;
    }

    if (currentBatchIndex == 0)
        resetBestFit();

    //Copy cell image and target area
    const size_t startOffset = batchSize * currentBatchIndex;
    const size_t offset = std::min(noCellImages - startOffset, batchSize);
    gpuErrchk(cudaMemcpy(cellImage, HOST_cellImages + startOffset * fullSize,
                         offset * fullSize * sizeof(uchar), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(targetArea, HOST_targetAreas + startOffset * 4,
                         offset * 4 * sizeof(size_t), cudaMemcpyHostToDevice));

    resetLowestVariant();

    return currentBatchIndex;
}

//Copies best fits from device to host and returns pointer
size_t *CUDAPhotomosaicData::getResults()
{
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

//Copies cell image to host memory at index i
void CUDAPhotomosaicData::setCellImage(const cv::Mat &t_cellImage, const size_t i)
{
    memcpy(HOST_cellImages + i * fullSize, t_cellImage.data, fullSize * sizeof(uchar));
}

//Returns pointer to cell image i on GPU
uchar *CUDAPhotomosaicData::getCellImage(const size_t i)
{
    return cellImage + i * fullSize;
}

//Copies library images to GPU
void CUDAPhotomosaicData::setLibraryImages(const std::vector<cv::Mat> &t_libraryImages)
{
    for (size_t i = 0; i < noLibraryImages; ++i)
    {
        gpuErrchk(cudaMemcpy(libraryImages + i * fullSize, t_libraryImages.at(i).data,
                             fullSize * sizeof(uchar), cudaMemcpyHostToDevice));
    }
}

//Returns pointer to library images on GPU
uchar *CUDAPhotomosaicData::getLibraryImages()
{
    return libraryImages;
}

//Copies mask image to GPU
void CUDAPhotomosaicData::setMaskImage(const cv::Mat &t_maskImage)
{
    gpuErrchk(cudaMemcpy(maskImage, t_maskImage.data, pixelCount * sizeof(uchar),
                         cudaMemcpyHostToDevice));
}

//Returns pointer to mask image on GPU
uchar *CUDAPhotomosaicData::getMaskImage()
{
    return maskImage;
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
    return bestFit + currentBatchIndex * batchSize + i;
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
