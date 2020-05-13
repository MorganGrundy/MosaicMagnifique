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
    const size_t reductionMemorySize = noLibraryImages * ((pixelCount + blockSize - 1) / blockSize
                                                          + 1) / 2;
    const size_t lowestVariantSize = 1;

    const size_t bestFitSize = noCellImages;
    const size_t repeatsSize = noLibraryImages;
    const size_t targetAreaSize = 4;

    //Calculate total memory needed on GPU
    totalMem = (cellImageSize + libraryImagesSize + maskImageSize) * sizeof(uchar)
            + (variantsSize + reductionMemorySize + lowestVariantSize) * sizeof(double)
            + (bestFitSize + repeatsSize + targetAreaSize) * sizeof(size_t);
    //If memory needed exceeds available memory then exit
    if (totalMem >= freeMem)
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
    //Batch allocate memory for cell image, library images, and mask image
    gpuErrchk(cudaMalloc(&mem, (cellImageSize + libraryImagesSize + maskImageSize)
                         * sizeof(uchar)));
    cellImage = static_cast<uchar *>(mem);
    libraryImages = cellImage + cellImageSize;
    maskImage = libraryImages + libraryImagesSize;

    //Batch allocate memory for variants, reduce memory, and lowest variant
    gpuErrchk(cudaMalloc(&mem, (variantsSize + reductionMemorySize + lowestVariantSize)
                         * sizeof(double)));
    variants = static_cast<double *>(mem);
    reductionMemory = variants + variantsSize;
    lowestVariant = reductionMemory + reductionMemorySize;

    //Batch allocate memory for best fit, repeats, and target area
    gpuErrchk(cudaMalloc(&mem, (bestFitSize + repeatsSize + targetAreaSize) * sizeof(size_t)));
    bestFit = static_cast<size_t *>(mem);
    repeats = bestFit + bestFitSize;
    targetArea = repeats + repeatsSize;

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
    if (static_cast<size_t>(currentBatchIndex) >= noCellImages)
    {
        currentBatchIndex = -1;
        return -1;
    }

    //Copy cell image and target area
    gpuErrchk(cudaMemcpy(cellImage, HOST_cellImages + currentBatchIndex * fullSize,
                         fullSize * sizeof(uchar), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(targetArea, HOST_targetAreas + currentBatchIndex * 4,
                         4 * sizeof(size_t), cudaMemcpyHostToDevice));

    clearVariants();
    clearRepeats();
    resetLowestVariant();

    return currentBatchIndex;
}

//Copies best fits from device to host and returns pointer
size_t *CUDAPhotomosaicData::getResults()
{
    gpuErrchk(cudaMemcpy(HOST_bestFit, bestFit, noCellImages * sizeof(size_t), cudaMemcpyDeviceToHost));
    return HOST_bestFit;
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

//Returns pointer to cell image on GPU
uchar *CUDAPhotomosaicData::getCellImage()
{
    return cellImage;
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

//Returns pointer to target area on GPU
size_t *CUDAPhotomosaicData::getTargetArea()
{
    return targetArea;
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

//Sets variants to 0
void CUDAPhotomosaicData::clearVariants()
{
    gpuErrchk(cudaMemset(variants, 0, pixelCount * noLibraryImages * sizeof(double)));
}

//Returns pointer to variants on GPU
double *CUDAPhotomosaicData::getVariants()
{
    return variants;
}

//Sets best fit to number of library images
void CUDAPhotomosaicData::resetBestFit()
{
    gpuErrchk(cudaMemcpy(bestFit, &noLibraryImages, sizeof(size_t), cudaMemcpyHostToDevice));
    for (size_t i = 1; i < noCellImages; ++i)
        gpuErrchk(cudaMemcpy(bestFit + i, bestFit, sizeof(size_t),
                             cudaMemcpyDeviceToDevice));
}

//Returns pointer to best fit on GPU
size_t *CUDAPhotomosaicData::getBestFit()
{
    return bestFit + currentBatchIndex;
}

//Sets lowest variant to max double
void CUDAPhotomosaicData::resetLowestVariant()
{
    double doubleMax = std::numeric_limits<double>::max();
    gpuErrchk(cudaMemcpy(lowestVariant, &doubleMax, sizeof(double), cudaMemcpyHostToDevice));
}

//Returns pointer to lowest variant on GPU
double *CUDAPhotomosaicData::getLowestVariant()
{
    return lowestVariant;
}

//Returns pointer to reduction memory on GPU
double *CUDAPhotomosaicData::getReductionMemory()
{
    return reductionMemory;
}
