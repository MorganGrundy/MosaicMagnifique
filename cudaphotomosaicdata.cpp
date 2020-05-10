#include "cudaphotomosaicdata.h"

#include <limits>

CUDAPhotomosaicData::CUDAPhotomosaicData(const size_t t_imageSize, const size_t t_imageChannels,
                                         const size_t t_noLibraryImages, const bool t_euclidean)
    : imageSize{t_imageSize}, imageChannels{t_imageChannels},
      pixelCount{t_imageSize * t_imageSize}, fullSize{t_imageSize * t_imageSize * t_imageChannels},
      noLibraryImages{t_noLibraryImages},
      euclidean{t_euclidean},
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
void CUDAPhotomosaicData::mallocData()
{
    void *mem;
    //Batch allocate memory for cell image, library images, and mask image
    gpuErrchk(cudaMalloc(&mem, (fullSize + fullSize * noLibraryImages + pixelCount)
                         * sizeof(uchar)));
    cellImage = static_cast<uchar *>(mem);
    libraryImages = cellImage + fullSize;
    maskImage = libraryImages + fullSize * noLibraryImages;

    const size_t reduceMemSize = (pixelCount + blockSize - 1) / blockSize;
    //Batch allocate memory for variants, reduce memory, and lowest variant
    gpuErrchk(cudaMalloc(&mem, (pixelCount * noLibraryImages + reduceMemSize * noLibraryImages + 1)
                         * sizeof(double)));
    variants = static_cast<double *>(mem);
    reductionMemory = variants + pixelCount * noLibraryImages;
    lowestVariant = reductionMemory + reduceMemSize * noLibraryImages;

    //Batch allocate memory for best fit, repeats, and target area
    gpuErrchk(cudaMalloc(&mem, (1 + noLibraryImages + 4) * sizeof(size_t)));
    bestFit = static_cast<size_t *>(mem);
    repeats = bestFit + 1;
    targetArea = repeats + noLibraryImages;

    dataIsAllocated = true;
}

//Frees memory on GPU
void CUDAPhotomosaicData::freeData()
{
    if (dataIsAllocated)
    {
        //Free uchar memory (cell image, library images, and mask image)
        gpuErrchk(cudaFree(cellImage));

        //Free double memory (variants, reduction memory, lowest variant)
        gpuErrchk(cudaFree(variants));

        //Free size_t memory (best fit, repeats, target
        gpuErrchk(cudaFree(bestFit));

        dataIsAllocated = false;
    }
}

//Returns block size
size_t CUDAPhotomosaicData::getBlockSize()
{
    return blockSize;
}

//Copies cell image to GPU
void CUDAPhotomosaicData::setCellImage(const cv::Mat &t_cellImage)
{
    gpuErrchk(cudaMemcpy(cellImage, t_cellImage.data, fullSize * sizeof(uchar),
                         cudaMemcpyHostToDevice));
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
        const size_t offset = i * fullSize;
        gpuErrchk(cudaMemcpy(libraryImages + offset, t_libraryImages.at(i).data,
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

//Copies target area to GPU
void CUDAPhotomosaicData::setTargetArea(const size_t (&t_targetArea)[4])
{
    gpuErrchk(cudaMemcpy(targetArea, t_targetArea, 4 * sizeof(size_t), cudaMemcpyHostToDevice));
}

//Returns pointer to target area on GPU
size_t *CUDAPhotomosaicData::getTargetArea()
{
    return targetArea;
}

//Copies repeats to GPU
void CUDAPhotomosaicData::setRepeats(const size_t *t_repeats)
{
    gpuErrchk(cudaMemcpy(repeats, t_repeats, noLibraryImages * sizeof(size_t),
                         cudaMemcpyHostToDevice));
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
}

//Returns pointer to best fit on GPU
size_t *CUDAPhotomosaicData::getBestFit()
{
    return bestFit;
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
