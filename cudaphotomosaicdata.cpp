#include "cudaphotomosaicdata.h"

#include <limits>

CUDAPhotomosaicData::CUDAPhotomosaicData(const size_t t_imageSize, const size_t t_imageChannels,
                                         const size_t t_noLibraryImages, const bool t_euclidean)
    : imageSize{t_imageSize}, imageChannels{t_imageChannels}, noLibraryImages{t_noLibraryImages},
      euclidean{t_euclidean}
{
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);
    blockSize = deviceProp.maxThreadsPerBlock;
}

CUDAPhotomosaicData::~CUDAPhotomosaicData()
{
    freeData();
}

//Allocates memory on GPU for Photomosaic data
cudaError_t CUDAPhotomosaicData::mallocData()
{
    cudaError_t result;

    const size_t pixelCount = imageSize * imageSize;
    const size_t fullSize = pixelCount * imageChannels;

    void *mem;
    //Batch allocate memory for cell image, library images, and mask image
    if ((result = cudaMalloc(&mem, (fullSize + fullSize * noLibraryImages + pixelCount)
                             * sizeof(uchar))) != cudaSuccess)
        return result;

    //Cell image
    cellImage = static_cast<uchar *>(mem);
    //Library images
    libraryImages = cellImage + fullSize;
    //Mask images
    maskImage = libraryImages + fullSize * noLibraryImages;

    const size_t reduceMemSize = (pixelCount + blockSize - 1) / blockSize;
    //Batch allocate memory for variants, reduce memory, and lowest variant
    if ((result = cudaMalloc(&mem, (pixelCount * noLibraryImages + reduceMemSize * noLibraryImages
                                    + 1) * sizeof(double))) != cudaSuccess)
        return result;

    //Variants
    variants = static_cast<double *>(mem);
    //Reduction memory
    reductionMemory = variants + pixelCount * noLibraryImages;
    //Lowest variant
    lowestVariant = reductionMemory + reduceMemSize * noLibraryImages;

    //Batch allocate memory for best fit, repeats, and target area
    if ((result = cudaMalloc(&mem, (1 + noLibraryImages + 4) * sizeof(size_t))) != cudaSuccess)
        return result;

    //Best fit
    bestFit = static_cast<size_t *>(mem);
    //Repeats
    repeats = bestFit + 1;
    //Target area
    targetArea = repeats + noLibraryImages;

    return result;
}

//Frees memory on GPU
cudaError_t CUDAPhotomosaicData::freeData()
{
    cudaError_t result;

    //Free uchar memory (cell image, library images, and mask image)
    if ((result = cudaFree(cellImage)) != cudaSuccess)
        return result;

    //Free double memory (variants, reduction memory, lowest variant)
    if ((result = cudaFree(variants)) != cudaSuccess)
        return result;

    //Free size_t memory (best fit, repeats, target
    if ((result = cudaFree(bestFit)) != cudaSuccess)
        return result;

    return result;
}

//Copies cell image to GPU
cudaError_t CUDAPhotomosaicData::setCellImage(const cv::Mat &t_cellImage)
{
    cudaError_t result;

    const size_t fullSize = imageSize * imageSize * imageChannels;
    result = cudaMemcpy(cellImage, t_cellImage.data, fullSize * sizeof(uchar),
                        cudaMemcpyHostToDevice);
    return result;
}

//Copies library images to GPU
cudaError_t CUDAPhotomosaicData::setLibraryImages(const std::vector<cv::Mat> &t_libraryImages)
{
    cudaError_t result;

    const size_t fullSize = imageSize * imageSize * imageChannels;
    for (size_t i = 0; i < noLibraryImages; ++i)
    {
        const size_t offset = i * fullSize;
        if ((result = cudaMemcpy(libraryImages+offset, t_libraryImages.at(i).data,
                                 fullSize * sizeof(uchar), cudaMemcpyHostToDevice)) != cudaSuccess)
            return result;
    }
    return result;
}

//Copies mask image to GPU
cudaError_t CUDAPhotomosaicData::setMaskImage(const cv::Mat &t_maskImage)
{
    cudaError_t result;
    const size_t pixelCount = imageSize * imageSize;
    result = cudaMemcpy(maskImage, t_maskImage.data, pixelCount * sizeof(uchar),
                        cudaMemcpyHostToDevice);
    return result;
}

//Copies target area to GPU
cudaError_t CUDAPhotomosaicData::setTargetArea(const size_t (&t_targetArea)[4])
{
    cudaError_t result;
    result = cudaMemcpy(targetArea, t_targetArea, 4 * sizeof(size_t), cudaMemcpyHostToDevice);
    return result;
}

//Copies repeats to GPU
cudaError_t CUDAPhotomosaicData::setRepeats(const size_t *t_repeats)
{
    cudaError_t result;
    result = cudaMemcpy(repeats, t_repeats, noLibraryImages * sizeof(size_t),
                        cudaMemcpyHostToDevice);
    return result;
}

//Sets variants to 0
cudaError_t CUDAPhotomosaicData::clearVariants()
{
    cudaError_t result;
    const size_t pixelCount = imageSize * imageSize;
    result = cudaMemset(variants, 0, pixelCount * noLibraryImages * sizeof(double));
    return result;
}

//Sets best fit to number of library images
cudaError_t CUDAPhotomosaicData::resetBestFit()
{
    cudaError_t result;
    result = cudaMemcpy(bestFit, &noLibraryImages, sizeof(size_t), cudaMemcpyHostToDevice);
    return result;
}

//Sets lowest variant to max double
cudaError_t CUDAPhotomosaicData::resetLowestVariant()
{
    cudaError_t result;
    double doubleMax = std::numeric_limits<double>::max();
    result = cudaMemcpy(lowestVariant, &doubleMax, sizeof(double), cudaMemcpyHostToDevice);
    return result;
}
