#ifndef CUDAPHOTOMOSAICDATA_H
#define CUDAPHOTOMOSAICDATA_H

#include <vector>
#include <cuda.h>
#include <cuda_runtime_api.h>

#include <opencv2/core/mat.hpp>

class CUDAPhotomosaicData
{
public:
    CUDAPhotomosaicData(const size_t t_imageSize, const size_t t_imageChannels,
                        const size_t t_noLibraryImages, const bool t_euclidean);
    ~CUDAPhotomosaicData();

    //Allocates memory on GPU for Photomosaic data
    cudaError_t mallocData();

    //Frees memory on GPU
    cudaError_t freeData();

    //Copies cell image to GPU
    cudaError_t setCellImage(const cv::Mat &t_cellImage);

    //Copies library images to GPU
    cudaError_t setLibraryImages(const std::vector<cv::Mat> &t_libraryImages);

    //Copies mask image to GPU
    cudaError_t setMaskImage(const cv::Mat &t_maskImage);

    //Copies target area to GPU
    cudaError_t setTargetArea(const size_t (&t_targetArea)[4]);

    //Copies repeats to GPU
    cudaError_t setRepeats(const size_t *t_repeats);

    //Sets variants to 0
    cudaError_t clearVariants();

    //Sets best fit to number of library images
    cudaError_t resetBestFit();

    //Sets lowest variant to max double
    cudaError_t resetLowestVariant();

    //-------------------------------------
    size_t blockSize;

    //Stores image data
    uchar *cellImage, *libraryImages, *maskImage;

    double *variants, *reductionMemory, *lowestVariant;
    size_t *bestFit;

    size_t *targetArea;
    const size_t imageSize, imageChannels, noLibraryImages;

    size_t *repeats;

    const bool euclidean;
};

#endif // CUDAPHOTOMOSAICDATA_H
