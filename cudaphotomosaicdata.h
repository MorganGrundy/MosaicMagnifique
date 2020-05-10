#ifndef CUDAPHOTOMOSAICDATA_H
#define CUDAPHOTOMOSAICDATA_H

#include <vector>
#include <cuda.h>
#include <cuda_runtime_api.h>

#include <opencv2/core/mat.hpp>

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true)
{
   if (code != cudaSuccess)
   {
      fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort)
          exit(code);
   }
}

class CUDAPhotomosaicData
{
public:
    CUDAPhotomosaicData(const size_t t_imageSize, const size_t t_imageChannels,
                        const size_t t_noLibraryImages, const bool t_euclidean);
    ~CUDAPhotomosaicData();

    //Allocates memory on GPU for Photomosaic data
    void mallocData();

    //Frees memory on GPU
    void freeData();

    //Returns block size
    size_t getBlockSize();

    //Copies cell image to GPU
    void setCellImage(const cv::Mat &t_cellImage);
    //Returns pointer to cell image on GPU
    uchar *getCellImage();

    //Copies library images to GPU
    void setLibraryImages(const std::vector<cv::Mat> &t_libraryImages);
    //Returns pointer to library images on GPU
    uchar *getLibraryImages();

    //Copies mask image to GPU
    void setMaskImage(const cv::Mat &t_maskImage);
    //Returns pointer to mask image on GPU
    uchar *getMaskImage();

    //Copies target area to GPU
    void setTargetArea(const size_t (&t_targetArea)[4]);
    //Returns pointer to target area on GPU
    size_t *getTargetArea();

    //Copies repeats to GPU
    void setRepeats(const size_t *t_repeats);
    //Returns pointer to repeats on GPU
    size_t *getRepeats();

    //Sets variants to 0
    void clearVariants();
    //Returns pointer to variants on GPU
    double *getVariants();

    //Sets best fit to number of library images
    void resetBestFit();
    //Returns pointer to best fit on GPU
    size_t *getBestFit();

    //Sets lowest variant to max double
    void resetLowestVariant();
    //Returns pointer to lowest variant on GPU
    double *getLowestVariant();

    //Returns pointer to reduction memory on GPU
    double *getReductionMemory();

    //-------------------------------------
    const size_t imageSize; //Size of images (width == height)
    const size_t imageChannels; //Channels in main and library images

    const size_t pixelCount; //Number of pixels in images (width * height)
    const size_t fullSize; //Number of uchar in images (width * height * channels)

    const size_t noLibraryImages; //Number of library images

    const bool euclidean; //Which difference formula to use euclidean (true) or CIEDE2000 (false)

private:
    bool dataIsAllocated; //Stores if any data has been allocated on GPU

    size_t blockSize; //Number of threads per block

    uchar *cellImage; //Stores a cell from the main image
    uchar *libraryImages; //Stores all library images
    uchar *maskImage; //Stores mask image

    size_t *targetArea; //Stores bounds of cell (as custom cell shapes can exceed image bounds)

    double *variants; //Stores results of difference formula for each pixel
    double *reductionMemory; //Memory used to perform sum reduction on variants

    double *lowestVariant; //Stores the lowest variant seen
    size_t *bestFit; //Stores the index of the lowest variant seen

    size_t *repeats; //Stores for each library image the value to add to variant for current cell
};

#endif // CUDAPHOTOMOSAICDATA_H
