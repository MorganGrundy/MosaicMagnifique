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
                        const size_t t_noXCellImages, const size_t t_noYCellImages,
                        const size_t t_noLibraryImages,
                        const bool t_euclidean,
                        const size_t t_repeatRange, const size_t t_repeatAddition);
    ~CUDAPhotomosaicData();

    //Allocates memory on GPU for Photomosaic data
    bool mallocData();

    //Frees memory on GPU
    void freeData();

    //Copies next batch of host data to device, returns batch index
    int copyNextBatchToDevice();

    //Copies best fits from device to host and returns pointer
    size_t *getResults();

    //Returns batch size
    size_t getBatchSize();

    //Returns current batch index
    size_t getBatchIndex();

    //Returns block size
    size_t getBlockSize();

    //Copies cell image to host memory at index i
    void setCellImage(const cv::Mat &t_cellImage, const size_t i);
    //Returns pointer to cell image on GPU
    uchar *getCellImage(const size_t i);

    //Copies library images to GPU
    void setLibraryImages(const std::vector<cv::Mat> &t_libraryImages);
    //Returns pointer to library images on GPU
    uchar *getLibraryImages();

    //Copies mask image to GPU
    void setMaskImage(const cv::Mat &t_maskImage,
                      const bool t_flippedHorizontal, const bool t_flippedVertical);
    //Set if and how cells should flip on alternate rows/columns
    void setFlipStates(const bool t_colFlipHorizontal, const bool t_colFlipVertical,
                       const bool t_rowFlipHorizontal, const bool t_rowFlipVertical);
    //Returns pointer to mask image on GPU
    uchar *getMaskImage(const int t_gridX, const int t_gridY);

    //Copies target area to host memory at index i
    void setTargetArea(const size_t (&t_targetArea)[4], const size_t i);
    //Returns pointer to target area on GPU
    size_t *getTargetArea(const size_t i);

    //Sets repeats to 0
    void clearRepeats();
    //Returns pointer to repeats on GPU
    size_t *getRepeats();

    //Returns pointer to variants on GPU
    double *getVariants(const size_t i);

    //Sets best fit to number of library images
    void resetBestFit();
    //Returns pointer to best fit on GPU
    size_t *getBestFit(const size_t i);

    //Sets lowest variant to max double
    void resetLowestVariant();
    //Returns pointer to lowest variant on GPU
    double *getLowestVariant(const size_t i);

    //Returns pointer to reduction memory on GPU
    double *getReductionMemory(const size_t i);

    //-------------------------------------
    const size_t imageSize; //Size of images (width == height)
    const size_t imageChannels; //Channels in main and library images

    const size_t pixelCount; //Number of pixels in images (width * height)
    const size_t fullSize; //Number of uchar in images (width * height * channels)

    const size_t noXCellImages, noYCellImages; //Number of cell images per row and column
    const size_t noCellImages; //Number of cell images
    const size_t noLibraryImages; //Number of library images

    const bool euclidean; //Which difference formula to use euclidean (true) or CIEDE2000 (false)

    const size_t repeatRange; //Range to look for repeats
    const size_t repeatAddition; //Value to add to variants based on number of repeats in range

private:
    bool dataIsAllocated; //Stores if any data has been allocated on GPU
    int currentBatchIndex; //Stores index of current batch of data loaded on device
    size_t noOfBatch; //Number of data batches
    size_t batchSize; //Number of cells in each batch

    size_t blockSize; //Number of threads per block

    uchar *HOST_cellImages; //Stores on host all cells from main image
    uchar *cellImage; //Stores cells from the main image
    uchar *libraryImages; //Stores all library images
    //Controls if and how cells should flip on alternate rows/columns
    bool m_colFlipHorizontal, m_colFlipVertical, m_rowFlipHorizontal, m_rowFlipVertical;
    uchar *maskImages; //Stores mask images

    size_t *HOST_targetAreas; //Stores on host bounds of all cells
    size_t *targetArea; //Stores bounds of cells (as custom cell shapes can exceed image bounds)

    double *variants; //Stores results of difference formula for each pixel
    size_t reductionMemorySize; //Size of reductionMemory
    double *reductionMemory; //Memory used to perform sum reduction on variants

    double *maxVariant; //Stores maximum variant value
    double *lowestVariant; //Stores the lowest variants seen
    size_t *HOST_bestFit; //Stores on host the lowest variant seen for each cell
    size_t *bestFit; //Stores the index of the lowest variant seen for each cell

    size_t *repeats; //Stores for each library image the value to add to variant for current cell
};

#endif // CUDAPHOTOMOSAICDATA_H
