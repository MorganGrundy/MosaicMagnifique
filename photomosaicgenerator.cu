#include <cuda.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>
#include <math_constants.h>

#include "cudaphotomosaicdata.h"
#include "reduction.cuh"

//Calculates the euclidean difference between main image and library images
__global__
void euclideanDifferenceKernel(uchar *im_1, uchar *im_2, size_t noLibIm, uchar *mask_im,
                               size_t size, size_t channels, size_t *target_area, double *variants)
{
    const size_t index = (blockIdx.x * blockDim.x + threadIdx.x) * channels;
    const size_t stride = blockDim.x * gridDim.x * channels;
    for (size_t i = index; i < size * size * channels * noLibIm; i += stride)
    {
        const size_t im_1_index = i % (size * size * channels);
        const size_t grayscaleIndex = im_1_index / channels;

        const size_t row = grayscaleIndex / size;
        if (row < target_area[0] || row >= target_area[1])
        {
            variants[i / channels] = 0;
            continue;
        }

        const size_t col = grayscaleIndex % size;
        if (col < target_area[2] || col >= target_area[3])
        {
            variants[i / channels] = 0;
            continue;
        }

        if (mask_im[grayscaleIndex] == 0)
            variants[i / channels] = 0;
        else
            variants[i / channels] = sqrt(pow((double) im_1[im_1_index] - im_2[i], 2.0) +
                                          pow((double) im_1[im_1_index + 1] - im_2[i + 1], 2.0) +
                                          pow((double) im_1[im_1_index + 2] - im_2[i + 2], 2.0));
    }
}

//Converts degrees to radians
__device__
constexpr double degToRadKernel(const double deg)
{
    return (deg * CUDART_PI) / 180;
}

//Kernel that calculates the CIEDE2000 difference between main image and library images
__global__
void CIEDE2000DifferenceKernel(uchar *im_1, uchar *im_2, size_t noLibIm, uchar *mask_im,
                               size_t size, size_t channels, size_t *target_area, double *variants)
{
    const size_t index = (blockIdx.x * blockDim.x + threadIdx.x) * channels;
    const size_t stride = blockDim.x * gridDim.x * channels;
    for (size_t i = index; i < size * size * channels * noLibIm; i += stride)
    {
        const size_t im_1_index = i % (size * size * channels);
        const size_t grayscaleIndex = im_1_index / channels;

        const size_t row = grayscaleIndex / size;
        if (row < target_area[0] || row >= target_area[1])
        {
            variants[i / channels] = 0;
            continue;
        }

        const size_t col = grayscaleIndex % size;
        if (col < target_area[2] || col >= target_area[3])
        {
            variants[i / channels] = 0;
            continue;
        }

        if (mask_im[grayscaleIndex] == 0)
            variants[i / channels] = 0;
        else
        {
            const double k_L = 1.0, k_C = 1.0, k_H = 1.0;
            constexpr double deg360InRad = degToRadKernel(360.0);
            constexpr double deg180InRad = degToRadKernel(180.0);
            const double pow25To7 = 6103515625.0; //pow(25, 7)

            const double C1 = sqrt((double) (im_1[im_1_index + 1] * im_1[im_1_index + 1]) +
                    (im_1[im_1_index + 2] * im_1[im_1_index + 2]));
            const double C2 = sqrt((double) (im_2[i + 1] * im_2[i + 1]) +
                    (im_2[i + 2] * im_2[i + 2]));
            const double barC = (C1 + C2) / 2.0;

            const double G = 0.5 * (1 - sqrt(pow(barC, 7) / (pow(barC, 7) + pow25To7)));

            const double a1Prime = (1.0 + G) * im_1[im_1_index + 1];
            const double a2Prime = (1.0 + G) * im_2[i + 1];

            const double CPrime1 = sqrt((a1Prime * a1Prime) +
                                        (im_1[im_1_index + 2] * im_1[im_1_index + 2]));
            const double CPrime2 = sqrt((a2Prime * a2Prime) +(im_2[i + 2] * im_2[i + 2]));

            double hPrime1;
            if (im_1[im_1_index + 2] == 0 && a1Prime == 0.0)
                hPrime1 = 0.0;
            else
            {
                hPrime1 = atan2((double) im_1[im_1_index + 2], a1Prime);
                //This must be converted to a hue angle in degrees between 0 and 360 by
                //addition of 2 pi to negative hue angles.
                if (hPrime1 < 0)
                    hPrime1 += deg360InRad;
            }

            double hPrime2;
            if (im_2[i + 2] == 0 && a2Prime == 0.0)
                hPrime2 = 0.0;
            else
            {
                hPrime2 = atan2((double) im_2[i + 2], a2Prime);
                //This must be converted to a hue angle in degrees between 0 and 360 by
                //addition of 2pi to negative hue angles.
                if (hPrime2 < 0)
                    hPrime2 += deg360InRad;
            }

            const double deltaLPrime = im_2[i] - im_1[im_1_index];
            const double deltaCPrime = CPrime2 - CPrime1;

            double deltahPrime;
            const double CPrimeProduct = CPrime1 * CPrime2;
            if (CPrimeProduct == 0.0)
                deltahPrime = 0;
            else
            {
                //Avoid the fabs() call
                deltahPrime = hPrime2 - hPrime1;
                if (deltahPrime < -deg180InRad)
                    deltahPrime += deg360InRad;
                else if (deltahPrime > deg180InRad)
                    deltahPrime -= deg360InRad;
            }

            const double deltaHPrime = 2.0 * sqrt(CPrimeProduct) * sin(deltahPrime / 2.0);

            const double barLPrime = (im_1[im_1_index] + im_2[i]) / 2.0;
            const double barCPrime = (CPrime1 + CPrime2) / 2.0;

            double barhPrime;
            const double hPrimeSum = hPrime1 + hPrime2;
            if (CPrime1 * CPrime2 == 0.0)
                barhPrime = hPrimeSum;
            else
            {
                if (fabs(hPrime1 - hPrime2) <= deg180InRad)
                    barhPrime = hPrimeSum / 2.0;
                else
                {
                    if (hPrimeSum < deg360InRad)
                        barhPrime = (hPrimeSum + deg360InRad) / 2.0;
                    else
                        barhPrime = (hPrimeSum - deg360InRad) / 2.0;
                }
            }

            const double T = 1.0 - (0.17 * cos(barhPrime - degToRadKernel(30.0))) +
                    (0.24 * cos(2.0 * barhPrime)) +
                    (0.32 * cos((3.0 * barhPrime) + degToRadKernel(6.0))) -
                    (0.20 * cos((4.0 * barhPrime) - degToRadKernel(63.0)));

            const double deltaTheta = degToRadKernel(30.0) *
                    exp(-pow((barhPrime - degToRadKernel(275.0)) / degToRadKernel(25.0), 2.0));

            const double R_C = 2.0 * sqrt(pow(barCPrime, 7.0) /
                                          (pow(barCPrime, 7.0) + pow25To7));

            const double S_L = 1 + ((0.015 * pow(barLPrime - 50.0, 2.0)) /
                                    sqrt(20 + pow(barLPrime - 50.0, 2.0)));
            const double S_C = 1 + (0.045 * barCPrime);
            const double S_H = 1 + (0.015 * barCPrime * T);

            const double R_T = (-sin(2.0 * deltaTheta)) * R_C;


            variants[i / channels] = (double) sqrt(pow(deltaLPrime / (k_L * S_L), 2.0) +
                                                   pow(deltaCPrime / (k_C * S_C), 2.0) +
                                                   pow(deltaHPrime / (k_H * S_H), 2.0) +
                                                   (R_T * (deltaCPrime / (k_C * S_C)) *
                                                    (deltaHPrime / (k_H * S_H))));
        }
    }
}

//Calculates repeats in range
__global__
void calculateRepeats(bool *states, size_t *bestFit, size_t *repeats,
                      const int noXCell,
                      const int leftRange, const int rightRange,
                      const int upRange,
                      const size_t repeatAddition)
{
    for (int y = -upRange; y < 0; ++y)
    {
        for (int x = -leftRange; x <= rightRange; ++x)
        {
            if (states[y * noXCell + x])
                repeats[bestFit[y * noXCell + x]] += repeatAddition;
        }
    }
    for (int x = -leftRange; x < 0; ++x)
    {
        if (states[x])
            repeats[bestFit[x]] += repeatAddition;
    }
}

//Adds repeat values to variants
__global__
void addRepeatsKernel(double *variants, size_t *repeats, size_t noLibIm)
{
    const size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t stride = blockDim.x * gridDim.x;
    for (size_t i = index; i < noLibIm; i += stride)
        variants[i] += repeats[i];
}

//Finds lowest value in variants
__global__
void findLowestKernel(double *lowestVariant, size_t *bestFit, double *variants, size_t noLibIm)
{
    for (size_t i = 0; i < noLibIm; ++i)
    {
        if (variants[i] < *lowestVariant)
        {
            *lowestVariant = variants[i];
            *bestFit = i;
        }
    }
}

size_t differenceGPU(CUDAPhotomosaicData &photomosaicData)
{
    size_t numBlocks;
    const int batchSize = static_cast<int>(photomosaicData.getBatchSize());
    int batchIndex;

    //Create streams
    const static size_t maxStreams = 8;
    const size_t noOfStreams = std::min(maxStreams, photomosaicData.getBatchSize());
    size_t curStream = 0;
    cudaStream_t streams[maxStreams];
    for (size_t i = 0; i < noOfStreams; ++i)
        gpuErrchk(cudaStreamCreate(&streams[i]));

    //Loop over all batches
    while ((batchIndex = photomosaicData.copyNextBatchToDevice()) != -1)
    {
        //Loop over all data in batch
        for (size_t i = 0; i < batchSize
             && batchIndex * batchSize + i < photomosaicData.noCellImages; ++i)
        {
            //Skip if cell invalid
            if (!photomosaicData.getCellState(i))
                continue;

            const int x = (batchIndex * batchSize + static_cast<int>(i))
                    % static_cast<int>(photomosaicData.noXCellImages);
            const int y = (batchIndex * batchSize + static_cast<int>(i))
                    / static_cast<int>(photomosaicData.noXCellImages);

            //Calculate differences
            numBlocks = (photomosaicData.pixelCount * photomosaicData.noLibraryImages
                         + photomosaicData.getBlockSize() - 1) / photomosaicData.getBlockSize();
            if (photomosaicData.euclidean)
                euclideanDifferenceKernel<<<static_cast<unsigned int>(numBlocks),
                        static_cast<unsigned int>(photomosaicData.getBlockSize()),
                        0, streams[curStream]>>>(
                                photomosaicData.getCellImage(i),
                                photomosaicData.getLibraryImages(), photomosaicData.noLibraryImages,
                                photomosaicData.getMaskImage(x, y),
                                photomosaicData.imageSize, photomosaicData.imageChannels,
                                photomosaicData.getTargetArea(i),
                                photomosaicData.getVariants(i));
            else
                CIEDE2000DifferenceKernel<<<static_cast<unsigned int>(numBlocks),
                        static_cast<unsigned int>(photomosaicData.getBlockSize()),
                        0, streams[curStream]>>>(
                                photomosaicData.getCellImage(i),
                                photomosaicData.getLibraryImages(), photomosaicData.noLibraryImages,
                                photomosaicData.getMaskImage(x, y),
                                photomosaicData.imageSize, photomosaicData.imageChannels,
                                photomosaicData.getTargetArea(i),
                                photomosaicData.getVariants(i));

            //Move to next stream
            ++curStream;
            if (curStream == noOfStreams)
                curStream = 0;
        }
        //Perform sum reduction on all image variants
        reduceAddData(photomosaicData, streams, noOfStreams);

        //Loop over all data in batch
        for (size_t i = 0; i < batchSize
             && batchIndex * batchSize + i < photomosaicData.noCellImages; ++i)
        {
            //Skip if cell invalid
            if (!photomosaicData.getCellState(i))
                continue;

            //Calculate repeats
            photomosaicData.clearRepeats();
            const int x = (batchIndex * batchSize + static_cast<int>(i))
                    % static_cast<int>(photomosaicData.noXCellImages);
            const int y = (batchIndex * batchSize + static_cast<int>(i))
                    / static_cast<int>(photomosaicData.noXCellImages);

            const int leftRange = std::min(static_cast<int>(photomosaicData.repeatRange), x);
            const int rightRange = std::min(static_cast<int>(photomosaicData.repeatRange),
                                            static_cast<int>(photomosaicData.noXCellImages)
                                            - x - 1);
            const int upRange = std::min(static_cast<int>(photomosaicData.repeatRange), y);
            calculateRepeats<<<1, 1>>>(photomosaicData.getCellStateGPU(i),
                                       photomosaicData.getBestFit(i), photomosaicData.getRepeats(),
                                       static_cast<int>(photomosaicData.noXCellImages),
                                       leftRange, rightRange, upRange,
                                       photomosaicData.repeatAddition);

            //Adds repeat values to differences
            numBlocks = (photomosaicData.noLibraryImages
                         + photomosaicData.getBlockSize() - 1) / photomosaicData.getBlockSize();
            addRepeatsKernel<<<static_cast<unsigned int>(numBlocks),
                    static_cast<unsigned int>(photomosaicData.getBlockSize())>>>(
                            photomosaicData.getVariants(i),
                            photomosaicData.getRepeats(),
                            photomosaicData.noLibraryImages);

            //Find lowest variant
            findLowestKernel<<<1, 1>>>(photomosaicData.getLowestVariant(i),
                                       photomosaicData.getBestFit(i),
                                       photomosaicData.getVariants(i),
                                       photomosaicData.noLibraryImages);
        }
    }
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());

    //Destroy streams
    for (size_t i = 0; i < noOfStreams; ++i)
        gpuErrchk(cudaStreamDestroy(streams[i]));

    return 0;
}
