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
#ifndef __CUDACC__ 
#define __CUDACC__
#endif

#include <math_constants.h>
#include <algorithm>
#include <cuda_runtime.h>
#include <cuda.h>
#include <device_launch_parameters.h>
#include <math.h>

#include "ColourDifference.cuh"

//Calculates the difference (using template function func) between two images (im_1, im_2) storing in variants
//Parts of the image can be ignored using im_mask (variant is set to 0)
//Image rows = size, cols = size
template<p_dfColourDifference func>
__global__
void imageDifference(float *im_1, float *im_2, unsigned char *im_mask,
                     const size_t size, double *variants)
{
    const size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t stride = blockDim.x * gridDim.x;
    for (size_t i = index; i < size * size; i += stride)
    {
        variants[i] = func(im_1 + i * 3, im_2 + i * 3) * (im_mask[i] != 0);
    }
}

//Calculates the difference (using template function func) between two images (im_1, im_2) storing in variants
//Parts of the image can be ignored using im_mask (variant is set to 0)
//Image rows = size, cols = size, channels = channels
//Edge case equivalent of imageDifference:
//target_area contains bounds (min row, max row, min col, max col), variant set to 0 for out of bound pixels
template<p_dfColourDifference func>
__global__
void imageDifferenceEdge(float *im_1, float *im_2, unsigned char *mask_im,
                         const size_t size, size_t *target_area, double *variants)
{
    const size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t stride = blockDim.x * gridDim.x;
    const size_t imageSize = size * size;
    for (size_t i = index; i < imageSize; i += stride)
    {
        const size_t row = i / size;
        const size_t col = i % size;
        bool valid = row >= target_area[0];
        valid = row < target_area[1] && valid;
        valid = col >= target_area[2] && valid;
        valid = col < target_area[3] && valid;
        valid = mask_im[i] != 0 && valid;
        variants[i] = func(im_1 + i*3, im_2 + i*3) * valid;
    }
}

//Explicit instantiation of templates
template __global__ void imageDifference<euclideanDifference>(float *im_1, float *im_2, unsigned char *im_mask, const size_t size, double *variants);
template __global__ void imageDifference<CIEDE2000Difference>(float *im_1, float *im_2, unsigned char *im_mask, const size_t size, double *variants);
template __global__ void imageDifferenceEdge<euclideanDifference>(float *im_1, float *im_2, unsigned char *im_mask, const size_t size, size_t *target_area, double *variants);
template __global__ void imageDifferenceEdge<CIEDE2000Difference>(float *im_1, float *im_2, unsigned char *im_mask, const size_t size, size_t *target_area, double *variants);

//Wrapper for imageDifference kernel with euclideanDifference kernel
//target_area is unused, it is just there so the function parameters match the edge case one
void euclideanDifferenceKernelWrapper(float *main_im, float *lib_im, unsigned char *mask_im,
                                      const size_t size, [[maybe_unused]]size_t *target_area, double *variants,
                                      const size_t blockSize, cudaStream_t stream)
{
    const size_t numBlocks = (size * size + blockSize - 1) / blockSize;
    imageDifference<euclideanDifference><<<static_cast<unsigned int>(numBlocks),
                               static_cast<unsigned int>(blockSize),
                               0, stream>>>(main_im, lib_im, mask_im, size, variants);
}

//Wrapper for imageDifference kernel with euclideanDifference kernel
void euclideanDifferenceEdgeKernelWrapper(float *main_im, float *lib_im, unsigned char *mask_im,
                                          const size_t size, size_t *target_area, double *variants,
                                          const size_t blockSize, cudaStream_t stream)
{
    const size_t numBlocks = (size * size + blockSize - 1) / blockSize;
    imageDifferenceEdge<euclideanDifference><<<static_cast<unsigned int>(numBlocks),
                                  static_cast<unsigned int>(blockSize),
                                  0, stream>>>(main_im, lib_im, mask_im, size, target_area, variants);
}

//Wrapper for imageDifference kernel with CIEDE2000Difference kernel
//target_area is unused, it is just there so the function parameters match the edge case one
void CIEDE2000DifferenceKernelWrapper(float *main_im, float *lib_im, unsigned char *mask_im,
                                      const size_t size, [[maybe_unused]]size_t *target_area, double *variants,
                                      const size_t blockSize, cudaStream_t stream)
{
    const size_t numBlocks = (size * size + blockSize - 1) / blockSize;
    imageDifference<CIEDE2000Difference><<<static_cast<unsigned int>(numBlocks),
                               static_cast<unsigned int>(blockSize),
                               0, stream>>>(main_im, lib_im, mask_im, size, variants);
}

//Wrapper for imageDifference kernel with CIEDE2000Difference kernel
void CIEDE2000DifferenceEdgeKernelWrapper(float *main_im, float *lib_im, unsigned char *mask_im,
                                          const size_t size, size_t *target_area, double *variants,
                                          const size_t blockSize, cudaStream_t stream)
{
    const size_t numBlocks = (size * size + blockSize - 1) / blockSize;
    imageDifferenceEdge<CIEDE2000Difference><<<static_cast<unsigned int>(numBlocks),
                                   static_cast<unsigned int>(blockSize),
                                   0, stream>>>(main_im, lib_im, mask_im, size, target_area, variants);
}

//Calculates repeats in range and adds to variants
__global__
void calculateRepeats(double **variants, size_t noMainImage,
                     size_t *bestFit, const size_t bestFitMax, const size_t gridWidth,
                     const int leftRange, const int rightRange, const int upRange,
                     const size_t repeatAddition)
{
    for (int y = -upRange; y < 0; ++y)
    {
        for (int x = -leftRange; x <= rightRange; ++x)
        {
            if (bestFit[y * gridWidth + x] < bestFitMax)
            {
                for (size_t mainI = 0; mainI < noMainImage; ++mainI)
                    variants[mainI][bestFit[y * gridWidth + x]] += repeatAddition;
            }
        }
    }
    for (int x = -leftRange; x < 0; ++x)
    {
        if (bestFit[x] < bestFitMax)
        {
            for (size_t mainI = 0; mainI < noMainImage; ++mainI)
                variants[mainI][bestFit[x]] += repeatAddition;
        }
    }
}

//Wrapper for calculate repeats kernel
//Calculates repeats in range and adds to variants
void calculateRepeatsKernelWrapper(double **variants, size_t noMainImage,
                                   size_t *bestFit, const size_t bestFitMax,
                                   const size_t gridWidth, const int x, const int y,
                                   const int padGrid,
                                   const size_t repeatRange, const size_t repeatAddition)
{
    const size_t paddedX = x + padGrid;
    const size_t paddedY = y + padGrid;

    const int leftRange = static_cast<int>(std::min(repeatRange, paddedX));
    const int rightRange = static_cast<int>(std::min(repeatRange, gridWidth - paddedX - 1));
    const int upRange = static_cast<int>(std::min(repeatRange, paddedY));
    calculateRepeats<<<1, 1>>>(variants, noMainImage,
                               bestFit + paddedY * gridWidth + paddedX, bestFitMax, gridWidth,
                               leftRange, rightRange, upRange,
                               repeatAddition);
}

//Finds lowest value in variants
__global__
void findLowestKernel(double *lowestVariant, size_t *bestFit, double **variants, const size_t noLibIm, const size_t noMainImage)
{
    for (size_t mainI = 0; mainI < noMainImage; ++mainI)
    {
        for (size_t libI = 0; libI < noLibIm; ++libI)
        {
            if (variants[mainI][libI] < *lowestVariant)
            {
                *lowestVariant = variants[mainI][libI];
                *bestFit = libI;
            }
        }
    }
}

//Wrapper for find lowest kernel
//Finds lowest value in variants
void findLowestKernelWrapper(double *lowestVariant, size_t *bestFit, double **variants, const size_t noLibIm, const size_t noMainImage)
{
    findLowestKernel<<<1, 1>>>(lowestVariant, bestFit, variants, noLibIm, noMainImage);
}

//Flattens size elements that have spacing elements between them
//Grid x size should be set to size of the first dim of data
//Block x size should be set to at most spacing
__global__
void flattenKernel(double **data, const size_t size, const size_t spacing)
{
    for (size_t i = threadIdx.x; i < size; i += blockDim.x)
    {
        data[blockIdx.x][i] = data[blockIdx.x][i * spacing];
        __syncthreads();
    }
}

//Wrapper for flatten kernel
//Flattens size elements that have spacing elements between them
//Grid x size should be set to size of the first dim of data
//Block x size should be set to at most spacing
void flattenKernelWrapper(double **data, const size_t noMainImage, const size_t size, const size_t spacing, const size_t maxBlockSize)
{
    //Only need to flatten if spacing is > 1
    if (spacing > 1)
    {
        const size_t blockSize = std::min(maxBlockSize, spacing);
        flattenKernel<<<static_cast<unsigned int>(noMainImage),
            static_cast<unsigned int>(blockSize)>>>(data, size, spacing);
    }
}