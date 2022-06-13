#pragma once

#include <cuda_runtime.h>
//Wrapper for imageEuclideanDifference kernel
//target_area is unused, it is just there so the function parameters match the edge case one
void euclideanDifferenceKernelWrapper(float *main_im, float *lib_im, unsigned char *mask_im,
									  size_t size, size_t *target_area, double *variants,
                                      size_t blockSize, cudaStream_t stream);

//Wrapper for imageEuclideanDifferenceEdge kernel
void euclideanDifferenceEdgeKernelWrapper(float *main_im, float *lib_im, unsigned char *mask_im,
                                      size_t size, size_t *target_area, double *variants,
                                      size_t blockSize, cudaStream_t stream);

//Wrapper for imageCIEDE2000Difference kernel
//target_area is unused, it is just there so the function parameters match the edge case one
void CIEDE2000DifferenceKernelWrapper(float *main_im, float *lib_im, unsigned char *mask_im,
									  size_t size, size_t *target_area, double *variants,
                                      size_t blockSize, cudaStream_t stream);

//Wrapper for imageCIEDE2000DifferenceEdge kernel
void CIEDE2000DifferenceEdgeKernelWrapper(float *main_im, float *lib_im, unsigned char *mask_im,
                                          size_t size, size_t *target_area, double *variants,
                                          size_t blockSize, cudaStream_t stream);

//Wrapper for calculate repeats kernel
void calculateRepeatsKernelWrapper(double *variants,
                                   size_t *bestFit, const size_t bestFitMax,
                                   const size_t gridWidth, const int x, const int y,
                                   const int padGrid,
                                   const size_t repeatRange, const size_t repeatAddition);

//Wrapper for find lowest kernel
void findLowestKernelWrapper(double *lowestVariant, size_t *bestFit, double *variants,
                             size_t noLibIm);