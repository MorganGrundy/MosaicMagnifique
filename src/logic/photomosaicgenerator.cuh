#pragma once

//Wrapper for euclidean difference kernel
void euclideanDifferenceKernelWrapper(float *im_1, float *im_2, size_t noLibIm,
                                      unsigned char *mask_im,
									  size_t size, size_t channels, size_t *target_area,
									  double *variants, size_t blockSize);

//Wrapper for euclidean difference kernel
void CIEDE2000DifferenceKernelWrapper(float *im_1, float *im_2, size_t noLibIm,
                                      unsigned char *mask_im,
									  size_t size, size_t channels, size_t *target_area,
									  double *variants, size_t blockSize);

//Wrapper for calculate repeats kernel
void calculateRepeatsKernelWrapper(double *variants,
                                   size_t *bestFit, const size_t bestFitMax,
                                   const size_t gridWidth, const int x, const int y,
                                   const int padGrid,
                                   const size_t repeatRange, const size_t repeatAddition);

//Wrapper for find lowest kernel
void findLowestKernelWrapper(double *lowestVariant, size_t *bestFit, double *variants,
                             size_t noLibIm);