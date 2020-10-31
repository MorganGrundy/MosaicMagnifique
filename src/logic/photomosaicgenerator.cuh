#ifndef CUDA_PHOTOMOSAICGENERATOR_H
#define CUDA_PHOTOMOSAICGENERATOR_H

//Wrapper for euclidean difference kernel
void euclideanDifferenceKernelWrapper(float *im_1, float *im_2, size_t noLibIm, uchar *mask_im,
									  size_t size, size_t channels, size_t *target_area,
									  double *variants, size_t blockSize);

//Wrapper for euclidean difference kernel
void CIEDE2000DifferenceKernelWrapper(float *im_1, float *im_2, size_t noLibIm, uchar *mask_im,
									  size_t size, size_t channels, size_t *target_area,
									  double *variants, size_t blockSize);

//Wrapper for calculate repeats kernel
void calculateRepeatsKernelWrapper(bool *states, size_t *bestFit, size_t *repeats,
const int noXCell, const int leftRange, const int rightRange, const int upRange,
const size_t repeatAddition);

//Wrapper for add repeats kernel
void addRepeatsKernelWrapper(double *variants, size_t *repeats, size_t noLibIm, size_t blockSize);

//Wrapper for find lowest kernel
void findLowestKernelWrapper(double *lowestVariant, size_t *bestFit, double *variants,
size_t noLibIm);

#endif // CUDA_PHOTOMOSAICGENERATOR_H
