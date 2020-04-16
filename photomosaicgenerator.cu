#include <cuda.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>
#include <math_constants.h>
#include <limits>

#define uchar unsigned char

//Kernel that calculates the euclidean difference between two images for each corresponding pixel
__global__
void euclideanDifferenceKernel(uchar *im_1, uchar *im_2, uchar *mask_im, int size, int channels,
                               double *variants)
{
    int index = (blockIdx.x * blockDim.x + threadIdx.x) * channels;
    int stride = blockDim.x * gridDim.x * channels;
    for (int i = index; i < size * channels; i += stride)
    {
        if (mask_im[i / channels] != 0)
            variants[i / channels] = sqrt(pow((double) im_1[i] - im_2[i], 2.0) +
                                          pow((double) im_1[i + 1] - im_2[i + 1], 2.0) +
                                          pow((double) im_1[i + 2] - im_2[i + 2], 2.0));
    }
}

//Converts degrees to radians
__device__
constexpr double degToRadKernel(const double deg)
{
    return (deg * CUDART_PI) / 180;
}

//Kernel that calculates the CIEDE2000 difference between two images for each corresponding pixel
__global__
void CIEDE2000DifferenceKernel(uchar *im_1, uchar *im_2, uchar *mask_im, int size, int channels,
                               double *variants)
{
    int index = (blockIdx.x * blockDim.x + threadIdx.x) * channels;
    int stride = blockDim.x * gridDim.x * channels;
    for (int i = index; i < size * channels; i += stride)
    {
        if (mask_im[i / channels] != 0)
        {
            const double k_L = 1.0, k_C = 1.0, k_H = 1.0;
            constexpr double deg360InRad = degToRadKernel(360.0);
            constexpr double deg180InRad = degToRadKernel(180.0);
            const double pow25To7 = 6103515625.0; //pow(25, 7)

            const double C1 = sqrt((double) (im_1[i + 1] * im_1[i + 1]) +
                    (im_1[i + 2] * im_1[i + 2]));
            const double C2 = sqrt((double) (im_2[i + 1] * im_2[i + 1]) +
                    (im_2[i + 2] * im_2[i + 2]));
            const double barC = (C1 + C2) / 2.0;

            const double G = 0.5 * (1 - sqrt(pow(barC, 7) / (pow(barC, 7) + pow25To7)));

            const double a1Prime = (1.0 + G) * im_1[i + 1];
            const double a2Prime = (1.0 + G) * im_2[i + 1];

            const double CPrime1 = sqrt((a1Prime * a1Prime) +
                                        (im_1[i + 2] * im_1[i + 2]));
            const double CPrime2 = sqrt((a2Prime * a2Prime) +(im_2[i + 2] * im_2[i + 2]));

            double hPrime1;
            if (im_1[i + 2] == 0 && a1Prime == 0.0)
                hPrime1 = 0.0;
            else
            {
                hPrime1 = atan2((double) im_1[i + 2], a1Prime);
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

            const double deltaLPrime = im_2[i] - im_1[i];
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

            const double barLPrime = (im_1[i] + im_2[i]) / 2.0;
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

//Kernel that performs a step in the sum reduction
//N = size of data
//If N is even then will sum half the elements in data
//If N is odd then will sum half-1 the elements in data
__global__
void reduceStep(double *data, int N, int halfN)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = index; i + halfN < N; i += stride)
    {
        data[i] += data[i + halfN];
    }
}

__global__
void compareKernel(double *lowestVariant, size_t *bestFit, double *newVariant, size_t index)
{
    if (*newVariant < *lowestVariant)
    {
        *lowestVariant = *newVariant;
        *bestFit = index;
    }
}

__global__
void addSingleKernel(double *dst, size_t *src)
{
    *dst += *src;
}

extern "C"
size_t differenceGPU(uchar *main_im, uchar **t_lib_im, size_t noLibIm, uchar *t_mask_im,
                     int im_size[3], int target_area[4], size_t *t_repeats, int repeatAddition, bool euclidean)
{
    int pixelCount = im_size[0] * im_size[1];

    //Allocate list of variants (for each pixel)
    double *variants, *lowestVariant;
    cudaMalloc((void **)&variants, pixelCount * sizeof(double));

    //Initialise lowestVariant with largest possible value
    cudaMalloc((void **)&lowestVariant, sizeof(double));
    double doubleMax = std::numeric_limits<double>::max();
    cudaMemcpy(lowestVariant, &doubleMax, sizeof(double), cudaMemcpyHostToDevice);

    size_t *bestFit;
    cudaMalloc((void **)&bestFit, sizeof(size_t));
    cudaMemset(bestFit, noLibIm, sizeof(size_t));

    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);
    const int maxThreadsPerBlock = deviceProp.maxThreadsPerBlock;
    for (size_t i = 0; i < noLibIm; ++i)
    {
        cudaMemset(variants, 0, pixelCount * sizeof(double));

        int blockSize = maxThreadsPerBlock;
        int numBlocks = (pixelCount + blockSize - 1) / blockSize;

        //Calculate euclidean differences
        if (euclidean)
            euclideanDifferenceKernel<<<numBlocks, blockSize>>>(main_im, t_lib_im[i], t_mask_im,
                                                                pixelCount, im_size[2], variants);
        else
            CIEDE2000DifferenceKernel<<<numBlocks, blockSize>>>(main_im, t_lib_im[i], t_mask_im,
                                                                pixelCount, im_size[2], variants);

        //Adds repeat value to first difference value
        addSingleKernel<<<1,1>>>(variants, t_repeats+i);
        //Calculate sum of differences
        int reduceSize = pixelCount;
        while (reduceSize > 1)
        {
            int halfSize = (reduceSize + 1) / 2;

            blockSize = (maxThreadsPerBlock <= halfSize) ? maxThreadsPerBlock : halfSize;
            numBlocks = (halfSize + blockSize - 1) / blockSize;

            reduceStep<<<numBlocks, blockSize>>>(variants, reduceSize, halfSize);
            reduceSize = halfSize;
        }

        compareKernel<<<1,1>>>(lowestVariant, bestFit, variants, i);
    }
    cudaDeviceSynchronize();

    //Copy result from GPU to CPU
    size_t result;
    cudaMemcpy(&result, bestFit, sizeof(size_t), cudaMemcpyDeviceToHost);

    //Free memory on GPU
    cudaFree(variants);
    cudaFree(lowestVariant);
    cudaFree(bestFit);

    return result;
}
