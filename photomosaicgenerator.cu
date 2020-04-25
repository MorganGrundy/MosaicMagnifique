#include <cuda.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>
#include <math_constants.h>
#include <limits>

#define uchar unsigned char

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
            continue;

        const size_t col = grayscaleIndex % size;
        if (col < target_area[2] || col >= target_area[3])
            continue;

        if (mask_im[grayscaleIndex] != 0)
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
            continue;

        const size_t col = grayscaleIndex % size;
        if (col < target_area[2] || col >= target_area[3])
            continue;

        if (mask_im[grayscaleIndex] != 0)
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

//Performs sum reduction in a single warp
template <size_t blockSize>
__device__
void warpReduce(volatile double *sdata, const size_t tid)
{
    if (blockSize >= 64)
        sdata[tid] += sdata[tid + 32];
    if (blockSize >= 32)
        sdata[tid] += sdata[tid + 16];
    if (blockSize >= 16)
        sdata[tid] += sdata[tid + 8];
    if (blockSize >= 8)
        sdata[tid] += sdata[tid + 4];
    if (blockSize >= 4)
        sdata[tid] += sdata[tid + 2];
    if (blockSize >= 2)
        sdata[tid] += sdata[tid + 1];

}

//Performs sum reduction
template <size_t blockSize>
__global__
void reduce(double *g_idata, double *g_odata, const size_t N, const size_t noLibIm)
{
    extern __shared__ double sdata[];

    for (size_t libI = 0; libI < noLibIm; ++libI)
    {
        size_t offset = libI * N;
        //Each thread loads atleast one element from global to shared memory
        size_t tid = threadIdx.x;
        size_t i = blockIdx.x * blockSize * 2 + threadIdx.x;
        size_t gridSize = blockSize * 2 * gridDim.x;
        sdata[tid] = 0;

        while (i < N)
        {
            sdata[tid] += (i + blockSize < N) ?
                        g_idata[i + offset] + g_idata[i + blockSize + offset] : g_idata[i + offset];
            i += gridSize;
        }
        __syncthreads();

        //Do reduction in shared memory
        if (blockSize >= 2048)
        {
            if (tid < 1024)
                sdata[tid] += sdata[tid + 1024];
            __syncthreads();
        }
        if (blockSize >= 1024)
        {
            if (tid < 512)
                sdata[tid] += sdata[tid + 512];
            __syncthreads();
        }
        if (blockSize >= 512)
        {
            if (tid < 256)
                sdata[tid] += sdata[tid + 256];
            __syncthreads();
        }
        if (blockSize >= 256)
        {
            if (tid < 128)
                sdata[tid] += sdata[tid + 128];
            __syncthreads();
        }
        if (blockSize >= 128)
        {
            if (tid < 64)
                sdata[tid] += sdata[tid + 64];
            __syncthreads();
        }

        if (tid < 32)
            warpReduce<blockSize>(sdata, tid);

        //Write result for this block to global memory
        if (tid == 0)
            g_odata[blockIdx.x + libI * gridDim.x] = sdata[0];
    }
}

void reduceData(double *data, double *output, const size_t N, const size_t maxBlockSize, const size_t noLibIm)
{
    size_t reduceDataSize = N;

    //Number of blocks needed assuming max block size
    size_t numBlocks = ((reduceDataSize + maxBlockSize - 1) / maxBlockSize + 1) / 2;

    //Minimum number of threads per block
    size_t reduceBlockSize;

    //Stores number of threads to use per block (power of 2)
    size_t threads = maxBlockSize;

    do
    {
        //Calculate new number of blocks and threads
        numBlocks = ((reduceDataSize + maxBlockSize - 1) / maxBlockSize + 1) / 2;
        reduceBlockSize = (reduceDataSize + numBlocks - 1) / numBlocks;
        while (threads > reduceBlockSize * 2)
            threads >>= 1;

        //Reduce
        switch (threads)
        {
        case 2048:
            reduce<2048><<<static_cast<unsigned int>(numBlocks),
                    static_cast<unsigned int>(threads),
                    static_cast<unsigned int>(threads * sizeof(double))
                    >>>(data, output, reduceDataSize, noLibIm);
            break;
        case 1024:
            reduce<1024><<<static_cast<unsigned int>(numBlocks),
                    static_cast<unsigned int>(threads),
                    static_cast<unsigned int>(threads * sizeof(double))
                    >>>(data, output, reduceDataSize, noLibIm);
            break;
        case 512:
            reduce<512><<<static_cast<unsigned int>(numBlocks),
                    static_cast<unsigned int>(threads),
                    static_cast<unsigned int>(threads * sizeof(double))
                    >>>(data, output, reduceDataSize, noLibIm);
            break;
        case 256:
            reduce<256><<<static_cast<unsigned int>(numBlocks),
                    static_cast<unsigned int>(threads),
                    static_cast<unsigned int>(threads * sizeof(double))
                    >>>(data, output, reduceDataSize, noLibIm);
            break;
        case 128:
            reduce<128><<<static_cast<unsigned int>(numBlocks),
                    static_cast<unsigned int>(threads),
                    static_cast<unsigned int>(threads * sizeof(double))
                    >>>(data, output, reduceDataSize, noLibIm);
            break;
        case 64:
            reduce<64><<<static_cast<unsigned int>(numBlocks),
                    static_cast<unsigned int>(threads),
                    static_cast<unsigned int>(threads * sizeof(double))
                    >>>(data, output, reduceDataSize, noLibIm);
            break;
        case 32:
            reduce<32><<<static_cast<unsigned int>(numBlocks),
                    static_cast<unsigned int>(threads),
                    static_cast<unsigned int>(threads * sizeof(double))
                    >>>(data, output, reduceDataSize, noLibIm);
            break;
        case 16:
            reduce<16><<<static_cast<unsigned int>(numBlocks),
                    static_cast<unsigned int>(threads),
                    static_cast<unsigned int>(threads * sizeof(double))
                    >>>(data, output, reduceDataSize, noLibIm);
            break;
        case 8:
            reduce<8><<<static_cast<unsigned int>(numBlocks),
                    static_cast<unsigned int>(threads),
                    static_cast<unsigned int>(threads * sizeof(double))
                    >>>(data, output, reduceDataSize, noLibIm);
            break;
        case 4:
            reduce<4><<<static_cast<unsigned int>(numBlocks),
                    static_cast<unsigned int>(threads),
                    static_cast<unsigned int>(threads * sizeof(double))
                    >>>(data, output, reduceDataSize, noLibIm);
            break;
        case 2:
            reduce<2><<<static_cast<unsigned int>(numBlocks),
                    static_cast<unsigned int>(threads),
                    static_cast<unsigned int>(threads * sizeof(double))
                    >>>(data, output, reduceDataSize, noLibIm);
            break;
        case 1:
            reduce<1><<<static_cast<unsigned int>(numBlocks),
                    static_cast<unsigned int>(threads),
                    static_cast<unsigned int>(threads * sizeof(double))
                    >>>(data, output, reduceDataSize, noLibIm);
            break;
        }

        //Copy results back to data
        cudaMemcpy(data, output, numBlocks * noLibIm * sizeof(double), cudaMemcpyDeviceToDevice);

        //New data length is equal to number of blocks
        reduceDataSize = numBlocks;
    }
    while (numBlocks > 1); //Keep reducing until only 1 block was used
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

extern "C"
size_t differenceGPU(uchar *main_im, uchar *t_lib_im, size_t noLibIm, uchar *t_mask_im,
                     size_t im_size[2], size_t *target_area,
                     size_t *t_repeats, bool euclidean, double *variants)
{
    size_t pixelCount = im_size[0] * im_size[0];

    //Initialise lowestVariant with largest possible value
    double *lowestVariant;
    cudaMalloc((void **)&lowestVariant, sizeof(double));
    double doubleMax = std::numeric_limits<double>::max();
    cudaMemcpy(lowestVariant, &doubleMax, sizeof(double), cudaMemcpyHostToDevice);

    size_t *bestFit;
    cudaMalloc((void **)&bestFit, sizeof(size_t));
    cudaMemcpy(bestFit, &noLibIm, sizeof(size_t), cudaMemcpyHostToDevice);

    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);
    const size_t blockSize = deviceProp.maxThreadsPerBlock;

    double *reduceTmpMemory;
    const size_t reduceMemSize = (pixelCount + blockSize - 1) / blockSize;
    cudaMalloc((void **)&reduceTmpMemory, reduceMemSize * noLibIm * sizeof(double));

    size_t numBlocks = (pixelCount * noLibIm + blockSize - 1) / blockSize;

    if (euclidean)
        euclideanDifferenceKernel<<<static_cast<unsigned int>(numBlocks),
                static_cast<unsigned int>(blockSize)>>>(main_im, t_lib_im, noLibIm, t_mask_im,
                                                        im_size[0], im_size[1], target_area,
                                                        variants);
    else
        CIEDE2000DifferenceKernel<<<static_cast<unsigned int>(numBlocks),
                static_cast<unsigned int>(blockSize)>>>(main_im, t_lib_im, noLibIm, t_mask_im,
                                                        im_size[0], im_size[1], target_area,
                                                        variants);

    //Perform sum reduction on all image variants
    reduceData(variants, reduceTmpMemory, pixelCount, blockSize, noLibIm);

    //Adds repeat value to first difference value
    numBlocks = (noLibIm + blockSize - 1) / blockSize;
    addRepeatsKernel<<<static_cast<unsigned int>(numBlocks),
            static_cast<unsigned int>(blockSize)>>>(variants, t_repeats, noLibIm);

    //Find lowest variant
    findLowestKernel<<<1,1>>>(lowestVariant, bestFit, variants, noLibIm);

    cudaDeviceSynchronize();

    //Copy result from GPU to CPU
    size_t result;
    cudaMemcpy(&result, bestFit, sizeof(size_t), cudaMemcpyDeviceToHost);

    //Free memory on GPU
    cudaFree(lowestVariant);
    cudaFree(bestFit);

    return result;
}
