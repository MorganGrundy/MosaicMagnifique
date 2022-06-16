#pragma once

#include <cuda_runtime.h>
#include <math.h>
#include <math_constants.h>
#include <device_launch_parameters.h>
#include <cuda.h>

__global__ void euclidNoPow(float *colour1, float *colour2, double *result, int t_size)
{
    for (int i = 0; i < t_size; ++i)
    {
        result[i] = sqrt(((double)colour1[i * 3] - colour2[i * 3]) * ((double)colour1[i * 3] - colour2[i * 3]) +
            ((double)colour1[i * 3 + 1] - colour2[i * 3 + 1]) * ((double)colour1[i * 3 + 1] - colour2[i * 3 + 1]) +
            ((double)colour1[i * 3 + 2] - colour2[i * 3 + 2]) * ((double)colour1[i * 3 + 2] - colour2[i * 3 + 2]));
    }
}

__global__ void euclidPow(float *colour1, float *colour2, double *result, int t_size)
{
    for (int i = 0; i < t_size; ++i)
    {
        result[i] = sqrt(pow((double)colour1[i * 3] - colour2[i * 3], 2.0) +
            pow((double)colour1[i * 3 + 1] - colour2[i * 3 + 1], 2.0) +
            pow((double)colour1[i * 3 + 2] - colour2[i * 3 + 2], 2.0));
    }
}

void euclidNoPowWrapper(float *colour1, float *colour2, double *result, int t_size)
{
    euclidNoPow<<<1, 1>>>(colour1, colour2, result, t_size);
}

void euclidPowWrapper(float *colour1, float *colour2, double *result, int t_size)
{
    euclidPow<<<1, 1>>>(colour1, colour2, result, t_size);
}