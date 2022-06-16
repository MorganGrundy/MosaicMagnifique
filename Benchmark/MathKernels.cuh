#pragma once

#include <cuda_runtime.h>

void euclidNoPowWrapper(float *colour1, float *colour2, double *result, int t_size);

void euclidPowWrapper(float *colour1, float *colour2, double *result, int t_size);