#pragma once

#include <cuda_runtime.h>

void euclidNoPowWrapper(float *colour1, float *colour2, double *result, int t_size);

void euclidPowWrapper(float *colour1, float *colour2, double *result, int t_size);

void CIEDE2000NewWrapper(float *colour1, float *colour2, double *result, int t_size);

void CIEDE2000OldWrapper(float *colour1, float *colour2, double *result, int t_size);

void CIEDE2000VariantDeltaThetaWrapper(float *colour1, float *colour2, double *result, int t_size);

void CIEDE2000VariantS_LWrapper(float *colour1, float *colour2, double *result, int t_size);