#include "mainwindow.h"

#include <QApplication>

#ifdef CUDA
#include <iostream>
#include <cuda_runtime.h>

int checkCUDAState()
{
    int deviceCount, device;
    int gpuDeviceCount = 0;
    cudaDeviceProp properties;
    cudaError_t cudaResultCode = cudaGetDeviceCount(&deviceCount);
    if (cudaResultCode != cudaSuccess)
        deviceCount = 0;

    //Check devices are not emulation only (9999)
    for (device = 0; device < deviceCount; ++device) {
        cudaGetDeviceProperties(&properties, device);
        if (properties.major != 9999)
            ++gpuDeviceCount;
    }
    std::cout << gpuDeviceCount << " GPU CUDA device(s) found\n";

    if (gpuDeviceCount > 0)
        return 0; //Success
    else
        return 1; //Failure
}
#endif

int main(int argc, char *argv[])
{
#ifdef CUDA
    //Check valid CUDA GPU's available
    if (checkCUDAState())
        return -1;

    //Initialise CUDA
    int *deviceInit;
    cudaMalloc(&deviceInit, 0 * sizeof(int));
#endif

    QApplication a(argc, argv);
    MainWindow w;
    w.show();
    return a.exec();
}
