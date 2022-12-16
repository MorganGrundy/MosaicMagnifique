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

#ifndef CUDAPHOTOMOSAICGENERATORBASE_H
#define CUDAPHOTOMOSAICGENERATORBASE_H
#ifdef CUDA

#include "..\photomosaicgeneratorbase.h"
#include "..\..\Other\TimingLogger.h"

//Generates a Photomosaic on GPU using CUDA
class CUDAPhotomosaicGenerator : public PhotomosaicGeneratorBase
{
public:
    CUDAPhotomosaicGenerator(const int device);

    //Generate best fits for Photomosaic cells
    //Returns true if successful
    bool generateBestFits() override;

private:
    int m_device;

    //Copies mat to device pointer
    template <typename T>
    cudaError copyMatToDevice(const cv::Mat &t_mat, T *t_device) const;

    //The max block size we can use on the device
    size_t m_blockSize;

    //CUDA streams
    static constexpr size_t streamCount = 16;

    //The max variant, the lowest variant device memory gets reset to this
    static constexpr double maxVariant = std::numeric_limits<double>::max();

    //Device memory
    // To clarify the naming m is member variable, h is host memory, d is device memory.
    // m_h_d is a host array of device memory. m_d_d is a device array of device memory.
    std::vector<float *> m_h_d_libIm;
    std::vector<uchar *> m_h_d_maskImages;
    std::vector<double *> m_h_d_variants;
    double **m_d_d_variants;
    std::vector<float *> m_h_d_cellImages;
    size_t *m_d_targetArea;
    std::vector<double *> m_h_d_reductionMems;
    size_t *m_d_bestFit;
    double *m_d_lowestVariant;

    //Attempts to allocate all of the device memory
    //Returns whether it was all allocated
    bool allocateDeviceMemory(TimingLogger &timingLogger,
        const std::vector<cv::Mat> &mainImages, std::vector<cv::Mat> &libImages);
    //Attempts to free all of the device memory
    void freeDeviceMemory(TimingLogger &timingLogger);
};

#endif // CUDA
#endif // CUDAPHOTOMOSAICGENERATORBASE_H
