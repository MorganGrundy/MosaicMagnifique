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

//Generates a Photomosaic on GPU using CUDA
class CUDAPhotomosaicGenerator : public PhotomosaicGeneratorBase
{
public:
    CUDAPhotomosaicGenerator();

    //Sets CUDA library images
    void setCUDALibrary(const std::vector<cv::cuda::GpuMat> &t_lib);

    //Generate best fits for Photomosaic cells
    //Returns true if successful
    bool generateBestFits() override;

private:
    std::vector<cv::cuda::GpuMat> m_cudaLib;

    //Copies mat to device pointer
    template <typename T>
    void copyMatToDevice(const cv::Mat &t_mat, T *t_device) const;

    //Copies mat to device pointer
    template <typename T>
    void copyMatToDevice(const cv::cuda::GpuMat &t_mat, T *t_device) const;

    //Performs preprocessing steps on library images: resize, convert colour space
    std::vector<cv::cuda::GpuMat> preprocessCUDALibraryImages();
};

#endif // CUDA
#endif // CUDAPHOTOMOSAICGENERATORBASE_H
