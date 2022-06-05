#pragma once

#include "..\ImageLibrary\ImageLibrary.h"

#include <opencv2/cudaimgproc.hpp>

class CUDAImageLibrary : public ImageLibrary
{
public:

protected:
    //Internals of addImage, adds image at the given index in relevant containers
    //In addition to the base class functionality it uploads the image to the gpu
    void addImageInternal(const size_t index, const cv::Mat &t_im) override;

private:
    std::vector<cv::cuda::GpuMat> m_gpuImages;
    std::vector<cv::cuda::GpuMat> m_gpuResizedImages;
};

