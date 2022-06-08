#pragma once

#include "..\ImageLibrary.h"

#include <opencv2/cudaimgproc.hpp>

class CUDAImageLibrary : public ImageLibrary
{
public:
    CUDAImageLibrary(const size_t t_imageSize);

    //Set image size
    void setImageSize(const size_t t_size) override;

    //Returns const reference to CUDA library images
    const std::vector<cv::cuda::GpuMat> &getCUDAImages() const;

    //Removes the image at given index
    void removeAtIndex(const size_t t_index) override;
    //Clear image library
    void clear() override;

protected:
    //Internals of addImage, adds image at the given index in relevant containers
    //In addition to the base class functionality it uploads the image to the gpu
    void addImageInternal(const size_t index, const cv::Mat &t_im) override;

private:
    std::vector<cv::cuda::GpuMat> m_gpuImages;
    std::vector<cv::cuda::GpuMat> m_gpuResizedImages;
};

