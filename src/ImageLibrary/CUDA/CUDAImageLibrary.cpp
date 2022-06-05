#include "CUDAImageLibrary.h"

//Internals of addImage, adds image at the given index in relevant containers
//In addition to the base class functionality it uploads the image to the gpu
void CUDAImageLibrary::addImageInternal(const size_t index, const cv::Mat &t_im)
{
	ImageLibrary::addImageInternal(index, t_im);

	auto it = m_gpuImages.insert(m_gpuImages.begin() + index, cv::cuda::GpuMat(t_im));
	m_gpuResizedImages.insert(m_gpuResizedImages.begin() + index, it->clone());
}
