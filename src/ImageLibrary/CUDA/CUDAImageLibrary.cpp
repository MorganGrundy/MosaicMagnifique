#include "CUDAImageLibrary.h"

//Set image size
void CUDAImageLibrary::setImageSize(const size_t t_size)
{
	if (t_size == m_imageSize)
		return;

	ImageLibrary::setImageSize(t_size);

	//Resize library images to new size
	ImageUtility::batchResizeMat(m_gpuImages, m_gpuResizedImages, static_cast<int>(t_size),
		static_cast<int>(t_size), ImageUtility::ResizeType::EXACT);
}

//Returns const reference to GPU library images
const std::vector<cv::cuda::GpuMat> &CUDAImageLibrary::getGPUImages() const
{
	return m_gpuResizedImages;
}

//Removes the image at given index
void CUDAImageLibrary::removeAtIndex(const size_t t_index)
{
	ImageLibrary::removeAtIndex(t_index);

	m_gpuImages.erase(m_gpuImages.begin() + t_index);
	m_gpuResizedImages.erase(m_gpuResizedImages.begin() + t_index);
}

//Clear image library
void CUDAImageLibrary::clear()
{
	ImageLibrary::clear();

	m_gpuImages.clear();
	m_gpuResizedImages.clear();
}

//Internals of addImage, adds image at the given index in relevant containers
//In addition to the base class functionality it uploads the image to the gpu
void CUDAImageLibrary::addImageInternal(const size_t index, const cv::Mat &t_im)
{
	ImageLibrary::addImageInternal(index, t_im);

	auto it = m_gpuImages.insert(m_gpuImages.begin() + index, cv::cuda::GpuMat(t_im));
	m_gpuResizedImages.insert(m_gpuResizedImages.begin() + index, it->clone());
}
