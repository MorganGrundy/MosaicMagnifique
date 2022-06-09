#pragma once

#include <gtest/gtest.h>
#include <gmock/gmock-matchers.h>
#include <opencv2/cudawarping.hpp>

#include "testutility.h"
#include "..\src\ImageLibrary\CUDA\CUDAImageLibrary.h"

//Adds images to a CUDAImageLibrary, tests that the Mat and GpuMat match
TEST(CUDAImageLibrary, Compare_AddImage)
{
	srand(static_cast<unsigned int>(time(NULL)));

	const size_t imageSize = 128;
	const size_t noLibImages = 500;

	//Create library
	CUDAImageLibrary lib(imageSize);
	for (size_t i = 0; i < noLibImages; ++i)
		lib.addImage(TestUtility::createRandomImage(imageSize, imageSize));

	ASSERT_TRUE(TestUtility::compareImages(lib.getImages(), lib.getCUDAImages()));
}

//Sets image size on a CUDAImageLibrary, tests that the Mat and GpuMat match
//Disabled test as it will fail. We use cv::resize and cv::cuda::resize to resize the images
// but they give different results, so we can't really test this...
TEST(CUDAImageLibrary, DISABLED_Compare_SetImageSize)
{
	srand(static_cast<unsigned int>(time(NULL)));

	const size_t imageSize = 128;
	const size_t noLibImages = 500;

	//Create library
	CUDAImageLibrary lib(imageSize);
	for (size_t i = 0; i < noLibImages; ++i)
		lib.addImage(TestUtility::createRandomImage(imageSize, imageSize));

	//Set image size
	for (const size_t &size : { 512, 256, 128, 64, 32, 16 })
	{
		lib.setImageSize(size);
		ASSERT_TRUE(TestUtility::compareImages(lib.getImages(), lib.getCUDAImages()) << "Image size " << size);
	}
}

//Removes images from a CUDAImageLibrary, tests that the Mat and GpuMat match
TEST(CUDAImageLibrary, Compare_RemoveAtIndex)
{
	srand(static_cast<unsigned int>(time(NULL)));

	const size_t imageSize = 128;
	const size_t noLibImages = 500;
	const size_t noImagesToRemove = 100;

	//Create library
	CUDAImageLibrary lib(imageSize);
	for (size_t i = 0; i < noLibImages; ++i)
		lib.addImage(TestUtility::createRandomImage(imageSize, imageSize));

	for (size_t i = 0; i < noImagesToRemove; ++i)
	{
		//Remove a random image
		lib.removeAtIndex(TestUtility::randNum<size_t>(0, lib.getImages().size()));
		//Compare images
		ASSERT_TRUE(TestUtility::compareImages(lib.getImages(), lib.getCUDAImages()));
	}
}