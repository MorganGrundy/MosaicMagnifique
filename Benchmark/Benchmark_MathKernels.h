#pragma once

#include <chrono>
#include <iostream>
#include <chrono>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

#include "MathKernels.cuh"
#include "..\test\TestUtility.h"

template <typename inT, typename outT>
void MathKernelsCompare(int size, void (*newKernelPtr)(inT*, inT*, outT*, int), void (*oldKernelPtr)(inT *, inT *, outT *, int))
{
	auto h_colour1 = TestUtility::createRandom<inT>(size * 3, { {0, 255} });
	auto h_colour2 = TestUtility::createRandom<inT>(size * 3, { {0, 255} });
	inT *d_colour1, *d_colour2;
	cudaMalloc((void **)&d_colour1, size * 3 * sizeof(inT));
	cudaMalloc((void **)&d_colour2, size * 3 * sizeof(inT));
	cudaMemcpy(d_colour1, h_colour1.data(), size * 3 * sizeof(inT), cudaMemcpyHostToDevice);
	cudaMemcpy(d_colour2, h_colour2.data(), size * 3 * sizeof(inT), cudaMemcpyHostToDevice);

	std::vector<outT> h_resultNew(size, 0), h_resultOld(size, 0);
	outT *d_resultNew, *d_resultOld;
	cudaMalloc((void **)&d_resultNew, size * sizeof(outT));
	cudaMalloc((void **)&d_resultOld, size * sizeof(outT));
	cudaMemcpy(d_resultNew, h_resultNew.data(), size * sizeof(outT), cudaMemcpyHostToDevice);
	cudaMemcpy(d_resultOld, h_resultOld.data(), size * sizeof(outT), cudaMemcpyHostToDevice);

	cudaDeviceSynchronize();

	auto newStart = std::chrono::high_resolution_clock::now();
	newKernelPtr(d_colour1, d_colour2, d_resultNew, size);
	cudaDeviceSynchronize();
	auto newDuration = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - newStart);

	auto oldStart = std::chrono::high_resolution_clock::now();
	oldKernelPtr(d_colour1, d_colour2, d_resultOld, size);
	cudaDeviceSynchronize();
	auto oldDuration = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - oldStart);

	cudaMemcpy(h_resultNew.data(), d_resultNew, size * sizeof(outT), cudaMemcpyDeviceToHost);
	cudaMemcpy(h_resultOld.data(), d_resultOld, size * sizeof(outT), cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();

	outT totalDiff = 0.0;
	size_t significantDiffCount = 0;
	for (int i = 0; i < size; ++i)
	{
		totalDiff += std::abs(h_resultNew.at(i) - h_resultOld.at(i));
		if (std::abs(h_resultNew.at(i) - h_resultOld.at(i)) > 0.0001)
			++significantDiffCount;
	}

	if (significantDiffCount == 0)
	{
		std::cout << "New = " << newDuration.count() << "\n";
		std::cout << "Old = " << oldDuration.count() << "\n";
	}
	else
	{
		std::cout << "Kernel results do not match! " << significantDiffCount << " / " << size << "\n";
	}
	std::cout << "Total abs diff = " << totalDiff << "\n";

	cudaFree(&h_colour1);
	cudaFree(&h_colour2);
	cudaFree(&h_resultNew);
	cudaFree(&h_resultOld);
}

void Benchmark_MathKernels_Euclidean()
{
	std::cout << __FILE__ << "   " << __func__ << "\n";

	MathKernelsCompare(1000000, euclidNoPowWrapper, euclidPowWrapper);
}

void Benchmark_MathKernels_CIEDE2000()
{
	std::cout << __FILE__ << "   " << __func__ << "\n";

	MathKernelsCompare(1000000, CIEDE2000NewWrapper, CIEDE2000OldWrapper);
}

void Benchmark_MathKernels()
{
	Benchmark_MathKernels_Euclidean();
	Benchmark_MathKernels_CIEDE2000();
}