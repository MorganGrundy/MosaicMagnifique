#pragma once

#include <chrono>
#include <iostream>
#include <chrono>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

#include "MathKernels.cuh"
#include "..\test\TestUtility.h"

void Benchmark_MathKernels_Euclidean()
{
	std::cout << __FILE__ << "   " << __func__ << "\n";

	int size = 1000000;

	auto h_colour1 = TestUtility::createRandomFloats(size * 3, { {0, 255} });
	auto h_colour2 = TestUtility::createRandomFloats(size * 3, { {0, 255} });
	float *d_colour1, *d_colour2;
	cudaMalloc((void **)&d_colour1, size * 3 * sizeof(float));
	cudaMalloc((void **)&d_colour2, size * 3 * sizeof(float));
	cudaMemcpy(d_colour1, h_colour1.data(), size * 3 * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_colour2, h_colour2.data(), size * 3 * sizeof(float), cudaMemcpyHostToDevice);

	std::vector<double> h_result1(size, 0), h_result2(size, 0);
	double *d_result1, *d_result2;
	cudaMalloc((void **)&d_result1, size * sizeof(double));
	cudaMalloc((void **)&d_result2, size * sizeof(double));
	cudaMemcpy(d_result1, h_result1.data(), size * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(d_result2, h_result2.data(), size * sizeof(double), cudaMemcpyHostToDevice);

	cudaDeviceSynchronize();

	auto powStart = std::chrono::high_resolution_clock::now();
	euclidPowWrapper(d_colour1, d_colour2, d_result1, size);
	cudaDeviceSynchronize();
	auto powDuration = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - powStart);

	auto noPowStart = std::chrono::high_resolution_clock::now();
	euclidNoPowWrapper(d_colour1, d_colour2, d_result2, size);
	cudaDeviceSynchronize();
	auto noPowDuration = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - noPowStart);

	cudaMemcpy(h_result1.data(), d_result1, size * sizeof(double), cudaMemcpyDeviceToHost);
	cudaMemcpy(h_result2.data(), d_result2, size * sizeof(double), cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();

	double totalDiff = 0.0;
	int significantDiffCount = 0;
	for (int i = 0; i < size; ++i)
	{
		totalDiff += std::abs(h_result1.at(i) - h_result2.at(i));
		if (std::abs(h_result1.at(i) - h_result2.at(i)) > 0.0001)
			++significantDiffCount;
	}

	if (significantDiffCount == 0)
	{
		std::cout << "With pow = " << powDuration.count() << "\n";
		std::cout << "Without pow = " << noPowDuration.count() << "\n";
	}
	else
	{
		std::cout << "Kernel results do not match! " << significantDiffCount << " / " << size << "\n";
	}
	std::cout << "Total abs diff = " << totalDiff << "\n";

	cudaFree(&h_colour1);
	cudaFree(&h_colour2);
	cudaFree(&h_result1);
	cudaFree(&h_result2);
}

void Benchmark_MathKernels_CIEDE2000()
{
	std::cout << __FILE__ << "   " << __func__ << "\n";

	int size = 1000000;

	auto h_colour1 = TestUtility::createRandomFloats(size * 3, { {0, 100}, {-128, 127}, {-128, 127} });
	auto h_colour2 = TestUtility::createRandomFloats(size * 3, { {0, 100}, {-128, 127}, {-128, 127} });
	float *d_colour1, *d_colour2;
	cudaMalloc((void **)&d_colour1, size * 3 * sizeof(float));
	cudaMalloc((void **)&d_colour2, size * 3 * sizeof(float));
	cudaMemcpy(d_colour1, h_colour1.data(), size * 3 * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_colour2, h_colour2.data(), size * 3 * sizeof(float), cudaMemcpyHostToDevice);

	std::vector<double> h_result1(size, 0), h_result2(size, 0);
	double *d_result1, *d_result2;
	cudaMalloc((void **)&d_result1, size * sizeof(double));
	cudaMalloc((void **)&d_result2, size * sizeof(double));
	cudaMemcpy(d_result1, h_result1.data(), size * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(d_result2, h_result2.data(), size * sizeof(double), cudaMemcpyHostToDevice);

	cudaDeviceSynchronize();

	auto newStart = std::chrono::high_resolution_clock::now();
	CIEDE2000NewWrapper(d_colour1, d_colour2, d_result1, size);
	cudaDeviceSynchronize();
	auto newDuration = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - newStart);

	auto oldStart = std::chrono::high_resolution_clock::now();
	CIEDE2000OldWrapper(d_colour1, d_colour2, d_result2, size);
	cudaDeviceSynchronize();
	auto oldDuration = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - oldStart);

	cudaMemcpy(h_result1.data(), d_result1, size * sizeof(double), cudaMemcpyDeviceToHost);
	cudaMemcpy(h_result2.data(), d_result2, size * sizeof(double), cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();

	double totalDiff = 0.0;
	int significantDiffCount = 0;
	for (int i = 0; i < size; ++i)
	{
		totalDiff += std::abs(h_result1.at(i) - h_result2.at(i));
		if (std::abs(h_result1.at(i) - h_result2.at(i)) > 0.0001)
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
	cudaFree(&h_result1);
	cudaFree(&h_result2);
}

void Benchmark_MathKernels()
{
	//Benchmark_MathKernels_Euclidean();
	Benchmark_MathKernels_CIEDE2000();
}