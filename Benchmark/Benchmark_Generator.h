#pragma once

#include <QDir>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <chrono>
#include <utility>
#include <iomanip>

#include "..\src\Photomosaic\CUDA\CUDAPhotomosaicGenerator.h"
#include "..\src\Photomosaic\CPUPhotomosaicGenerator.h"
#include "..\src\ImageLibrary\ImageLibrary.h"
#include "..\src\Grid\GridGenerator.h"
#include "..\test\testutility.h"

//Generates a photomosaic with the given parameters and returns the time
size_t GeneratePhotomosaic(const QString &libFile, const std::string &mainImageFile, const QString &cellShapeFile, const bool useCUDA, const ColourDifference::Type colourDiff = ColourDifference::Type::CIE76,
    const int detail = 100, const size_t sizeSteps = 0, const size_t cellSize = 128, const double mainImageSize = 1.0, const int repeatRange = 0, const int repeatAddition = 0, const ColourScheme::Type colourScheme = ColourScheme::Type::NONE)
{
    //Choose which generator to use
    std::shared_ptr<PhotomosaicGeneratorBase> generator;
    if (useCUDA)
        generator = std::make_shared<CUDAPhotomosaicGenerator>(0);
    else
        generator = std::make_shared<CPUPhotomosaicGenerator>();

    //Set colour difference
    generator->setColourDifference(colourDiff);

    //Set colour scheme
    generator->setColourScheme(colourScheme);

    //Set cell group
    CellShape cellShape(cellSize);
    try
    {
        cellShape.loadFromFile(cellShapeFile);
    }
    catch (const std::exception &e) {}
    CellGroup cellGroup;
    cellGroup.setCellShape(cellShape);
    cellGroup.setSizeSteps(sizeSteps);
    cellGroup.setDetail(detail);
    generator->setCellGroup(cellGroup);

    //Set repeat range and addition
    generator->setRepeat(repeatRange, repeatAddition);

    //Load and set image library
    ImageLibrary lib(cellSize);
    lib.loadFromFile(libFile);
    if (lib.getImages().empty())
    {
        std::cout << "Image Library is empty.";
        throw std::invalid_argument("Image Library is empty.");
    }
    generator->setLibrary(lib.getImages());

    //Load and set main image
    cv::Mat mainImage = cv::imread(mainImageFile);
    if (mainImage.empty())
    {
        std::cout << "Main Image is empty.";
        throw std::invalid_argument("Main Image is empty.");
    }
    cv::resize(mainImage, mainImage, cv::Size(), mainImageSize, mainImageSize);
    generator->setMainImage(mainImage);

    //Generate grid state
    const GridUtility::MosaicBestFit gridState = GridGenerator::getGridState(cellGroup, mainImage, mainImage.rows, mainImage.cols);
    generator->setGridState(gridState);

    //Generate Photomosaic and measure time
    const auto startTime = std::chrono::high_resolution_clock::now();
    if (!generator->generateBestFits())
    {
        std::cout << "Failed to generate.";
        throw std::invalid_argument("Failed to generate.");
    }
    return std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - startTime).count();
}

void LogTimes(std::vector<size_t> &timeCPU, std::vector<size_t> &timeCUDA)
{
    std::sort(timeCPU.begin(), timeCPU.end());
    std::sort(timeCUDA.begin(), timeCUDA.end());

    const size_t textWidth = std::to_string(std::max(timeCPU.back(), timeCUDA.back())).length();
    const double averageCPU = std::round(std::accumulate(timeCPU.cbegin(), timeCPU.cend(), 0.0) / timeCPU.size());
    const double averageCUDA = std::round(std::accumulate(timeCUDA.cbegin(), timeCUDA.cend(), 0.0) / timeCUDA.size());

    std::cout << "Time (CPU):  " << std::setw(textWidth) << averageCPU << "ms -> ";
    for (auto time : timeCPU)
        std::cout << std::setw(textWidth) << time << "ms, ";
    std::cout << "\n";

    std::cout << "Time (CUDA): " << std::setw(textWidth) << averageCUDA << "ms -> ";
    for (auto time : timeCUDA)
        std::cout << std::setw(textWidth) << time << "ms, ";
    std::cout << "\n";

    const double speedupCUDA = averageCPU / averageCUDA;
    std::cout << "CUDA Speedup = x" << std::setprecision(6) << speedupCUDA << "\n";
}

//Generates a photomosaic with the given parameters on both CPU and CUDA iteration times, and outputs the times
void PerfGenerate(const size_t iterations, const QString &libFile, const std::string &mainImageFile, const QString &cellShapeFile, const ColourDifference::Type colourDiff = ColourDifference::Type::CIE76,
    const int detail = 100, const size_t sizeSteps = 0, const size_t cellSize = 128, const double mainImageSize = 1.0, const int repeatRange = 0, const int repeatAddition = 0, const ColourScheme::Type colourScheme = ColourScheme::Type::NONE)
{
    std::vector<size_t> timeCPU, timeCUDA;
    for (size_t i = 0; i < iterations; ++i)
    {
        for (const auto useCUDA : { false, true })
        {
            auto generateTime = GeneratePhotomosaic(libFile, mainImageFile, cellShapeFile, useCUDA, colourDiff, detail, sizeSteps, cellSize, mainImageSize, repeatRange, repeatAddition, colourScheme);
            if (useCUDA)
                timeCUDA.push_back(generateTime);
            else
                timeCPU.push_back(generateTime);
        }
    }
    LogTimes(timeCPU, timeCUDA);
}

void Benchmark_Generator_ColourDifference()
{
    std::cout << __FILE__ << "   " << __func__ << "\n";

    for (size_t colourDiff = 0; colourDiff < static_cast<size_t>(ColourDifference::Type::MAX); ++colourDiff)
    {
        std::cout << "Colour Difference = " << ColourDifference::Type_STR.at(colourDiff).toStdString() << "\n";
        PerfGenerate(2, TestUtility::LIB_FILE, TestUtility::EDGAR_PEREZ, "", static_cast<ColourDifference::Type>(colourDiff), 20);
    }
}

void Benchmark_Generator_Detail()
{
    std::cout << __FILE__ << "   " << __func__ << "\n";
    for (auto detail: {0, 25, 50, 75, 100})
    {
        std::cout << "Detail = " << detail << "%\n";
        PerfGenerate(2, TestUtility::LIB_FILE, TestUtility::EDGAR_PEREZ, "", ColourDifference::Type::CIE76, detail);
    }
}

void Benchmark_Generator_SizeSteps()
{
    std::cout << __FILE__ << "   " << __func__ << "\n";
    for (auto sizeSteps : { 0, 1, 2 })
    {
        std::cout << "Size Steps = " << sizeSteps << "\n";
        PerfGenerate(2, TestUtility::LIB_FILE, TestUtility::EDGAR_PEREZ, "", ColourDifference::Type::CIE76, 20, sizeSteps);
    }
}

void Benchmark_Generator_CellSize()
{
    std::cout << __FILE__ << "   " << __func__ << "\n";
    for (auto cellSize : { 128, 64, 32, 16, 8 })
    {
        std::cout << "Cell Size = " << cellSize << "\n";
        PerfGenerate(2, TestUtility::LIB_FILE, TestUtility::EDGAR_PEREZ, "", ColourDifference::Type::CIE76, 20, 0, cellSize);
    }
}

void Benchmark_Generator_Lib()
{
    std::cout << __FILE__ << "   " << __func__ << "\n";

    for (const auto &lib : { TestUtility::LIB_FILE, TestUtility::BIG_LIB_FILE })
    {
        std::cout << "Image Library = " << lib.toStdString() << "\n";
        PerfGenerate(2, lib, TestUtility::EDGAR_PEREZ, "", ColourDifference::Type::CIE76, 20);
    }
}

void Benchmark_Generator()
{
    Benchmark_Generator_ColourDifference();
    Benchmark_Generator_Detail();
    Benchmark_Generator_SizeSteps();
    Benchmark_Generator_CellSize();
    Benchmark_Generator_Lib();
}