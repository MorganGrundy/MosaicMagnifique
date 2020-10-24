#ifndef TST_GENERATOR_H
#define TST_GENERATOR_H

#include <gtest/gtest.h>
#include <gmock/gmock-matchers.h>
#include <cuda_runtime.h>

#include "cpuphotomosaicgenerator.h"
#include "cudaphotomosaicgenerator.h"
#include "tst_colourdifference.h"
#include "gridgenerator.h"

using namespace testing;

enum class ColourSpace {BGR, CIELAB};

//Generates a random image of given size and colour space
cv::Mat createRandomImage(const int width, const int height, const ColourSpace colourSpace)
{
    cv::Mat randIm(height, width, CV_32FC3);

    cv::Vec3f *p_im;
    for (int row = 0; row < height; ++row)
    {
        p_im = randIm.ptr<cv::Vec3f>(row);
        for (int col = 0; col < width; ++col)
        {
            if (colourSpace == ColourSpace::BGR)
            {
                p_im[col][0] = (rand() * 100) / RAND_MAX;
                p_im[col][1] = (rand() * 100) / RAND_MAX;
                p_im[col][2] = (rand() * 100) / RAND_MAX;
            }
            else if (colourSpace == ColourSpace::CIELAB)
            {
                p_im[col][0] = randFloat(0, 100);
                p_im[col][1] = randFloat(-128, 127);
                p_im[col][2] = randFloat(-128, 127);
            }
        }
    }

    return randIm;
}

//Compare
bool bestFitsEqual(const GridUtility::MosaicBestFit &bestFit1,
                   const GridUtility::MosaicBestFit &bestFit2)
{
    if (bestFit1.size() != bestFit2.size())
        return false;

    for (size_t step = 0; step < bestFit1.size(); ++step)
    {
        if (bestFit1.at(step).size() != bestFit2.at(step).size())
            return false;

        for (size_t y = 0; y < bestFit1.at(step).size(); ++y)
        {
            if (bestFit1.at(step).at(y).size() != bestFit2.at(step).at(y).size())
                return false;

            for (size_t x = 0; x < bestFit1.at(step).at(y).size(); ++x)
            {
                if (bestFit1.at(step).at(y).at(x) != bestFit2.at(step).at(y).at(x))
                    return false;
            }
        }
    }

    return true;
}

TEST(Generator, BestFits_NoRepeats)
{
    srand(static_cast<unsigned int>(time(NULL)));

    CPUPhotomosaicGenerator generator;
    CUDAPhotomosaicGenerator generatorCUDA;
    generatorCUDA.setLibraryBatchSize(100);

    generator.setMode(CPUPhotomosaicGenerator::Mode::CIE76);
    generatorCUDA.setMode(CPUPhotomosaicGenerator::Mode::CIE76);

    const size_t mosaicWidth = 1000, mosaicHeight = 1000;
    const cv::Mat mainImage = createRandomImage(mosaicWidth, mosaicHeight, ColourSpace::CIELAB);
    generator.setMainImage(mainImage);
    generatorCUDA.setMainImage(mainImage);

    //Square cell shape
    const size_t cellSize = 64;
    CellGroup cellGroup;
    cellGroup.setCellShape(CellShape(cellSize));
    cellGroup.setSizeSteps(0);
    cellGroup.setDetail(100);
    generator.setCellGroup(cellGroup);
    generatorCUDA.setCellGroup(cellGroup);

    //Create image library
    const size_t libSize = 100;
    std::vector<cv::Mat> lib;
    for (size_t i = 0; i < libSize; ++i)
        lib.push_back(createRandomImage(cellSize, cellSize, ColourSpace::CIELAB));
    generator.setLibrary(lib);
    generatorCUDA.setLibrary(lib);

    const GridUtility::MosaicBestFit gridState =
        GridGenerator::getGridState(cellGroup, mainImage, mosaicHeight, mosaicWidth);
    generator.setGridState(gridState);
    generatorCUDA.setGridState(gridState);

    generator.setRepeat(0, 0);
    generatorCUDA.setRepeat(0, 0);

    ASSERT_TRUE(generator.generateBestFits());
    ASSERT_TRUE(generatorCUDA.generateBestFits());
    ASSERT_TRUE(bestFitsEqual(generator.getBestFits(), generatorCUDA.getBestFits()));
}

TEST(Generator, BestFits_Repeats)
{
    srand(static_cast<unsigned int>(time(NULL)));

    CPUPhotomosaicGenerator generator;
    CUDAPhotomosaicGenerator generatorCUDA;
    generatorCUDA.setLibraryBatchSize(100);

    generator.setMode(CPUPhotomosaicGenerator::Mode::CIE76);
    generatorCUDA.setMode(CPUPhotomosaicGenerator::Mode::CIE76);

    const size_t mosaicWidth = 1000, mosaicHeight = 1000;
    const cv::Mat mainImage = createRandomImage(mosaicWidth, mosaicHeight, ColourSpace::CIELAB);
    generator.setMainImage(mainImage);
    generatorCUDA.setMainImage(mainImage);

    //Square cell shape
    const size_t cellSize = 64;
    CellGroup cellGroup;
    cellGroup.setCellShape(CellShape(cellSize));
    cellGroup.setSizeSteps(0);
    cellGroup.setDetail(100);
    generator.setCellGroup(cellGroup);
    generatorCUDA.setCellGroup(cellGroup);

    //Create image library
    const size_t libSize = 100;
    std::vector<cv::Mat> lib;
    for (size_t i = 0; i < libSize; ++i)
        lib.push_back(createRandomImage(cellSize, cellSize, ColourSpace::CIELAB));
    generator.setLibrary(lib);
    generatorCUDA.setLibrary(lib);

    const GridUtility::MosaicBestFit gridState =
        GridGenerator::getGridState(cellGroup, mainImage, mosaicHeight, mosaicWidth);
    generator.setGridState(gridState);
    generatorCUDA.setGridState(gridState);

    generator.setRepeat(20, 10000);
    generatorCUDA.setRepeat(20, 10000);

    ASSERT_TRUE(generator.generateBestFits());
    ASSERT_TRUE(generatorCUDA.generateBestFits());
    ASSERT_TRUE(bestFitsEqual(generator.getBestFits(), generatorCUDA.getBestFits()));
}

#endif // TST_GENERATOR_H
