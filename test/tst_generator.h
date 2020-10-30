#ifndef TST_GENERATOR_H
#define TST_GENERATOR_H

#include <gtest/gtest.h>
#include <gmock/gmock-matchers.h>
#include <cuda_runtime.h>
#include <QDir>
#include <opencv2/imgcodecs.hpp>

#include "cpuphotomosaicgenerator.h"
#include "cudaphotomosaicgenerator.h"
#include "testutility.h"
#include "gridgenerator.h"
#include "imagelibrary.h"

using namespace testing;

namespace TST_Generator
{
//Check if best fits are equal
size_t bestFitsDiff(const GridUtility::MosaicBestFit &bestFit1,
                    const GridUtility::MosaicBestFit &bestFit2)
{
    const size_t maxSteps = std::max(bestFit1.size(), bestFit2.size());
    const size_t minSteps = std::min(bestFit1.size(), bestFit2.size());

    const size_t maxRows = std::max(bestFit1.at(0).size(), bestFit2.at(0).size());
    const size_t minRows = std::min(bestFit1.at(0).size(), bestFit2.at(0).size());

    const size_t maxCols = std::max(bestFit1.at(0).at(0).size(), bestFit2.at(0).at(0).size());
    const size_t minCols = std::min(bestFit1.at(0).at(0).size(), bestFit2.at(0).at(0).size());

    size_t differences = 0;
    if (bestFit1.size() != bestFit2.size())
        differences += (maxSteps - minSteps) * maxRows * maxCols;

    for (size_t step = 0; step < minSteps; ++step)
    {
        if (bestFit1.at(step).size() != bestFit2.at(step).size())
            differences += (maxRows - minRows) * maxCols;

        for (size_t y = 0; y < minRows; ++y)
        {
            if (bestFit1.at(step).at(y).size() != bestFit2.at(step).at(y).size())
                differences += (maxCols - minCols);

            for (size_t x = 0; x < minCols; ++x)
            {
                if (bestFit1.at(step).at(y).at(x) != bestFit2.at(step).at(y).at(x))
                {
                    ++differences;
                    std::cout << "Difference at (x:" << x << ", y:" << y << ")\n";
                }
            }
        }
    }

    return differences;
}

}

TEST(Generator, CIE76vsCUDA_CIE76)
{
    //Create folder for saving photomosaics
    QDir resultFolder(QDir::currentPath() + "/testcases/generator");
    if (!resultFolder.exists())
        resultFolder.mkpath(".");

    //Create photomosaic generators
    CPUPhotomosaicGenerator generator;
    CUDAPhotomosaicGenerator generatorCUDA;
    generatorCUDA.setLibraryBatchSize(100);

    //Set mode
    PhotomosaicGeneratorBase::Mode mode = PhotomosaicGeneratorBase::Mode::CIE76;
    generator.setMode(mode);
    generatorCUDA.setMode(mode);

    //Set cell group
    const size_t cellSize = 128;
    CellGroup cellGroup;
    cellGroup.setCellShape(CellShape(cellSize));
    cellGroup.setSizeSteps(0);
    cellGroup.setDetail(50);
    generator.setCellGroup(cellGroup);
    generatorCUDA.setCellGroup(cellGroup);

    //Set repeat range and addition
    generator.setRepeat(0, 0);
    generatorCUDA.setRepeat(0, 0);

    //Load and set image library
    ImageLibrary lib(128);
    lib.loadFromFile("E:/Desktop/MosaicMagnifique/MosaicMagnifique/Library/lib.mil");
    generator.setLibrary(lib.getImages());
    generatorCUDA.setLibrary(lib.getImages());

    //Folder containing multiple images
    QDir imageFolder("E:/Desktop/MosaicMagnifique/NewLib");
    QStringList images = imageFolder.entryList(QDir::Filter::Files);

    //Iterate over all images
    for (const auto &image: images)
    {
        //Load and set main image
        cv::Mat mainImage = cv::imread((imageFolder.path() + "/" + image).toStdString());
        generator.setMainImage(mainImage);
        generatorCUDA.setMainImage(mainImage);

        //Generate grid state
        const GridUtility::MosaicBestFit gridState =
            GridGenerator::getGridState(cellGroup, mainImage, mainImage.rows, mainImage.cols);
        generator.setGridState(gridState);
        generatorCUDA.setGridState(gridState);

        //Generate best fits
        ASSERT_TRUE(generator.generateBestFits());
        ASSERT_TRUE(generatorCUDA.generateBestFits());

        //Compare best fits
        size_t bestFitDiff = TST_Generator::bestFitsDiff(generator.getBestFits(),
                                                         generatorCUDA.getBestFits());

        EXPECT_EQ(bestFitDiff, 0) << image.toStdString();

        //Best fits different, save results
        if (bestFitDiff != 0)
        {
            //Build photomosaics
            cv::Mat result = generator.buildPhotomosaic();
            cv::Mat resultCUDA = generatorCUDA.buildPhotomosaic();

            //Save photomosaics
            cv::imwrite((resultFolder.path() + "/" + image).toStdString(), result);
            cv::imwrite((resultFolder.path() + "/CUDA-" + image).toStdString(), resultCUDA);
        }
    }
}

#endif // TST_GENERATOR_H
