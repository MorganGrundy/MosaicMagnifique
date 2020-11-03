#ifndef TST_GENERATOR_H
#define TST_GENERATOR_H

#include <gtest/gtest.h>
#include <gmock/gmock-matchers.h>
#include <QDir>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#ifdef CUDA
#include <cuda_runtime.h>
#include "cudaphotomosaicgenerator.h"
#endif

#include "cpuphotomosaicgenerator.h"
#include "testutility.h"
#include "gridgenerator.h"
#include "imagelibrary.h"

using namespace testing;

namespace TST_Generator
{
static const size_t ITERATIONS = 10;

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

////////////////////////////////////////////////////////////////////////////////////////////////////
//******************************************** DETAIL ********************************************//
////////////////////////////////////////////////////////////////////////////////////////////////////

//Generates photomosaic best fits using identical settings multiple times
//Expects all the best fits to be identical
TEST(Generator, RGB_EUCLIDEAN_Detail_100)
{
    //Create photomosaic generator
    CPUPhotomosaicGenerator generator;

    //Set mode
    PhotomosaicGeneratorBase::Mode mode = PhotomosaicGeneratorBase::Mode::RGB_EUCLIDEAN;
    generator.setMode(mode);

    //Set cell group
    const size_t cellSize = 128;
    CellGroup cellGroup;
    cellGroup.setCellShape(CellShape(cellSize));
    cellGroup.setSizeSteps(0);
    cellGroup.setDetail(100);
    generator.setCellGroup(cellGroup);

    //Set repeat range and addition
    generator.setRepeat(0, 0);

    //Load and set image library
    ImageLibrary lib(128);
    lib.loadFromFile("E:/Desktop/MosaicMagnifique/MosaicMagnifique/Library/lib.mil");
    generator.setLibrary(lib.getImages());

    //Folder containing multiple images
    QDir imageFolder("E:/Desktop/MosaicMagnifique/NewLib");
    QStringList images = imageFolder.entryList(QDir::Filter::Files);

    //Select a random main image from list
    cv::Mat mainImage = cv::imread((imageFolder.path() + "/" +
                                    images.at((rand() * images.size()) / RAND_MAX)).toStdString());
    cv::resize(mainImage, mainImage, cv::Size(), 0.5, 0.5);
    generator.setMainImage(mainImage);

    GridUtility::MosaicBestFit lastBestFit;
    for (size_t i = 0; i < TST_Generator::ITERATIONS; ++i)
    {
        //Generate grid state
        const GridUtility::MosaicBestFit gridState =
            GridGenerator::getGridState(cellGroup, mainImage, mainImage.rows, mainImage.cols);
        generator.setGridState(gridState);

        //Generate best fits
        ASSERT_TRUE(generator.generateBestFits());

        if (i > 0)
        {
            //Compare best fits
            size_t bestFitDiff = TST_Generator::bestFitsDiff(generator.getBestFits(), lastBestFit);
            EXPECT_EQ(bestFitDiff, 0);
        }
        lastBestFit = generator.getBestFits();
    }
}

//Generates photomosaic best fits using identical settings multiple times
//Expects all the best fits to be identical
TEST(Generator, RGB_EUCLIDEAN_Detail_50)
{
    //Create photomosaic generator
    CPUPhotomosaicGenerator generator;

    //Set mode
    PhotomosaicGeneratorBase::Mode mode = PhotomosaicGeneratorBase::Mode::RGB_EUCLIDEAN;
    generator.setMode(mode);

    //Set cell group
    const size_t cellSize = 128;
    CellGroup cellGroup;
    cellGroup.setCellShape(CellShape(cellSize));
    cellGroup.setSizeSteps(0);
    cellGroup.setDetail(50);
    generator.setCellGroup(cellGroup);

    //Set repeat range and addition
    generator.setRepeat(0, 0);

    //Load and set image library
    ImageLibrary lib(128);
    lib.loadFromFile("E:/Desktop/MosaicMagnifique/MosaicMagnifique/Library/lib.mil");
    generator.setLibrary(lib.getImages());

    //Folder containing multiple images
    QDir imageFolder("E:/Desktop/MosaicMagnifique/NewLib");
    QStringList images = imageFolder.entryList(QDir::Filter::Files);

    //Select a random main image from list
    cv::Mat mainImage = cv::imread((imageFolder.path() + "/" +
                                    images.at((rand() * images.size()) / RAND_MAX)).toStdString());
    cv::resize(mainImage, mainImage, cv::Size(), 0.5, 0.5);
    generator.setMainImage(mainImage);

    GridUtility::MosaicBestFit lastBestFit;
    for (size_t i = 0; i < TST_Generator::ITERATIONS; ++i)
    {
        //Generate grid state
        const GridUtility::MosaicBestFit gridState =
            GridGenerator::getGridState(cellGroup, mainImage, mainImage.rows, mainImage.cols);
        generator.setGridState(gridState);

        //Generate best fits
        ASSERT_TRUE(generator.generateBestFits());

        if (i > 0)
        {
            //Compare best fits
            size_t bestFitDiff = TST_Generator::bestFitsDiff(generator.getBestFits(), lastBestFit);
            EXPECT_EQ(bestFitDiff, 0);
        }
        lastBestFit = generator.getBestFits();
    }
}

//Generates photomosaic best fits using identical settings multiple times
//Expects all the best fits to be identical
TEST(Generator, CIE76_Detail_100)
{
    //Create photomosaic generator
    CPUPhotomosaicGenerator generator;

    //Set mode
    PhotomosaicGeneratorBase::Mode mode = PhotomosaicGeneratorBase::Mode::CIE76;
    generator.setMode(mode);

    //Set cell group
    const size_t cellSize = 128;
    CellGroup cellGroup;
    cellGroup.setCellShape(CellShape(cellSize));
    cellGroup.setSizeSteps(0);
    cellGroup.setDetail(100);
    generator.setCellGroup(cellGroup);

    //Set repeat range and addition
    generator.setRepeat(0, 0);

    //Load and set image library
    ImageLibrary lib(128);
    lib.loadFromFile("E:/Desktop/MosaicMagnifique/MosaicMagnifique/Library/lib.mil");
    generator.setLibrary(lib.getImages());

    //Folder containing multiple images
    QDir imageFolder("E:/Desktop/MosaicMagnifique/NewLib");
    QStringList images = imageFolder.entryList(QDir::Filter::Files);

    //Select a random main image from list
    cv::Mat mainImage = cv::imread((imageFolder.path() + "/" +
                                    images.at((rand() * images.size()) / RAND_MAX)).toStdString());
    cv::resize(mainImage, mainImage, cv::Size(), 0.5, 0.5);
    generator.setMainImage(mainImage);

    GridUtility::MosaicBestFit lastBestFit;
    for (size_t i = 0; i < TST_Generator::ITERATIONS; ++i)
    {
        //Generate grid state
        const GridUtility::MosaicBestFit gridState =
            GridGenerator::getGridState(cellGroup, mainImage, mainImage.rows, mainImage.cols);
        generator.setGridState(gridState);

        //Generate best fits
        ASSERT_TRUE(generator.generateBestFits());

        if (i > 0)
        {
            //Compare best fits
            size_t bestFitDiff = TST_Generator::bestFitsDiff(generator.getBestFits(), lastBestFit);
            EXPECT_EQ(bestFitDiff, 0);
        }
        lastBestFit = generator.getBestFits();
    }
}

//Generates photomosaic best fits using identical settings multiple times
//Expects all the best fits to be identical
TEST(Generator, CIE76_Detail_50)
{
    //Create photomosaic generator
    CPUPhotomosaicGenerator generator;

    //Set mode
    PhotomosaicGeneratorBase::Mode mode = PhotomosaicGeneratorBase::Mode::CIE76;
    generator.setMode(mode);

    //Set cell group
    const size_t cellSize = 128;
    CellGroup cellGroup;
    cellGroup.setCellShape(CellShape(cellSize));
    cellGroup.setSizeSteps(0);
    cellGroup.setDetail(50);
    generator.setCellGroup(cellGroup);

    //Set repeat range and addition
    generator.setRepeat(0, 0);

    //Load and set image library
    ImageLibrary lib(128);
    lib.loadFromFile("E:/Desktop/MosaicMagnifique/MosaicMagnifique/Library/lib.mil");
    generator.setLibrary(lib.getImages());

    //Folder containing multiple images
    QDir imageFolder("E:/Desktop/MosaicMagnifique/NewLib");
    QStringList images = imageFolder.entryList(QDir::Filter::Files);

    //Select a random main image from list
    cv::Mat mainImage = cv::imread((imageFolder.path() + "/" +
                                    images.at((rand() * images.size()) / RAND_MAX)).toStdString());
    cv::resize(mainImage, mainImage, cv::Size(), 0.5, 0.5);
    generator.setMainImage(mainImage);

    GridUtility::MosaicBestFit lastBestFit;
    for (size_t i = 0; i < TST_Generator::ITERATIONS; ++i)
    {
        //Generate grid state
        const GridUtility::MosaicBestFit gridState =
            GridGenerator::getGridState(cellGroup, mainImage, mainImage.rows, mainImage.cols);
        generator.setGridState(gridState);

        //Generate best fits
        ASSERT_TRUE(generator.generateBestFits());

        if (i > 0)
        {
            //Compare best fits
            size_t bestFitDiff = TST_Generator::bestFitsDiff(generator.getBestFits(), lastBestFit);
            EXPECT_EQ(bestFitDiff, 0);
        }
        lastBestFit = generator.getBestFits();
    }
}

//Generates photomosaic best fits using identical settings multiple times
//Expects all the best fits to be identical
TEST(Generator, CIEDE2000_Detail_100)
{
    //Create photomosaic generator
    CPUPhotomosaicGenerator generator;

    //Set mode
    PhotomosaicGeneratorBase::Mode mode = PhotomosaicGeneratorBase::Mode::CIEDE2000;
    generator.setMode(mode);

    //Set cell group
    const size_t cellSize = 128;
    CellGroup cellGroup;
    cellGroup.setCellShape(CellShape(cellSize));
    cellGroup.setSizeSteps(0);
    cellGroup.setDetail(100);
    generator.setCellGroup(cellGroup);

    //Set repeat range and addition
    generator.setRepeat(0, 0);

    //Load and set image library
    ImageLibrary lib(128);
    lib.loadFromFile("E:/Desktop/MosaicMagnifique/MosaicMagnifique/Library/lib.mil");
    generator.setLibrary(lib.getImages());

    //Folder containing multiple images
    QDir imageFolder("E:/Desktop/MosaicMagnifique/NewLib");
    QStringList images = imageFolder.entryList(QDir::Filter::Files);

    //Select a random main image from list
    cv::Mat mainImage = cv::imread((imageFolder.path() + "/" +
                                    images.at((rand() * images.size()) / RAND_MAX)).toStdString());
    cv::resize(mainImage, mainImage, cv::Size(), 0.5, 0.5);
    generator.setMainImage(mainImage);

    GridUtility::MosaicBestFit lastBestFit;
    for (size_t i = 0; i < TST_Generator::ITERATIONS; ++i)
    {
        //Generate grid state
        const GridUtility::MosaicBestFit gridState =
            GridGenerator::getGridState(cellGroup, mainImage, mainImage.rows, mainImage.cols);
        generator.setGridState(gridState);

        //Generate best fits
        ASSERT_TRUE(generator.generateBestFits());

        if (i > 0)
        {
            //Compare best fits
            size_t bestFitDiff = TST_Generator::bestFitsDiff(generator.getBestFits(), lastBestFit);
            EXPECT_EQ(bestFitDiff, 0);
        }
        lastBestFit = generator.getBestFits();
    }
}

//Generates photomosaic best fits using identical settings multiple times
//Expects all the best fits to be identical
TEST(Generator, CIEDE2000_Detail_50)
{
    //Create photomosaic generator
    CPUPhotomosaicGenerator generator;

    //Set mode
    PhotomosaicGeneratorBase::Mode mode = PhotomosaicGeneratorBase::Mode::CIEDE2000;
    generator.setMode(mode);

    //Set cell group
    const size_t cellSize = 128;
    CellGroup cellGroup;
    cellGroup.setCellShape(CellShape(cellSize));
    cellGroup.setSizeSteps(0);
    cellGroup.setDetail(50);
    generator.setCellGroup(cellGroup);

    //Set repeat range and addition
    generator.setRepeat(0, 0);

    //Load and set image library
    ImageLibrary lib(128);
    lib.loadFromFile("E:/Desktop/MosaicMagnifique/MosaicMagnifique/Library/lib.mil");
    generator.setLibrary(lib.getImages());

    //Folder containing multiple images
    QDir imageFolder("E:/Desktop/MosaicMagnifique/NewLib");
    QStringList images = imageFolder.entryList(QDir::Filter::Files);

    //Select a random main image from list
    cv::Mat mainImage = cv::imread((imageFolder.path() + "/" +
                                    images.at((rand() * images.size()) / RAND_MAX)).toStdString());
    cv::resize(mainImage, mainImage, cv::Size(), 0.5, 0.5);
    generator.setMainImage(mainImage);

    GridUtility::MosaicBestFit lastBestFit;
    for (size_t i = 0; i < TST_Generator::ITERATIONS; ++i)
    {
        //Generate grid state
        const GridUtility::MosaicBestFit gridState =
            GridGenerator::getGridState(cellGroup, mainImage, mainImage.rows, mainImage.cols);
        generator.setGridState(gridState);

        //Generate best fits
        ASSERT_TRUE(generator.generateBestFits());

        if (i > 0)
        {
            //Compare best fits
            size_t bestFitDiff = TST_Generator::bestFitsDiff(generator.getBestFits(), lastBestFit);
            EXPECT_EQ(bestFitDiff, 0);
        }
        lastBestFit = generator.getBestFits();
    }
}


#ifdef CUDA
////////////////////////////////////////////////////////////////////////////////////////////////////
//***************************************** CUDA DETAIL ******************************************//
////////////////////////////////////////////////////////////////////////////////////////////////////

//Generates (with CUDA) photomosaic best fits using identical settings multiple times
//Expects all the best fits to be identical
TEST(Generator, CUDA_RGB_EUCLIDEAN_Detail_100)
{
    //Create photomosaic generators
    CUDAPhotomosaicGenerator generator;
    generator.setLibraryBatchSize(1000);

    //Set mode
    PhotomosaicGeneratorBase::Mode mode = PhotomosaicGeneratorBase::Mode::RGB_EUCLIDEAN;
    generator.setMode(mode);

    //Set cell group
    const size_t cellSize = 128;
    CellGroup cellGroup;
    cellGroup.setCellShape(CellShape(cellSize));
    cellGroup.setSizeSteps(0);
    cellGroup.setDetail(100);
    generator.setCellGroup(cellGroup);

    //Set repeat range and addition
    generator.setRepeat(0, 0);

    //Load and set image library
    ImageLibrary lib(128);
    lib.loadFromFile("E:/Desktop/MosaicMagnifique/MosaicMagnifique/Library/lib.mil");
    generator.setLibrary(lib.getImages());

    //Folder containing multiple images
    QDir imageFolder("E:/Desktop/MosaicMagnifique/NewLib");
    QStringList images = imageFolder.entryList(QDir::Filter::Files);

    //Select a random main image from list
    cv::Mat mainImage = cv::imread((imageFolder.path() + "/" +
                                    images.at((rand() * images.size()) / RAND_MAX)).toStdString());
    cv::resize(mainImage, mainImage, cv::Size(), 0.5, 0.5);
    generator.setMainImage(mainImage);

    GridUtility::MosaicBestFit lastBestFit;
    for (size_t i = 0; i < TST_Generator::ITERATIONS; ++i)
    {
        //Generate grid state
        const GridUtility::MosaicBestFit gridState =
            GridGenerator::getGridState(cellGroup, mainImage, mainImage.rows, mainImage.cols);
        generator.setGridState(gridState);

        //Generate best fits
        ASSERT_TRUE(generator.generateBestFits());

        if (i > 0)
        {
            //Compare best fits
            size_t bestFitDiff = TST_Generator::bestFitsDiff(generator.getBestFits(), lastBestFit);
            EXPECT_EQ(bestFitDiff, 0);
        }
        lastBestFit = generator.getBestFits();
    }
}

//Generates (with CUDA) photomosaic best fits using identical settings multiple times
//Expects all the best fits to be identical
TEST(Generator, CUDA_RGB_EUCLIDEAN_Detail_50)
{
    //Create photomosaic generators
    CUDAPhotomosaicGenerator generator;
    generator.setLibraryBatchSize(1000);

    //Set mode
    PhotomosaicGeneratorBase::Mode mode = PhotomosaicGeneratorBase::Mode::RGB_EUCLIDEAN;
    generator.setMode(mode);

    //Set cell group
    const size_t cellSize = 128;
    CellGroup cellGroup;
    cellGroup.setCellShape(CellShape(cellSize));
    cellGroup.setSizeSteps(0);
    cellGroup.setDetail(50);
    generator.setCellGroup(cellGroup);

    //Set repeat range and addition
    generator.setRepeat(0, 0);

    //Load and set image library
    ImageLibrary lib(128);
    lib.loadFromFile("E:/Desktop/MosaicMagnifique/MosaicMagnifique/Library/lib.mil");
    generator.setLibrary(lib.getImages());

    //Folder containing multiple images
    QDir imageFolder("E:/Desktop/MosaicMagnifique/NewLib");
    QStringList images = imageFolder.entryList(QDir::Filter::Files);

    //Select a random main image from list
    cv::Mat mainImage = cv::imread((imageFolder.path() + "/" +
                                    images.at((rand() * images.size()) / RAND_MAX)).toStdString());
    cv::resize(mainImage, mainImage, cv::Size(), 0.5, 0.5);
    generator.setMainImage(mainImage);

    GridUtility::MosaicBestFit lastBestFit;
    for (size_t i = 0; i < TST_Generator::ITERATIONS; ++i)
    {
        //Generate grid state
        const GridUtility::MosaicBestFit gridState =
            GridGenerator::getGridState(cellGroup, mainImage, mainImage.rows, mainImage.cols);
        generator.setGridState(gridState);

        //Generate best fits
        ASSERT_TRUE(generator.generateBestFits());

        if (i > 0)
        {
            //Compare best fits
            size_t bestFitDiff = TST_Generator::bestFitsDiff(generator.getBestFits(), lastBestFit);
            EXPECT_EQ(bestFitDiff, 0);
        }
        lastBestFit = generator.getBestFits();
    }
}

//Generates (with CUDA) photomosaic best fits using identical settings multiple times
//Expects all the best fits to be identical
TEST(Generator, CUDA_CIE76_Detail_100)
{
    //Create photomosaic generators
    CUDAPhotomosaicGenerator generator;
    generator.setLibraryBatchSize(1000);

    //Set mode
    PhotomosaicGeneratorBase::Mode mode = PhotomosaicGeneratorBase::Mode::CIE76;
    generator.setMode(mode);

    //Set cell group
    const size_t cellSize = 128;
    CellGroup cellGroup;
    cellGroup.setCellShape(CellShape(cellSize));
    cellGroup.setSizeSteps(0);
    cellGroup.setDetail(100);
    generator.setCellGroup(cellGroup);

    //Set repeat range and addition
    generator.setRepeat(0, 0);

    //Load and set image library
    ImageLibrary lib(128);
    lib.loadFromFile("E:/Desktop/MosaicMagnifique/MosaicMagnifique/Library/lib.mil");
    generator.setLibrary(lib.getImages());

    //Folder containing multiple images
    QDir imageFolder("E:/Desktop/MosaicMagnifique/NewLib");
    QStringList images = imageFolder.entryList(QDir::Filter::Files);

    //Select a random main image from list
    cv::Mat mainImage = cv::imread((imageFolder.path() + "/" +
                                    images.at((rand() * images.size()) / RAND_MAX)).toStdString());
    cv::resize(mainImage, mainImage, cv::Size(), 0.5, 0.5);
    generator.setMainImage(mainImage);

    GridUtility::MosaicBestFit lastBestFit;
    for (size_t i = 0; i < TST_Generator::ITERATIONS; ++i)
    {
        //Generate grid state
        const GridUtility::MosaicBestFit gridState =
            GridGenerator::getGridState(cellGroup, mainImage, mainImage.rows, mainImage.cols);
        generator.setGridState(gridState);

        //Generate best fits
        ASSERT_TRUE(generator.generateBestFits());

        if (i > 0)
        {
            //Compare best fits
            size_t bestFitDiff = TST_Generator::bestFitsDiff(generator.getBestFits(), lastBestFit);
            EXPECT_EQ(bestFitDiff, 0);
        }
        lastBestFit = generator.getBestFits();
    }
}

//Generates (with CUDA) photomosaic best fits using identical settings multiple times
//Expects all the best fits to be identical
TEST(Generator, CUDA_CIE76_Detail_50)
{
    //Create photomosaic generators
    CUDAPhotomosaicGenerator generator;
    generator.setLibraryBatchSize(1000);

    //Set mode
    PhotomosaicGeneratorBase::Mode mode = PhotomosaicGeneratorBase::Mode::CIE76;
    generator.setMode(mode);

    //Set cell group
    const size_t cellSize = 128;
    CellGroup cellGroup;
    cellGroup.setCellShape(CellShape(cellSize));
    cellGroup.setSizeSteps(0);
    cellGroup.setDetail(50);
    generator.setCellGroup(cellGroup);

    //Set repeat range and addition
    generator.setRepeat(0, 0);

    //Load and set image library
    ImageLibrary lib(128);
    lib.loadFromFile("E:/Desktop/MosaicMagnifique/MosaicMagnifique/Library/lib.mil");
    generator.setLibrary(lib.getImages());

    //Folder containing multiple images
    QDir imageFolder("E:/Desktop/MosaicMagnifique/NewLib");
    QStringList images = imageFolder.entryList(QDir::Filter::Files);

    //Select a random main image from list
    cv::Mat mainImage = cv::imread((imageFolder.path() + "/" +
                                    images.at((rand() * images.size()) / RAND_MAX)).toStdString());
    cv::resize(mainImage, mainImage, cv::Size(), 0.5, 0.5);
    generator.setMainImage(mainImage);

    GridUtility::MosaicBestFit lastBestFit;
    for (size_t i = 0; i < TST_Generator::ITERATIONS; ++i)
    {
        //Generate grid state
        const GridUtility::MosaicBestFit gridState =
            GridGenerator::getGridState(cellGroup, mainImage, mainImage.rows, mainImage.cols);
        generator.setGridState(gridState);

        //Generate best fits
        ASSERT_TRUE(generator.generateBestFits());

        if (i > 0)
        {
            //Compare best fits
            size_t bestFitDiff = TST_Generator::bestFitsDiff(generator.getBestFits(), lastBestFit);
            EXPECT_EQ(bestFitDiff, 0);
        }
        lastBestFit = generator.getBestFits();
    }
}

//Generates (with CUDA) photomosaic best fits using identical settings multiple times
//Expects all the best fits to be identical
TEST(Generator, CUDA_CIEDE2000_Detail_100)
{
    //Create photomosaic generators
    CUDAPhotomosaicGenerator generator;
    generator.setLibraryBatchSize(1000);

    //Set mode
    PhotomosaicGeneratorBase::Mode mode = PhotomosaicGeneratorBase::Mode::CIEDE2000;
    generator.setMode(mode);

    //Set cell group
    const size_t cellSize = 128;
    CellGroup cellGroup;
    cellGroup.setCellShape(CellShape(cellSize));
    cellGroup.setSizeSteps(0);
    cellGroup.setDetail(100);
    generator.setCellGroup(cellGroup);

    //Set repeat range and addition
    generator.setRepeat(0, 0);

    //Load and set image library
    ImageLibrary lib(128);
    lib.loadFromFile("E:/Desktop/MosaicMagnifique/MosaicMagnifique/Library/lib.mil");
    generator.setLibrary(lib.getImages());

    //Folder containing multiple images
    QDir imageFolder("E:/Desktop/MosaicMagnifique/NewLib");
    QStringList images = imageFolder.entryList(QDir::Filter::Files);

    //Select a random main image from list
    cv::Mat mainImage = cv::imread((imageFolder.path() + "/" +
                                    images.at((rand() * images.size()) / RAND_MAX)).toStdString());
    cv::resize(mainImage, mainImage, cv::Size(), 0.5, 0.5);
    generator.setMainImage(mainImage);

    GridUtility::MosaicBestFit lastBestFit;
    for (size_t i = 0; i < TST_Generator::ITERATIONS; ++i)
    {
        //Generate grid state
        const GridUtility::MosaicBestFit gridState =
            GridGenerator::getGridState(cellGroup, mainImage, mainImage.rows, mainImage.cols);
        generator.setGridState(gridState);

        //Generate best fits
        ASSERT_TRUE(generator.generateBestFits());

        if (i > 0)
        {
            //Compare best fits
            size_t bestFitDiff = TST_Generator::bestFitsDiff(generator.getBestFits(), lastBestFit);
            EXPECT_EQ(bestFitDiff, 0);
        }
        lastBestFit = generator.getBestFits();
    }
}

//Generates (with CUDA) photomosaic best fits using identical settings multiple times
//Expects all the best fits to be identical
TEST(Generator, CUDA_CIEDE2000_Detail_50)
{
    //Create photomosaic generators
    CUDAPhotomosaicGenerator generator;
    generator.setLibraryBatchSize(1000);

    //Set mode
    PhotomosaicGeneratorBase::Mode mode = PhotomosaicGeneratorBase::Mode::CIEDE2000;
    generator.setMode(mode);

    //Set cell group
    const size_t cellSize = 128;
    CellGroup cellGroup;
    cellGroup.setCellShape(CellShape(cellSize));
    cellGroup.setSizeSteps(0);
    cellGroup.setDetail(50);
    generator.setCellGroup(cellGroup);

    //Set repeat range and addition
    generator.setRepeat(0, 0);

    //Load and set image library
    ImageLibrary lib(128);
    lib.loadFromFile("E:/Desktop/MosaicMagnifique/MosaicMagnifique/Library/lib.mil");
    generator.setLibrary(lib.getImages());

    //Folder containing multiple images
    QDir imageFolder("E:/Desktop/MosaicMagnifique/NewLib");
    QStringList images = imageFolder.entryList(QDir::Filter::Files);

    //Select a random main image from list
    cv::Mat mainImage = cv::imread((imageFolder.path() + "/" +
                                    images.at((rand() * images.size()) / RAND_MAX)).toStdString());
    cv::resize(mainImage, mainImage, cv::Size(), 0.5, 0.5);
    generator.setMainImage(mainImage);

    GridUtility::MosaicBestFit lastBestFit;
    for (size_t i = 0; i < TST_Generator::ITERATIONS; ++i)
    {
        //Generate grid state
        const GridUtility::MosaicBestFit gridState =
            GridGenerator::getGridState(cellGroup, mainImage, mainImage.rows, mainImage.cols);
        generator.setGridState(gridState);

        //Generate best fits
        ASSERT_TRUE(generator.generateBestFits());

        if (i > 0)
        {
            //Compare best fits
            size_t bestFitDiff = TST_Generator::bestFitsDiff(generator.getBestFits(), lastBestFit);
            EXPECT_EQ(bestFitDiff, 0);
        }
        lastBestFit = generator.getBestFits();
    }
}

#endif


////////////////////////////////////////////////////////////////////////////////////////////////////
//******************************************** REPEATS *******************************************//
////////////////////////////////////////////////////////////////////////////////////////////////////

//Generates photomosaic best fits using identical settings multiple times
//Expects all the best fits to be identical
TEST(Generator, RGB_EUCLIDEAN_Detail_50_Repeats)
{
    //Create photomosaic generator
    CPUPhotomosaicGenerator generator;

    //Set mode
    PhotomosaicGeneratorBase::Mode mode = PhotomosaicGeneratorBase::Mode::RGB_EUCLIDEAN;
    generator.setMode(mode);

    //Set cell group
    const size_t cellSize = 128;
    CellGroup cellGroup;
    cellGroup.setCellShape(CellShape(cellSize));
    cellGroup.setSizeSteps(0);
    cellGroup.setDetail(50);
    generator.setCellGroup(cellGroup);

    //Set repeat range and addition
    generator.setRepeat(20, 10000);

    //Load and set image library
    ImageLibrary lib(128);
    lib.loadFromFile("E:/Desktop/MosaicMagnifique/MosaicMagnifique/Library/lib.mil");
    generator.setLibrary(lib.getImages());

    //Folder containing multiple images
    QDir imageFolder("E:/Desktop/MosaicMagnifique/NewLib");
    QStringList images = imageFolder.entryList(QDir::Filter::Files);

    //Select a random main image from list
    cv::Mat mainImage = cv::imread((imageFolder.path() + "/" +
                                    images.at((rand() * images.size()) / RAND_MAX)).toStdString());
    cv::resize(mainImage, mainImage, cv::Size(), 0.5, 0.5);
    generator.setMainImage(mainImage);

    GridUtility::MosaicBestFit lastBestFit;
    for (size_t i = 0; i < TST_Generator::ITERATIONS; ++i)
    {
        //Generate grid state
        const GridUtility::MosaicBestFit gridState =
            GridGenerator::getGridState(cellGroup, mainImage, mainImage.rows, mainImage.cols);
        generator.setGridState(gridState);

        //Generate best fits
        ASSERT_TRUE(generator.generateBestFits());

        if (i > 0)
        {
            //Compare best fits
            size_t bestFitDiff = TST_Generator::bestFitsDiff(generator.getBestFits(), lastBestFit);
            EXPECT_EQ(bestFitDiff, 0);
        }
        lastBestFit = generator.getBestFits();
    }
}

//Generates photomosaic best fits using identical settings multiple times
//Expects all the best fits to be identical
TEST(Generator, CIE76_Detail_50_Repeats)
{
    //Create photomosaic generator
    CPUPhotomosaicGenerator generator;

    //Set mode
    PhotomosaicGeneratorBase::Mode mode = PhotomosaicGeneratorBase::Mode::CIE76;
    generator.setMode(mode);

    //Set cell group
    const size_t cellSize = 128;
    CellGroup cellGroup;
    cellGroup.setCellShape(CellShape(cellSize));
    cellGroup.setSizeSteps(0);
    cellGroup.setDetail(50);
    generator.setCellGroup(cellGroup);

    //Set repeat range and addition
    generator.setRepeat(20, 10000);

    //Load and set image library
    ImageLibrary lib(128);
    lib.loadFromFile("E:/Desktop/MosaicMagnifique/MosaicMagnifique/Library/lib.mil");
    generator.setLibrary(lib.getImages());

    //Folder containing multiple images
    QDir imageFolder("E:/Desktop/MosaicMagnifique/NewLib");
    QStringList images = imageFolder.entryList(QDir::Filter::Files);

    //Select a random main image from list
    cv::Mat mainImage = cv::imread((imageFolder.path() + "/" +
                                    images.at((rand() * images.size()) / RAND_MAX)).toStdString());
    cv::resize(mainImage, mainImage, cv::Size(), 0.5, 0.5);
    generator.setMainImage(mainImage);

    GridUtility::MosaicBestFit lastBestFit;
    for (size_t i = 0; i < TST_Generator::ITERATIONS; ++i)
    {
        //Generate grid state
        const GridUtility::MosaicBestFit gridState =
            GridGenerator::getGridState(cellGroup, mainImage, mainImage.rows, mainImage.cols);
        generator.setGridState(gridState);

        //Generate best fits
        ASSERT_TRUE(generator.generateBestFits());

        if (i > 0)
        {
            //Compare best fits
            size_t bestFitDiff = TST_Generator::bestFitsDiff(generator.getBestFits(), lastBestFit);
            EXPECT_EQ(bestFitDiff, 0);
        }
        lastBestFit = generator.getBestFits();
    }
}

//Generates photomosaic best fits using identical settings multiple times
//Expects all the best fits to be identical
TEST(Generator, CIEDE2000_Detail_50_Repeats)
{
    //Create photomosaic generator
    CPUPhotomosaicGenerator generator;

    //Set mode
    PhotomosaicGeneratorBase::Mode mode = PhotomosaicGeneratorBase::Mode::CIEDE2000;
    generator.setMode(mode);

    //Set cell group
    const size_t cellSize = 128;
    CellGroup cellGroup;
    cellGroup.setCellShape(CellShape(cellSize));
    cellGroup.setSizeSteps(0);
    cellGroup.setDetail(50);
    generator.setCellGroup(cellGroup);

    //Set repeat range and addition
    generator.setRepeat(20, 10000);

    //Load and set image library
    ImageLibrary lib(128);
    lib.loadFromFile("E:/Desktop/MosaicMagnifique/MosaicMagnifique/Library/lib.mil");
    generator.setLibrary(lib.getImages());

    //Folder containing multiple images
    QDir imageFolder("E:/Desktop/MosaicMagnifique/NewLib");
    QStringList images = imageFolder.entryList(QDir::Filter::Files);

    //Select a random main image from list
    cv::Mat mainImage = cv::imread((imageFolder.path() + "/" +
                                    images.at((rand() * images.size()) / RAND_MAX)).toStdString());
    cv::resize(mainImage, mainImage, cv::Size(), 0.5, 0.5);
    generator.setMainImage(mainImage);

    GridUtility::MosaicBestFit lastBestFit;
    for (size_t i = 0; i < TST_Generator::ITERATIONS; ++i)
    {
        //Generate grid state
        const GridUtility::MosaicBestFit gridState =
            GridGenerator::getGridState(cellGroup, mainImage, mainImage.rows, mainImage.cols);
        generator.setGridState(gridState);

        //Generate best fits
        ASSERT_TRUE(generator.generateBestFits());

        if (i > 0)
        {
            //Compare best fits
            size_t bestFitDiff = TST_Generator::bestFitsDiff(generator.getBestFits(), lastBestFit);
            EXPECT_EQ(bestFitDiff, 0);
        }
        lastBestFit = generator.getBestFits();
    }
}


////////////////////////////////////////////////////////////////////////////////////////////////////
//****************************************** SIZE STEPS ******************************************//
////////////////////////////////////////////////////////////////////////////////////////////////////

//Generates photomosaic best fits using identical settings multiple times
//Expects all the best fits to be identical
TEST(Generator, RGB_EUCLIDEAN_Size_Steps)
{
    //Create photomosaic generator
    CPUPhotomosaicGenerator generator;

    //Set mode
    PhotomosaicGeneratorBase::Mode mode = PhotomosaicGeneratorBase::Mode::RGB_EUCLIDEAN;
    generator.setMode(mode);

    //Set cell group
    const size_t cellSize = 128;
    CellGroup cellGroup;
    cellGroup.setCellShape(CellShape(cellSize));
    cellGroup.setSizeSteps(1);
    cellGroup.setDetail(50);
    generator.setCellGroup(cellGroup);

    //Set repeat range and addition
    generator.setRepeat(0, 0);

    //Load and set image library
    ImageLibrary lib(128);
    lib.loadFromFile("E:/Desktop/MosaicMagnifique/MosaicMagnifique/Library/lib.mil");
    generator.setLibrary(lib.getImages());

    //Folder containing multiple images
    QDir imageFolder("E:/Desktop/MosaicMagnifique/NewLib");
    QStringList images = imageFolder.entryList(QDir::Filter::Files);

    //Select a random main image from list
    cv::Mat mainImage = cv::imread((imageFolder.path() + "/" +
                                    images.at((rand() * images.size()) / RAND_MAX)).toStdString());
    cv::resize(mainImage, mainImage, cv::Size(), 0.5, 0.5);
    generator.setMainImage(mainImage);

    GridUtility::MosaicBestFit lastBestFit;
    for (size_t i = 0; i < TST_Generator::ITERATIONS; ++i)
    {
        //Generate grid state
        const GridUtility::MosaicBestFit gridState =
            GridGenerator::getGridState(cellGroup, mainImage, mainImage.rows, mainImage.cols);
        generator.setGridState(gridState);

        //Generate best fits
        ASSERT_TRUE(generator.generateBestFits());

        if (i > 0)
        {
            //Compare best fits
            size_t bestFitDiff = TST_Generator::bestFitsDiff(generator.getBestFits(), lastBestFit);
            EXPECT_EQ(bestFitDiff, 0);
        }
        lastBestFit = generator.getBestFits();
    }
}

//Generates photomosaic best fits using identical settings multiple times
//Expects all the best fits to be identical
TEST(Generator, CIE76_Size_Steps)
{
    //Create photomosaic generator
    CPUPhotomosaicGenerator generator;

    //Set mode
    PhotomosaicGeneratorBase::Mode mode = PhotomosaicGeneratorBase::Mode::CIE76;
    generator.setMode(mode);

    //Set cell group
    const size_t cellSize = 128;
    CellGroup cellGroup;
    cellGroup.setCellShape(CellShape(cellSize));
    cellGroup.setSizeSteps(1);
    cellGroup.setDetail(50);
    generator.setCellGroup(cellGroup);

    //Set repeat range and addition
    generator.setRepeat(0, 0);

    //Load and set image library
    ImageLibrary lib(128);
    lib.loadFromFile("E:/Desktop/MosaicMagnifique/MosaicMagnifique/Library/lib.mil");
    generator.setLibrary(lib.getImages());

    //Folder containing multiple images
    QDir imageFolder("E:/Desktop/MosaicMagnifique/NewLib");
    QStringList images = imageFolder.entryList(QDir::Filter::Files);

    //Select a random main image from list
    cv::Mat mainImage = cv::imread((imageFolder.path() + "/" +
                                    images.at((rand() * images.size()) / RAND_MAX)).toStdString());
    cv::resize(mainImage, mainImage, cv::Size(), 0.5, 0.5);
    generator.setMainImage(mainImage);

    GridUtility::MosaicBestFit lastBestFit;
    for (size_t i = 0; i < TST_Generator::ITERATIONS; ++i)
    {
        //Generate grid state
        const GridUtility::MosaicBestFit gridState =
            GridGenerator::getGridState(cellGroup, mainImage, mainImage.rows, mainImage.cols);
        generator.setGridState(gridState);

        //Generate best fits
        ASSERT_TRUE(generator.generateBestFits());

        if (i > 0)
        {
            //Compare best fits
            size_t bestFitDiff = TST_Generator::bestFitsDiff(generator.getBestFits(), lastBestFit);
            EXPECT_EQ(bestFitDiff, 0);
        }
        lastBestFit = generator.getBestFits();
    }
}

//Generates photomosaic best fits using identical settings multiple times
//Expects all the best fits to be identical
TEST(Generator, CIEDE2000_Size_Steps)
{
    //Create photomosaic generator
    CPUPhotomosaicGenerator generator;

    //Set mode
    PhotomosaicGeneratorBase::Mode mode = PhotomosaicGeneratorBase::Mode::CIEDE2000;
    generator.setMode(mode);

    //Set cell group
    const size_t cellSize = 128;
    CellGroup cellGroup;
    cellGroup.setCellShape(CellShape(cellSize));
    cellGroup.setSizeSteps(0);
    cellGroup.setDetail(50);
    generator.setCellGroup(cellGroup);

    //Set repeat range and addition
    generator.setRepeat(0, 0);

    //Load and set image library
    ImageLibrary lib(128);
    lib.loadFromFile("E:/Desktop/MosaicMagnifique/MosaicMagnifique/Library/lib.mil");
    generator.setLibrary(lib.getImages());

    //Folder containing multiple images
    QDir imageFolder("E:/Desktop/MosaicMagnifique/NewLib");
    QStringList images = imageFolder.entryList(QDir::Filter::Files);

    //Select a random main image from list
    cv::Mat mainImage = cv::imread((imageFolder.path() + "/" +
                                    images.at((rand() * images.size()) / RAND_MAX)).toStdString());
    cv::resize(mainImage, mainImage, cv::Size(), 0.5, 0.5);
    generator.setMainImage(mainImage);

    GridUtility::MosaicBestFit lastBestFit;
    for (size_t i = 0; i < TST_Generator::ITERATIONS; ++i)
    {
        //Generate grid state
        const GridUtility::MosaicBestFit gridState =
            GridGenerator::getGridState(cellGroup, mainImage, mainImage.rows, mainImage.cols);
        generator.setGridState(gridState);

        //Generate best fits
        ASSERT_TRUE(generator.generateBestFits());

        if (i > 0)
        {
            //Compare best fits
            size_t bestFitDiff = TST_Generator::bestFitsDiff(generator.getBestFits(), lastBestFit);
            EXPECT_EQ(bestFitDiff, 0);
        }
        lastBestFit = generator.getBestFits();
    }
}


#ifdef CUDA
////////////////////////////////////////////////////////////////////////////////////////////////////
//***************************************** CUDA REPEATS *****************************************//
////////////////////////////////////////////////////////////////////////////////////////////////////

//Generates (with CUDA) photomosaic best fits using identical settings multiple times
//Expects all the best fits to be identical
TEST(Generator, CUDA_RGB_EUCLIDEAN_Repeats)
{
    //Create photomosaic generators
    CUDAPhotomosaicGenerator generator;
    generator.setLibraryBatchSize(1000);

    //Set mode
    PhotomosaicGeneratorBase::Mode mode = PhotomosaicGeneratorBase::Mode::RGB_EUCLIDEAN;
    generator.setMode(mode);

    //Set cell group
    const size_t cellSize = 128;
    CellGroup cellGroup;
    cellGroup.setCellShape(CellShape(cellSize));
    cellGroup.setSizeSteps(0);
    cellGroup.setDetail(50);
    generator.setCellGroup(cellGroup);

    //Set repeat range and addition
    generator.setRepeat(20, 10000);

    //Load and set image library
    ImageLibrary lib(128);
    lib.loadFromFile("E:/Desktop/MosaicMagnifique/MosaicMagnifique/Library/lib.mil");
    generator.setLibrary(lib.getImages());

    //Folder containing multiple images
    QDir imageFolder("E:/Desktop/MosaicMagnifique/NewLib");
    QStringList images = imageFolder.entryList(QDir::Filter::Files);

    //Select a random main image from list
    cv::Mat mainImage = cv::imread((imageFolder.path() + "/" +
                                    images.at((rand() * images.size()) / RAND_MAX)).toStdString());
    cv::resize(mainImage, mainImage, cv::Size(), 0.5, 0.5);
    generator.setMainImage(mainImage);

    GridUtility::MosaicBestFit lastBestFit;
    for (size_t i = 0; i < TST_Generator::ITERATIONS; ++i)
    {
        //Generate grid state
        const GridUtility::MosaicBestFit gridState =
            GridGenerator::getGridState(cellGroup, mainImage, mainImage.rows, mainImage.cols);
        generator.setGridState(gridState);

        //Generate best fits
        ASSERT_TRUE(generator.generateBestFits());

        if (i > 0)
        {
            //Compare best fits
            size_t bestFitDiff = TST_Generator::bestFitsDiff(generator.getBestFits(), lastBestFit);
            EXPECT_EQ(bestFitDiff, 0);
        }
        lastBestFit = generator.getBestFits();
    }
}

//Generates (with CUDA) photomosaic best fits using identical settings multiple times
//Expects all the best fits to be identical
TEST(Generator, CUDA_CIE76_Repeats)
{
    //Create photomosaic generators
    CUDAPhotomosaicGenerator generator;
    generator.setLibraryBatchSize(1000);

    //Set mode
    PhotomosaicGeneratorBase::Mode mode = PhotomosaicGeneratorBase::Mode::CIE76;
    generator.setMode(mode);

    //Set cell group
    const size_t cellSize = 128;
    CellGroup cellGroup;
    cellGroup.setCellShape(CellShape(cellSize));
    cellGroup.setSizeSteps(0);
    cellGroup.setDetail(50);
    generator.setCellGroup(cellGroup);

    //Set repeat range and addition
    generator.setRepeat(20, 10000);

    //Load and set image library
    ImageLibrary lib(128);
    lib.loadFromFile("E:/Desktop/MosaicMagnifique/MosaicMagnifique/Library/lib.mil");
    generator.setLibrary(lib.getImages());

    //Folder containing multiple images
    QDir imageFolder("E:/Desktop/MosaicMagnifique/NewLib");
    QStringList images = imageFolder.entryList(QDir::Filter::Files);

    //Select a random main image from list
    cv::Mat mainImage = cv::imread((imageFolder.path() + "/" +
                                    images.at((rand() * images.size()) / RAND_MAX)).toStdString());
    cv::resize(mainImage, mainImage, cv::Size(), 0.5, 0.5);
    generator.setMainImage(mainImage);

    GridUtility::MosaicBestFit lastBestFit;
    for (size_t i = 0; i < TST_Generator::ITERATIONS; ++i)
    {
        //Generate grid state
        const GridUtility::MosaicBestFit gridState =
            GridGenerator::getGridState(cellGroup, mainImage, mainImage.rows, mainImage.cols);
        generator.setGridState(gridState);

        //Generate best fits
        ASSERT_TRUE(generator.generateBestFits());

        if (i > 0)
        {
            //Compare best fits
            size_t bestFitDiff = TST_Generator::bestFitsDiff(generator.getBestFits(), lastBestFit);
            EXPECT_EQ(bestFitDiff, 0);
        }
        lastBestFit = generator.getBestFits();
    }
}

//Generates (with CUDA) photomosaic best fits using identical settings multiple times
//Expects all the best fits to be identical
TEST(Generator, CUDA_CIEDE2000_Repeats)
{
    //Create photomosaic generators
    CUDAPhotomosaicGenerator generator;
    generator.setLibraryBatchSize(1000);

    //Set mode
    PhotomosaicGeneratorBase::Mode mode = PhotomosaicGeneratorBase::Mode::CIEDE2000;
    generator.setMode(mode);

    //Set cell group
    const size_t cellSize = 128;
    CellGroup cellGroup;
    cellGroup.setCellShape(CellShape(cellSize));
    cellGroup.setSizeSteps(0);
    cellGroup.setDetail(50);
    generator.setCellGroup(cellGroup);

    //Set repeat range and addition
    generator.setRepeat(20, 10000);

    //Load and set image library
    ImageLibrary lib(128);
    lib.loadFromFile("E:/Desktop/MosaicMagnifique/MosaicMagnifique/Library/lib.mil");
    generator.setLibrary(lib.getImages());

    //Folder containing multiple images
    QDir imageFolder("E:/Desktop/MosaicMagnifique/NewLib");
    QStringList images = imageFolder.entryList(QDir::Filter::Files);

    //Select a random main image from list
    cv::Mat mainImage = cv::imread((imageFolder.path() + "/" +
                                    images.at((rand() * images.size()) / RAND_MAX)).toStdString());
    cv::resize(mainImage, mainImage, cv::Size(), 0.5, 0.5);
    generator.setMainImage(mainImage);

    GridUtility::MosaicBestFit lastBestFit;
    for (size_t i = 0; i < TST_Generator::ITERATIONS; ++i)
    {
        //Generate grid state
        const GridUtility::MosaicBestFit gridState =
            GridGenerator::getGridState(cellGroup, mainImage, mainImage.rows, mainImage.cols);
        generator.setGridState(gridState);

        //Generate best fits
        ASSERT_TRUE(generator.generateBestFits());

        if (i > 0)
        {
            //Compare best fits
            size_t bestFitDiff = TST_Generator::bestFitsDiff(generator.getBestFits(), lastBestFit);
            EXPECT_EQ(bestFitDiff, 0);
        }
        lastBestFit = generator.getBestFits();
    }
}


////////////////////////////////////////////////////////////////////////////////////////////////////
//*************************************** CUDA SIZE STEPS ****************************************//
////////////////////////////////////////////////////////////////////////////////////////////////////

//Generates photomosaic best fits using identical settings multiple times
//Expects all the best fits to be identical
TEST(Generator, CUDA_RGB_EUCLIDEAN_Size_Steps)
{
    //Create photomosaic generator
    CUDAPhotomosaicGenerator generator;
    generator.setLibraryBatchSize(1000);

    //Set mode
    PhotomosaicGeneratorBase::Mode mode = PhotomosaicGeneratorBase::Mode::RGB_EUCLIDEAN;
    generator.setMode(mode);

    //Set cell group
    const size_t cellSize = 128;
    CellGroup cellGroup;
    cellGroup.setCellShape(CellShape(cellSize));
    cellGroup.setSizeSteps(1);
    cellGroup.setDetail(50);
    generator.setCellGroup(cellGroup);

    //Set repeat range and addition
    generator.setRepeat(0, 0);

    //Load and set image library
    ImageLibrary lib(128);
    lib.loadFromFile("E:/Desktop/MosaicMagnifique/MosaicMagnifique/Library/lib.mil");
    generator.setLibrary(lib.getImages());

    //Folder containing multiple images
    QDir imageFolder("E:/Desktop/MosaicMagnifique/NewLib");
    QStringList images = imageFolder.entryList(QDir::Filter::Files);

    //Select a random main image from list
    cv::Mat mainImage = cv::imread((imageFolder.path() + "/" +
                                    images.at((rand() * images.size()) / RAND_MAX)).toStdString());
    cv::resize(mainImage, mainImage, cv::Size(), 0.5, 0.5);
    generator.setMainImage(mainImage);

    GridUtility::MosaicBestFit lastBestFit;
    for (size_t i = 0; i < TST_Generator::ITERATIONS; ++i)
    {
        //Generate grid state
        const GridUtility::MosaicBestFit gridState =
            GridGenerator::getGridState(cellGroup, mainImage, mainImage.rows, mainImage.cols);
        generator.setGridState(gridState);

        //Generate best fits
        ASSERT_TRUE(generator.generateBestFits());

        if (i > 0)
        {
            //Compare best fits
            size_t bestFitDiff = TST_Generator::bestFitsDiff(generator.getBestFits(), lastBestFit);
            EXPECT_EQ(bestFitDiff, 0);
        }
        lastBestFit = generator.getBestFits();
    }
}

//Generates photomosaic best fits using identical settings multiple times
//Expects all the best fits to be identical
TEST(Generator, CUDA_CIE76_Size_Steps)
{
    //Create photomosaic generator
    CUDAPhotomosaicGenerator generator;
    generator.setLibraryBatchSize(1000);

    //Set mode
    PhotomosaicGeneratorBase::Mode mode = PhotomosaicGeneratorBase::Mode::CIE76;
    generator.setMode(mode);

    //Set cell group
    const size_t cellSize = 128;
    CellGroup cellGroup;
    cellGroup.setCellShape(CellShape(cellSize));
    cellGroup.setSizeSteps(1);
    cellGroup.setDetail(50);
    generator.setCellGroup(cellGroup);

    //Set repeat range and addition
    generator.setRepeat(0, 0);

    //Load and set image library
    ImageLibrary lib(128);
    lib.loadFromFile("E:/Desktop/MosaicMagnifique/MosaicMagnifique/Library/lib.mil");
    generator.setLibrary(lib.getImages());

    //Folder containing multiple images
    QDir imageFolder("E:/Desktop/MosaicMagnifique/NewLib");
    QStringList images = imageFolder.entryList(QDir::Filter::Files);

    //Select a random main image from list
    cv::Mat mainImage = cv::imread((imageFolder.path() + "/" +
                                    images.at((rand() * images.size()) / RAND_MAX)).toStdString());
    cv::resize(mainImage, mainImage, cv::Size(), 0.5, 0.5);
    generator.setMainImage(mainImage);

    GridUtility::MosaicBestFit lastBestFit;
    for (size_t i = 0; i < TST_Generator::ITERATIONS; ++i)
    {
        //Generate grid state
        const GridUtility::MosaicBestFit gridState =
            GridGenerator::getGridState(cellGroup, mainImage, mainImage.rows, mainImage.cols);
        generator.setGridState(gridState);

        //Generate best fits
        ASSERT_TRUE(generator.generateBestFits());

        if (i > 0)
        {
            //Compare best fits
            size_t bestFitDiff = TST_Generator::bestFitsDiff(generator.getBestFits(), lastBestFit);
            EXPECT_EQ(bestFitDiff, 0);
        }
        lastBestFit = generator.getBestFits();
    }
}

//Generates photomosaic best fits using identical settings multiple times
//Expects all the best fits to be identical
TEST(Generator, CUDA_CIEDE2000_Size_Steps)
{
    //Create photomosaic generator
    CUDAPhotomosaicGenerator generator;
    generator.setLibraryBatchSize(1000);

    //Set mode
    PhotomosaicGeneratorBase::Mode mode = PhotomosaicGeneratorBase::Mode::CIEDE2000;
    generator.setMode(mode);

    //Set cell group
    const size_t cellSize = 128;
    CellGroup cellGroup;
    cellGroup.setCellShape(CellShape(cellSize));
    cellGroup.setSizeSteps(1);
    cellGroup.setDetail(50);
    generator.setCellGroup(cellGroup);

    //Set repeat range and addition
    generator.setRepeat(0, 0);

    //Load and set image library
    ImageLibrary lib(128);
    lib.loadFromFile("E:/Desktop/MosaicMagnifique/MosaicMagnifique/Library/lib.mil");
    generator.setLibrary(lib.getImages());

    //Folder containing multiple images
    QDir imageFolder("E:/Desktop/MosaicMagnifique/NewLib");
    QStringList images = imageFolder.entryList(QDir::Filter::Files);

    //Select a random main image from list
    cv::Mat mainImage = cv::imread((imageFolder.path() + "/" +
                                    images.at((rand() * images.size()) / RAND_MAX)).toStdString());
    cv::resize(mainImage, mainImage, cv::Size(), 0.5, 0.5);
    generator.setMainImage(mainImage);

    GridUtility::MosaicBestFit lastBestFit;
    for (size_t i = 0; i < TST_Generator::ITERATIONS; ++i)
    {
        //Generate grid state
        const GridUtility::MosaicBestFit gridState =
            GridGenerator::getGridState(cellGroup, mainImage, mainImage.rows, mainImage.cols);
        generator.setGridState(gridState);

        //Generate best fits
        ASSERT_TRUE(generator.generateBestFits());

        if (i > 0)
        {
            //Compare best fits
            size_t bestFitDiff = TST_Generator::bestFitsDiff(generator.getBestFits(), lastBestFit);
            EXPECT_EQ(bestFitDiff, 0);
        }
        lastBestFit = generator.getBestFits();
    }
}


////////////////////////////////////////////////////////////////////////////////////////////////////
//************************************** DETAIL CPU vs CUDA **************************************//
////////////////////////////////////////////////////////////////////////////////////////////////////

//Generates photomosaic best fits using identical settings with and without CUDA
//Expects the best fits to be identical
//Tests with multiple main images
TEST(Generator, RGB_EUCLIDEAN_Detail_100_vs_CUDA)
{
    //Create folder for saving photomosaics
    QDir resultFolder(QDir::currentPath() + "/testcases/generator");
    if (!resultFolder.exists())
        resultFolder.mkpath(".");

    //Create photomosaic generators
    CPUPhotomosaicGenerator generator;
    CUDAPhotomosaicGenerator generatorCUDA;
    generatorCUDA.setLibraryBatchSize(1000);

    //Set mode
    PhotomosaicGeneratorBase::Mode mode = PhotomosaicGeneratorBase::Mode::RGB_EUCLIDEAN;
    generator.setMode(mode);
    generatorCUDA.setMode(mode);

    //Set cell group
    const size_t cellSize = 128;
    CellGroup cellGroup;
    cellGroup.setCellShape(CellShape(cellSize));
    cellGroup.setSizeSteps(0);
    cellGroup.setDetail(100);
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
    for (auto it = images.cbegin();
         it != images.cend() && it != images.cbegin() + TST_Generator::ITERATIONS; ++it)
    {
        //Load and set main image
        cv::Mat mainImage = cv::imread((imageFolder.path() + "/" + *it).toStdString());
        cv::resize(mainImage, mainImage, cv::Size(), 0.5, 0.5);
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

        EXPECT_EQ(bestFitDiff, 0) << it->toStdString();

        //Best fits different, save results
        if (bestFitDiff != 0)
        {
            //Build photomosaics
            cv::Mat result = generator.buildPhotomosaic();
            cv::Mat resultCUDA = generatorCUDA.buildPhotomosaic();

            //Save photomosaics
            cv::imwrite((resultFolder.path() + "/" + *it).toStdString(), result);
            cv::imwrite((resultFolder.path() + "/CUDA-" + *it).toStdString(), resultCUDA);
        }
    }
}

//Generates photomosaic best fits using identical settings with and without CUDA
//Expects the best fits to be identical
//Tests with multiple main images
TEST(Generator, RGB_EUCLIDEAN_Detail_50_vs_CUDA)
{
    //Create folder for saving photomosaics
    QDir resultFolder(QDir::currentPath() + "/testcases/generator");
    if (!resultFolder.exists())
        resultFolder.mkpath(".");

    //Create photomosaic generators
    CPUPhotomosaicGenerator generator;
    CUDAPhotomosaicGenerator generatorCUDA;
    generatorCUDA.setLibraryBatchSize(1000);

    //Set mode
    PhotomosaicGeneratorBase::Mode mode = PhotomosaicGeneratorBase::Mode::RGB_EUCLIDEAN;
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
    for (auto it = images.cbegin();
         it != images.cend() && it != images.cbegin() + TST_Generator::ITERATIONS; ++it)
    {
        //Load and set main image
        cv::Mat mainImage = cv::imread((imageFolder.path() + "/" + *it).toStdString());
        cv::resize(mainImage, mainImage, cv::Size(), 0.5, 0.5);
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

        EXPECT_EQ(bestFitDiff, 0) << it->toStdString();

        //Best fits different, save results
        if (bestFitDiff != 0)
        {
            //Build photomosaics
            cv::Mat result = generator.buildPhotomosaic();
            cv::Mat resultCUDA = generatorCUDA.buildPhotomosaic();

            //Save photomosaics
            cv::imwrite((resultFolder.path() + "/" + *it).toStdString(), result);
            cv::imwrite((resultFolder.path() + "/CUDA-" + *it).toStdString(), resultCUDA);
        }
    }
}

//Generates photomosaic best fits using identical settings with and without CUDA
//Expects the best fits to be identical
//Tests with multiple main images
TEST(Generator, CIE76_Detail_100_vs_CUDA)
{
    //Create folder for saving photomosaics
    QDir resultFolder(QDir::currentPath() + "/testcases/generator");
    if (!resultFolder.exists())
        resultFolder.mkpath(".");

    //Create photomosaic generators
    CPUPhotomosaicGenerator generator;
    CUDAPhotomosaicGenerator generatorCUDA;
    generatorCUDA.setLibraryBatchSize(1000);

    //Set mode
    PhotomosaicGeneratorBase::Mode mode = PhotomosaicGeneratorBase::Mode::CIE76;
    generator.setMode(mode);
    generatorCUDA.setMode(mode);

    //Set cell group
    const size_t cellSize = 128;
    CellGroup cellGroup;
    cellGroup.setCellShape(CellShape(cellSize));
    cellGroup.setSizeSteps(0);
    cellGroup.setDetail(100);
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
    for (auto it = images.cbegin();
         it != images.cend() && it != images.cbegin() + TST_Generator::ITERATIONS; ++it)
    {
        //Load and set main image
        cv::Mat mainImage = cv::imread((imageFolder.path() + "/" + *it).toStdString());
        cv::resize(mainImage, mainImage, cv::Size(), 0.5, 0.5);
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

        EXPECT_EQ(bestFitDiff, 0) << it->toStdString();

        //Best fits different, save results
        if (bestFitDiff != 0)
        {
            //Build photomosaics
            cv::Mat result = generator.buildPhotomosaic();
            cv::Mat resultCUDA = generatorCUDA.buildPhotomosaic();

            //Save photomosaics
            cv::imwrite((resultFolder.path() + "/" + *it).toStdString(), result);
            cv::imwrite((resultFolder.path() + "/CUDA-" + *it).toStdString(), resultCUDA);
        }
    }
}

//Generates photomosaic best fits using identical settings with and without CUDA
//Expects the best fits to be identical
//Tests with multiple main images
TEST(Generator, CIE76_Detail_50_vs_CUDA)
{
    //Create folder for saving photomosaics
    QDir resultFolder(QDir::currentPath() + "/testcases/generator");
    if (!resultFolder.exists())
        resultFolder.mkpath(".");

    //Create photomosaic generators
    CPUPhotomosaicGenerator generator;
    CUDAPhotomosaicGenerator generatorCUDA;
    generatorCUDA.setLibraryBatchSize(1000);

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
    for (auto it = images.cbegin();
         it != images.cend() && it != images.cbegin() + TST_Generator::ITERATIONS; ++it)
    {
        //Load and set main image
        cv::Mat mainImage = cv::imread((imageFolder.path() + "/" + *it).toStdString());
        cv::resize(mainImage, mainImage, cv::Size(), 0.5, 0.5);
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

        EXPECT_EQ(bestFitDiff, 0) << it->toStdString();

        //Best fits different, save results
        if (bestFitDiff != 0)
        {
            //Build photomosaics
            cv::Mat result = generator.buildPhotomosaic();
            cv::Mat resultCUDA = generatorCUDA.buildPhotomosaic();

            //Save photomosaics
            cv::imwrite((resultFolder.path() + "/" + *it).toStdString(), result);
            cv::imwrite((resultFolder.path() + "/CUDA-" + *it).toStdString(), resultCUDA);
        }
    }
}

//Generates photomosaic best fits using identical settings with and without CUDA
//Expects the best fits to be identical
//Tests with multiple main images
TEST(Generator, CIEDE2000_Detail_100_vs_CUDA)
{
    //Create folder for saving photomosaics
    QDir resultFolder(QDir::currentPath() + "/testcases/generator");
    if (!resultFolder.exists())
        resultFolder.mkpath(".");

    //Create photomosaic generators
    CPUPhotomosaicGenerator generator;
    CUDAPhotomosaicGenerator generatorCUDA;
    generatorCUDA.setLibraryBatchSize(1000);

    //Set mode
    PhotomosaicGeneratorBase::Mode mode = PhotomosaicGeneratorBase::Mode::CIEDE2000;
    generator.setMode(mode);
    generatorCUDA.setMode(mode);

    //Set cell group
    const size_t cellSize = 128;
    CellGroup cellGroup;
    cellGroup.setCellShape(CellShape(cellSize));
    cellGroup.setSizeSteps(0);
    cellGroup.setDetail(100);
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
    for (auto it = images.cbegin();
         it != images.cend() && it != images.cbegin() + TST_Generator::ITERATIONS; ++it)
    {
        //Load and set main image
        cv::Mat mainImage = cv::imread((imageFolder.path() + "/" + *it).toStdString());
        cv::resize(mainImage, mainImage, cv::Size(), 0.5, 0.5);
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

        EXPECT_EQ(bestFitDiff, 0) << it->toStdString();

        //Best fits different, save results
        if (bestFitDiff != 0)
        {
            //Build photomosaics
            cv::Mat result = generator.buildPhotomosaic();
            cv::Mat resultCUDA = generatorCUDA.buildPhotomosaic();

            //Save photomosaics
            cv::imwrite((resultFolder.path() + "/" + *it).toStdString(), result);
            cv::imwrite((resultFolder.path() + "/CUDA-" + *it).toStdString(), resultCUDA);
        }
    }
}

//Generates photomosaic best fits using identical settings with and without CUDA
//Expects the best fits to be identical
//Tests with multiple main images
TEST(Generator, CIEDE2000_Detail_50_vs_CUDA)
{
    //Create folder for saving photomosaics
    QDir resultFolder(QDir::currentPath() + "/testcases/generator");
    if (!resultFolder.exists())
        resultFolder.mkpath(".");

    //Create photomosaic generators
    CPUPhotomosaicGenerator generator;
    CUDAPhotomosaicGenerator generatorCUDA;
    generatorCUDA.setLibraryBatchSize(1000);

    //Set mode
    PhotomosaicGeneratorBase::Mode mode = PhotomosaicGeneratorBase::Mode::CIEDE2000;
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
    for (auto it = images.cbegin();
         it != images.cend() && it != images.cbegin() + TST_Generator::ITERATIONS; ++it)
    {
        //Load and set main image
        cv::Mat mainImage = cv::imread((imageFolder.path() + "/" + *it).toStdString());
        cv::resize(mainImage, mainImage, cv::Size(), 0.5, 0.5);
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

        EXPECT_EQ(bestFitDiff, 0) << it->toStdString();

        //Best fits different, save results
        if (bestFitDiff != 0)
        {
            //Build photomosaics
            cv::Mat result = generator.buildPhotomosaic();
            cv::Mat resultCUDA = generatorCUDA.buildPhotomosaic();

            //Save photomosaics
            cv::imwrite((resultFolder.path() + "/" + *it).toStdString(), result);
            cv::imwrite((resultFolder.path() + "/CUDA-" + *it).toStdString(), resultCUDA);
        }
    }
}


////////////////////////////////////////////////////////////////////////////////////////////////////
//************************************* REPEATS CPU vs CUDA **************************************//
////////////////////////////////////////////////////////////////////////////////////////////////////

//Generates photomosaic best fits using identical settings with and without CUDA
//Expects the best fits to be identical
//Tests with multiple main images
TEST(Generator, RGB_EUCLIDEAN_Repeats_vs_CUDA)
{
    //Create folder for saving photomosaics
    QDir resultFolder(QDir::currentPath() + "/testcases/generator");
    if (!resultFolder.exists())
        resultFolder.mkpath(".");

    //Create photomosaic generators
    CPUPhotomosaicGenerator generator;
    CUDAPhotomosaicGenerator generatorCUDA;
    generatorCUDA.setLibraryBatchSize(1000);

    //Set mode
    PhotomosaicGeneratorBase::Mode mode = PhotomosaicGeneratorBase::Mode::RGB_EUCLIDEAN;
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
    generator.setRepeat(20, 10000);
    generatorCUDA.setRepeat(20, 10000);

    //Load and set image library
    ImageLibrary lib(128);
    lib.loadFromFile("E:/Desktop/MosaicMagnifique/MosaicMagnifique/Library/lib.mil");
    generator.setLibrary(lib.getImages());
    generatorCUDA.setLibrary(lib.getImages());

    //Folder containing multiple images
    QDir imageFolder("E:/Desktop/MosaicMagnifique/NewLib");
    QStringList images = imageFolder.entryList(QDir::Filter::Files);

    //Iterate over all images
    for (auto it = images.cbegin();
         it != images.cend() && it != images.cbegin() + TST_Generator::ITERATIONS; ++it)
    {
        //Load and set main image
        cv::Mat mainImage = cv::imread((imageFolder.path() + "/" + *it).toStdString());
        cv::resize(mainImage, mainImage, cv::Size(), 0.5, 0.5);
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

        EXPECT_EQ(bestFitDiff, 0) << it->toStdString();

        //Best fits different, save results
        if (bestFitDiff != 0)
        {
            //Build photomosaics
            cv::Mat result = generator.buildPhotomosaic();
            cv::Mat resultCUDA = generatorCUDA.buildPhotomosaic();

            //Save photomosaics
            cv::imwrite((resultFolder.path() + "/" + *it).toStdString(), result);
            cv::imwrite((resultFolder.path() + "/CUDA-" + *it).toStdString(), resultCUDA);
        }
    }
}

//Generates photomosaic best fits using identical settings with and without CUDA
//Expects the best fits to be identical
//Tests with multiple main images
TEST(Generator, CIE76_Repeats_vs_CUDA)
{
    //Create folder for saving photomosaics
    QDir resultFolder(QDir::currentPath() + "/testcases/generator");
    if (!resultFolder.exists())
        resultFolder.mkpath(".");

    //Create photomosaic generators
    CPUPhotomosaicGenerator generator;
    CUDAPhotomosaicGenerator generatorCUDA;
    generatorCUDA.setLibraryBatchSize(1000);

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
    generator.setRepeat(20, 10000);
    generatorCUDA.setRepeat(20, 10000);

    //Load and set image library
    ImageLibrary lib(128);
    lib.loadFromFile("E:/Desktop/MosaicMagnifique/MosaicMagnifique/Library/lib.mil");
    generator.setLibrary(lib.getImages());
    generatorCUDA.setLibrary(lib.getImages());

    //Folder containing multiple images
    QDir imageFolder("E:/Desktop/MosaicMagnifique/NewLib");
    QStringList images = imageFolder.entryList(QDir::Filter::Files);

    //Iterate over all images
    for (auto it = images.cbegin();
         it != images.cend() && it != images.cbegin() + TST_Generator::ITERATIONS; ++it)
    {
        //Load and set main image
        cv::Mat mainImage = cv::imread((imageFolder.path() + "/" + *it).toStdString());
        cv::resize(mainImage, mainImage, cv::Size(), 0.5, 0.5);
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

        EXPECT_EQ(bestFitDiff, 0) << it->toStdString();

        //Best fits different, save results
        if (bestFitDiff != 0)
        {
            //Build photomosaics
            cv::Mat result = generator.buildPhotomosaic();
            cv::Mat resultCUDA = generatorCUDA.buildPhotomosaic();

            //Save photomosaics
            cv::imwrite((resultFolder.path() + "/" + *it).toStdString(), result);
            cv::imwrite((resultFolder.path() + "/CUDA-" + *it).toStdString(), resultCUDA);
        }
    }
}

//Generates photomosaic best fits using identical settings with and without CUDA
//Expects the best fits to be identical
//Tests with multiple main images
TEST(Generator, CIEDE2000_Repeats_vs_CUDA)
{
    //Create folder for saving photomosaics
    QDir resultFolder(QDir::currentPath() + "/testcases/generator");
    if (!resultFolder.exists())
        resultFolder.mkpath(".");

    //Create photomosaic generators
    CPUPhotomosaicGenerator generator;
    CUDAPhotomosaicGenerator generatorCUDA;
    generatorCUDA.setLibraryBatchSize(1000);

    //Set mode
    PhotomosaicGeneratorBase::Mode mode = PhotomosaicGeneratorBase::Mode::CIEDE2000;
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
    generator.setRepeat(20, 10000);
    generatorCUDA.setRepeat(20, 10000);

    //Load and set image library
    ImageLibrary lib(128);
    lib.loadFromFile("E:/Desktop/MosaicMagnifique/MosaicMagnifique/Library/lib.mil");
    generator.setLibrary(lib.getImages());
    generatorCUDA.setLibrary(lib.getImages());

    //Folder containing multiple images
    QDir imageFolder("E:/Desktop/MosaicMagnifique/NewLib");
    QStringList images = imageFolder.entryList(QDir::Filter::Files);

    //Iterate over all images
    for (auto it = images.cbegin();
         it != images.cend() && it != images.cbegin() + TST_Generator::ITERATIONS; ++it)
    {
        //Load and set main image
        cv::Mat mainImage = cv::imread((imageFolder.path() + "/" + *it).toStdString());
        cv::resize(mainImage, mainImage, cv::Size(), 0.5, 0.5);
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

        EXPECT_EQ(bestFitDiff, 0) << it->toStdString();

        //Best fits different, save results
        if (bestFitDiff != 0)
        {
            //Build photomosaics
            cv::Mat result = generator.buildPhotomosaic();
            cv::Mat resultCUDA = generatorCUDA.buildPhotomosaic();

            //Save photomosaics
            cv::imwrite((resultFolder.path() + "/" + *it).toStdString(), result);
            cv::imwrite((resultFolder.path() + "/CUDA-" + *it).toStdString(), resultCUDA);
        }
    }
}


////////////////////////////////////////////////////////////////////////////////////////////////////
//************************************ SIZE STEPS CPU vs CUDA ************************************//
////////////////////////////////////////////////////////////////////////////////////////////////////

//Generates photomosaic best fits using identical settings with and without CUDA
//Expects the best fits to be identical
//Tests with multiple main images
TEST(Generator, RGB_EUCLIDEAN_Size_Steps_vs_CUDA)
{
    //Create folder for saving photomosaics
    QDir resultFolder(QDir::currentPath() + "/testcases/generator");
    if (!resultFolder.exists())
        resultFolder.mkpath(".");

    //Create photomosaic generators
    CPUPhotomosaicGenerator generator;
    CUDAPhotomosaicGenerator generatorCUDA;
    generatorCUDA.setLibraryBatchSize(1000);

    //Set mode
    PhotomosaicGeneratorBase::Mode mode = PhotomosaicGeneratorBase::Mode::RGB_EUCLIDEAN;
    generator.setMode(mode);
    generatorCUDA.setMode(mode);

    //Set cell group
    const size_t cellSize = 128;
    CellGroup cellGroup;
    cellGroup.setCellShape(CellShape(cellSize));
    cellGroup.setSizeSteps(1);
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
    for (auto it = images.cbegin();
         it != images.cend() && it != images.cbegin() + TST_Generator::ITERATIONS; ++it)
    {
        //Load and set main image
        cv::Mat mainImage = cv::imread((imageFolder.path() + "/" + *it).toStdString());
        cv::resize(mainImage, mainImage, cv::Size(), 0.5, 0.5);
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

        EXPECT_EQ(bestFitDiff, 0) << it->toStdString();

        //Best fits different, save results
        if (bestFitDiff != 0)
        {
            //Build photomosaics
            cv::Mat result = generator.buildPhotomosaic();
            cv::Mat resultCUDA = generatorCUDA.buildPhotomosaic();

            //Save photomosaics
            cv::imwrite((resultFolder.path() + "/" + *it).toStdString(), result);
            cv::imwrite((resultFolder.path() + "/CUDA-" + *it).toStdString(), resultCUDA);
        }
    }
}

//Generates photomosaic best fits using identical settings with and without CUDA
//Expects the best fits to be identical
//Tests with multiple main images
TEST(Generator, CIE76_Size_Steps_vs_CUDA)
{
    //Create folder for saving photomosaics
    QDir resultFolder(QDir::currentPath() + "/testcases/generator");
    if (!resultFolder.exists())
        resultFolder.mkpath(".");

    //Create photomosaic generators
    CPUPhotomosaicGenerator generator;
    CUDAPhotomosaicGenerator generatorCUDA;
    generatorCUDA.setLibraryBatchSize(1000);

    //Set mode
    PhotomosaicGeneratorBase::Mode mode = PhotomosaicGeneratorBase::Mode::CIE76;
    generator.setMode(mode);
    generatorCUDA.setMode(mode);

    //Set cell group
    const size_t cellSize = 128;
    CellGroup cellGroup;
    cellGroup.setCellShape(CellShape(cellSize));
    cellGroup.setSizeSteps(1);
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
    for (auto it = images.cbegin();
         it != images.cend() && it != images.cbegin() + TST_Generator::ITERATIONS; ++it)
    {
        //Load and set main image
        cv::Mat mainImage = cv::imread((imageFolder.path() + "/" + *it).toStdString());
        cv::resize(mainImage, mainImage, cv::Size(), 0.5, 0.5);
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

        EXPECT_EQ(bestFitDiff, 0) << it->toStdString();

        //Best fits different, save results
        if (bestFitDiff != 0)
        {
            //Build photomosaics
            cv::Mat result = generator.buildPhotomosaic();
            cv::Mat resultCUDA = generatorCUDA.buildPhotomosaic();

            //Save photomosaics
            cv::imwrite((resultFolder.path() + "/" + *it).toStdString(), result);
            cv::imwrite((resultFolder.path() + "/CUDA-" + *it).toStdString(), resultCUDA);
        }
    }
}

//Generates photomosaic best fits using identical settings with and without CUDA
//Expects the best fits to be identical
//Tests with multiple main images
TEST(Generator, CIEDE2000_Size_Steps_vs_CUDA)
{
    //Create folder for saving photomosaics
    QDir resultFolder(QDir::currentPath() + "/testcases/generator");
    if (!resultFolder.exists())
        resultFolder.mkpath(".");

    //Create photomosaic generators
    CPUPhotomosaicGenerator generator;
    CUDAPhotomosaicGenerator generatorCUDA;
    generatorCUDA.setLibraryBatchSize(1000);

    //Set mode
    PhotomosaicGeneratorBase::Mode mode = PhotomosaicGeneratorBase::Mode::CIEDE2000;
    generator.setMode(mode);
    generatorCUDA.setMode(mode);

    //Set cell group
    const size_t cellSize = 128;
    CellGroup cellGroup;
    cellGroup.setCellShape(CellShape(cellSize));
    cellGroup.setSizeSteps(1);
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
    for (auto it = images.cbegin();
         it != images.cend() && it != images.cbegin() + TST_Generator::ITERATIONS; ++it)
    {
        //Load and set main image
        cv::Mat mainImage = cv::imread((imageFolder.path() + "/" + *it).toStdString());
        cv::resize(mainImage, mainImage, cv::Size(), 0.5, 0.5);
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

        EXPECT_EQ(bestFitDiff, 0) << it->toStdString();

        //Best fits different, save results
        if (bestFitDiff != 0)
        {
            //Build photomosaics
            cv::Mat result = generator.buildPhotomosaic();
            cv::Mat resultCUDA = generatorCUDA.buildPhotomosaic();

            //Save photomosaics
            cv::imwrite((resultFolder.path() + "/" + *it).toStdString(), result);
            cv::imwrite((resultFolder.path() + "/CUDA-" + *it).toStdString(), resultCUDA);
        }
    }
}

#endif

#endif // TST_GENERATOR_H
