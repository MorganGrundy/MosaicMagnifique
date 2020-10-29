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
                    ++differences;
            }
        }
    }

    return differences;
}

//Generates a random string of given length
QString getRandomString(const size_t length = 10)
{
    const QString possibleCharacters("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
                                     "0123456789");

    QString randomString;
    for (size_t i = 0; i < length; ++i)
    {
        const QChar randomChar = possibleCharacters.at(rand() % possibleCharacters.length());
        randomString.append(randomChar);
    }

    return randomString;
}

//Saves a test case
void saveTestcase(const PhotomosaicGeneratorBase::Mode mode, const cv::Mat &mainImage,
                  const ImageLibrary &lib, const CellGroup &cellGroup,
                  const int repeatRange, const int repeatAddition)
{
    //Generate unique, random name for test case
    QString testcaseName;
    QDir testcaseDir;
    do
    {
        testcaseName = getRandomString();
        testcaseDir.setPath(QDir::currentPath() + "/testcases/generator/" + testcaseName);
    } while (testcaseDir.exists());

    //Create folder
    testcaseDir.mkpath(".");

    //Save main image
    cv::imwrite((testcaseDir.path() + "/MainImage.jpg").toStdString(), mainImage);

    //Save image library
    lib.saveToFile(testcaseDir.path() + "/lib.mil");

    //Save cell shape
    cellGroup.getCell(0).saveToFile(testcaseDir.path() + "/cell.mcs");

    //Save settings
    QFile file(testcaseDir.path() + "/settings.dat");
    file.open(QIODevice::WriteOnly);
    if (file.isWritable())
    {
        QDataStream out(&file);
        out.setVersion(QDataStream::Qt_5_0);
        out << static_cast<qint32>(
            static_cast<std::underlying_type_t<PhotomosaicGeneratorBase::Mode>>(mode));
        out << static_cast<quint32>(cellGroup.getSizeSteps());
        out << cellGroup.getDetail();
        out << static_cast<qint32>(repeatRange);
        out << static_cast<qint32>(repeatAddition);
    }

    file.close();
}

}

TEST(Generator, BestFits_Saved)
{
    srand(static_cast<unsigned int>(time(NULL)));

    //Get all test cases
    const QDir testcaseDir(QDir::currentPath() + "/testcases/generator");
    if (testcaseDir.exists())
    {
        const QStringList testcases = testcaseDir.entryList(
            QDir::Filter::Dirs | QDir::Filter::NoDotAndDotDot);

        //Iterate over test cases
        for (const auto &test: testcases)
        {
            //Open settings file
            QFile file(testcaseDir.path() + "/" + test + "/settings.dat");
            file.open(QIODevice::ReadOnly);
            EXPECT_TRUE(file.isReadable());
            if (file.isReadable())
            {
                QDataStream in(&file);
                in.setVersion(QDataStream::Qt_5_0);

                //Load settings
                qint32 modeAsInt;
                in >> modeAsInt;
                quint32 sizeSteps;
                in >> sizeSteps;
                double detail;
                in >> detail;
                qint32 repeatRange, repeatAddition;
                in >> repeatRange;
                in >> repeatAddition;

                //Load main image
                cv::Mat mainImage =
                    cv::imread((testcaseDir.path() + "/" + test + "/MainImage.jpg").toStdString());

                //Load image library
                ImageLibrary lib(0);
                lib.loadFromFile(testcaseDir.path() + "/" + test + "/lib.mil");

                //Load cell shape
                CellShape cellShape;
                cellShape.loadFromFile(testcaseDir.path() + "/" + test + "/cell.mcs");
                CellGroup cellGroup;
                cellGroup.setCellShape(cellShape);
                cellGroup.setDetail(detail * 100);
                cellGroup.setSizeSteps(static_cast<size_t>(sizeSteps));

                //Create generators
                CPUPhotomosaicGenerator generator;
                CUDAPhotomosaicGenerator generatorCUDA;
                generatorCUDA.setLibraryBatchSize(100);

                PhotomosaicGeneratorBase::Mode mode = static_cast<PhotomosaicGeneratorBase::Mode>(
                    static_cast<std::underlying_type_t<PhotomosaicGeneratorBase::Mode>>(modeAsInt));
                generator.setMode(mode);
                generatorCUDA.setMode(mode);

                generator.setMainImage(mainImage);
                generatorCUDA.setMainImage(mainImage);

                generator.setCellGroup(cellGroup);
                generatorCUDA.setCellGroup(cellGroup);

                generator.setLibrary(lib.getImages());
                generatorCUDA.setLibrary(lib.getImages());

                const GridUtility::MosaicBestFit gridState =
                    GridGenerator::getGridState(cellGroup, mainImage, mainImage.rows,
                                                mainImage.cols);
                generator.setGridState(gridState);
                generatorCUDA.setGridState(gridState);

                generator.setRepeat(static_cast<int>(repeatRange),
                                    static_cast<int>(repeatAddition));
                generatorCUDA.setRepeat(static_cast<int>(repeatRange),
                                        static_cast<int>(repeatAddition));

                EXPECT_TRUE(generator.generateBestFits()) << "CPU generate test: "
                                                          << test.toStdString();
                EXPECT_TRUE(generatorCUDA.generateBestFits()) << "GPU generate test: "
                                                              << test.toStdString();

                size_t bestFitDiff = TST_Generator::bestFitsDiff(generator.getBestFits(),
                                                                 generatorCUDA.getBestFits());

                EXPECT_EQ(bestFitDiff, 0) << test.toStdString() << " difference = " << bestFitDiff;
            }
        }
    }
}

TEST(DISABLE_Generator, BestFits_NoRepeats)
{
    srand(static_cast<unsigned int>(time(NULL)));

    CPUPhotomosaicGenerator generator;
    CUDAPhotomosaicGenerator generatorCUDA;
    generatorCUDA.setLibraryBatchSize(100);

    PhotomosaicGeneratorBase::Mode mode = PhotomosaicGeneratorBase::Mode::CIE76;
    generator.setMode(mode);
    generatorCUDA.setMode(mode);

    const size_t mosaicWidth = 128, mosaicHeight = 128;
    const cv::Mat mainImage = TestUtility::createRandomImage(mosaicWidth, mosaicHeight,
                                                             TestUtility::ColourSpace::CIELAB);
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
    ImageLibrary lib(libSize);
    for (size_t i = 0; i < libSize; ++i)
        lib.addImage(createRandomImage(cellSize, cellSize, TestUtility::ColourSpace::CIELAB));
    generator.setLibrary(lib.getImages());
    generatorCUDA.setLibrary(lib.getImages());

    const GridUtility::MosaicBestFit gridState =
        GridGenerator::getGridState(cellGroup, mainImage, mosaicHeight, mosaicWidth);
    generator.setGridState(gridState);
    generatorCUDA.setGridState(gridState);

    const int repeatRange = 0;
    const int repeatAddition = 0;
    generator.setRepeat(repeatRange, repeatAddition);
    generatorCUDA.setRepeat(repeatRange, repeatAddition);

    ASSERT_TRUE(generator.generateBestFits());
    ASSERT_TRUE(generatorCUDA.generateBestFits());

    size_t bestFitDiff = TST_Generator::bestFitsDiff(generator.getBestFits(),
                                                     generatorCUDA.getBestFits());

    if (bestFitDiff > 0)
        TST_Generator::saveTestcase(mode, mainImage, lib, cellGroup, repeatRange, repeatAddition);

    ASSERT_EQ(bestFitDiff, 0) << "Difference = " << bestFitDiff;
}

TEST(DISABLE_Generator, BestFits_Repeats)
{
    srand(static_cast<unsigned int>(time(NULL)));

    CPUPhotomosaicGenerator generator;
    CUDAPhotomosaicGenerator generatorCUDA;
    generatorCUDA.setLibraryBatchSize(100);

    PhotomosaicGeneratorBase::Mode mode = PhotomosaicGeneratorBase::Mode::CIE76;
    generator.setMode(mode);
    generatorCUDA.setMode(mode);

    const size_t mosaicWidth = 128, mosaicHeight = 128;
    const cv::Mat mainImage = TestUtility::createRandomImage(mosaicWidth, mosaicHeight,
                                                             TestUtility::ColourSpace::CIELAB);
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
    ImageLibrary lib(libSize);
    for (size_t i = 0; i < libSize; ++i)
        lib.addImage(createRandomImage(cellSize, cellSize, TestUtility::ColourSpace::CIELAB));
    generator.setLibrary(lib.getImages());
    generatorCUDA.setLibrary(lib.getImages());

    const GridUtility::MosaicBestFit gridState =
        GridGenerator::getGridState(cellGroup, mainImage, mosaicHeight, mosaicWidth);
    generator.setGridState(gridState);
    generatorCUDA.setGridState(gridState);

    const int repeatRange = 20;
    const int repeatAddition = 10000;
    generator.setRepeat(repeatRange, repeatAddition);
    generatorCUDA.setRepeat(repeatRange, repeatAddition);

    ASSERT_TRUE(generator.generateBestFits());
    ASSERT_TRUE(generatorCUDA.generateBestFits());

    size_t bestFitDiff = TST_Generator::bestFitsDiff(generator.getBestFits(),
                                                     generatorCUDA.getBestFits());

    if (bestFitDiff > 0)
        TST_Generator::saveTestcase(mode, mainImage, lib, cellGroup, repeatRange, repeatAddition);

    ASSERT_EQ(bestFitDiff, 0) << "Difference = " << bestFitDiff;
}

#endif // TST_GENERATOR_H
