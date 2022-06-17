#pragma once

#include <gtest/gtest.h>
#include <gmock/gmock-matchers.h>
#include <QDir>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#ifdef CUDA
#include <cuda_runtime.h>
#include "..\src\Photomosaic\CUDA\CUDAPhotomosaicGenerator.h"
#endif

#include "..\src\Photomosaic\CPUPhotomosaicGenerator.h"
#include "testutility.h"
#include "..\src\Grid\GridGenerator.h"
#include "..\src\ImageLibrary\ImageLibrary.h"

namespace TST_Generator
{
    static const size_t ITERATIONS = 10;

    //Check if best fits are equal
    ::testing::AssertionResult CompareBestFits(const GridUtility::MosaicBestFit &bestFit1, const GridUtility::MosaicBestFit &bestFit2)
    {

        if (bestFit1.size() != bestFit2.size())
            return ::testing::AssertionFailure() << "Size steps don't match: " << bestFit1.size() << " vs " << bestFit2.size() << ". ";

        for (size_t step = 0; step < bestFit1.size(); ++step)
        {
            if (bestFit1.at(step).size() != bestFit2.at(step).size())
                return ::testing::AssertionFailure() << "Rows don't match at (Step " << step << "): " << bestFit1.at(step).size() << " vs " << bestFit2.at(step).size() << ". ";

            for (size_t y = 0; y < bestFit1.at(step).size(); ++y)
            {
                if (bestFit1.at(step).at(y).size() != bestFit2.at(step).at(y).size())
                    return ::testing::AssertionFailure() << "Cols don't match at (Step " << step << ", y" << y << "): "
                    << bestFit1.at(step).at(y).size() << " vs " << bestFit2.at(step).at(y).size() << ". ";

                for (size_t x = 0; x < bestFit1.at(step).at(y).size(); ++x)
                {
                    if (bestFit1.at(step).at(y).at(x) != bestFit2.at(step).at(y).at(x))
                    {
                        auto assertFail = testing::AssertionFailure() << "Best fits don't match at (Step " << step << ", y" << y << ", x" << x << "): ";
                        if (bestFit1.at(step).at(y).at(x).has_value())
                            assertFail << bestFit1.at(step).at(y).at(x).value();
                        else
                            assertFail << "nullopt";
                        assertFail << " vs ";
                        if (bestFit2.at(step).at(y).at(x).has_value())
                            assertFail << bestFit2.at(step).at(y).at(x).value();
                        else
                            assertFail << "nullopt";
                        return assertFail << ". ";
                    }
                }
            }
        }

        return ::testing::AssertionSuccess();
    }
}

class GeneratorFixture : public ::testing::Test, public CPUPhotomosaicGenerator
{
protected:
    GeneratorFixture() : CPUPhotomosaicGenerator() {}

    void CreateCellGroup(const size_t t_cellSize = 128, const size_t t_steps = 0, const size_t t_detail = 100)
    {
        CellGroup cellGroup;
        cellGroup.setCellShape(CellShape(t_cellSize));
        cellGroup.setSizeSteps(t_steps);
        cellGroup.setDetail(t_detail);
        setCellGroup(cellGroup);
    }

    void CreateHexagonCellGroup(const size_t t_cellSize = 128, const size_t t_steps = 0, const size_t t_detail = 100)
    {
        CellGroup cellGroup;
        CellShape cellShape;
        cellShape.loadFromFile("E:/Desktop/MosaicMagnifique/MosaicMagnifique/Cells/Hexagon.mcs");
        cellGroup.setCellShape(cellShape.resized(t_cellSize));
        cellGroup.setSizeSteps(t_steps);
        cellGroup.setDetail(t_detail);
        setCellGroup(cellGroup);
    }

    void LoadImageLibrary()
    {
        ImageLibrary lib(m_cells.getCellSize(0, false));
        lib.loadFromFile("E:/Desktop/MosaicMagnifique/MosaicMagnifique/Library/lib.mil");
        setLibrary(lib.getImages());
    }

    void LoadRandomImage(const double t_scale)
    {
        //Folder containing multiple images
        QDir imageFolder("E:/Desktop/MosaicMagnifique/NewLib");
        QStringList images = imageFolder.entryList(QDir::Filter::Files);

        //Select a random main image from list
        cv::Mat mainImage = cv::imread((imageFolder.path() + "/" +
            images.at((rand() * images.size()) / RAND_MAX)).toStdString());
        cv::resize(mainImage, mainImage, cv::Size(), t_scale, t_scale);
        setMainImage(mainImage);
    }

    ::testing::AssertionResult TestBestFitsConsistency()
    {
        GridUtility::MosaicBestFit lastBestFit;
        for (size_t i = 0; i < TST_Generator::ITERATIONS; ++i)
        {
            //Generate grid state
            setGridState(GridGenerator::getGridState(m_cells, m_img, m_img.rows, m_img.cols));

            //Generate best fits
            generateBestFits();

            if (i > 0)
            {
                //Compare best fits
                if (::testing::AssertionResult compareResult = TST_Generator::CompareBestFits(getBestFits(), lastBestFit); !compareResult)
                    return compareResult;
            }
            lastBestFit = getBestFits();
        }

        return ::testing::AssertionSuccess();
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////
//******************************************** DETAIL ********************************************//
////////////////////////////////////////////////////////////////////////////////////////////////////

//Generates photomosaic best fits using identical settings multiple times
//Expects all the best fits to be identical
TEST_F(GeneratorFixture, CONSISTENCY_RGB_EUCLIDEAN_Detail_100)
{
    setColourDifference(ColourDifference::Type::RGB_EUCLIDEAN);
    setColourScheme(ColourScheme::Type::NONE);
    setRepeat(0, 0);
    CreateCellGroup(128, 0, 100);
    LoadImageLibrary();
    LoadRandomImage(0.5);

    ASSERT_TRUE(TestBestFitsConsistency());
}

//Generates photomosaic best fits using identical settings multiple times
//Expects all the best fits to be identical
TEST_F(GeneratorFixture, CONSISTENCY_RGB_EUCLIDEAN_Detail_50)
{
    setColourDifference(ColourDifference::Type::RGB_EUCLIDEAN);
    setColourScheme(ColourScheme::Type::NONE);
    setRepeat(0, 0);
    CreateCellGroup(128, 0, 50);
    LoadImageLibrary();
    LoadRandomImage(0.5);

    ASSERT_TRUE(TestBestFitsConsistency());
}

//Generates photomosaic best fits using identical settings multiple times
//Expects all the best fits to be identical
TEST_F(GeneratorFixture, CONSISTENCY_CIE76_Detail_100)
{
    setColourDifference(ColourDifference::Type::CIE76);
    setColourScheme(ColourScheme::Type::NONE);
    setRepeat(0, 0);
    CreateCellGroup(128, 0, 100);
    LoadImageLibrary();
    LoadRandomImage(0.5);

    ASSERT_TRUE(TestBestFitsConsistency());
}

//Generates photomosaic best fits using identical settings multiple times
//Expects all the best fits to be identical
TEST_F(GeneratorFixture, CONSISTENCY_CIE76_Detail_50)
{
    setColourDifference(ColourDifference::Type::CIE76);
    setColourScheme(ColourScheme::Type::NONE);
    setRepeat(0, 0);
    CreateCellGroup(128, 0, 50);
    LoadImageLibrary();
    LoadRandomImage(0.5);

    ASSERT_TRUE(TestBestFitsConsistency());
}

//Generates photomosaic best fits using identical settings multiple times
//Expects all the best fits to be identical
TEST_F(GeneratorFixture, CONSISTENCY_CIEDE2000_Detail_100)
{
    setColourDifference(ColourDifference::Type::CIEDE2000);
    setColourScheme(ColourScheme::Type::NONE);
    setRepeat(0, 0);
    CreateCellGroup(128, 0, 100);
    LoadImageLibrary();
    LoadRandomImage(0.5);

    ASSERT_TRUE(TestBestFitsConsistency());
}

//Generates photomosaic best fits using identical settings multiple times
//Expects all the best fits to be identical
TEST_F(GeneratorFixture, CONSISTENCY_CIEDE2000_Detail_50)
{
    setColourDifference(ColourDifference::Type::CIEDE2000);
    setColourScheme(ColourScheme::Type::NONE);
    setRepeat(0, 0);
    CreateCellGroup(128, 0, 50);
    LoadImageLibrary();
    LoadRandomImage(0.5);

    ASSERT_TRUE(TestBestFitsConsistency());
}


////////////////////////////////////////////////////////////////////////////////////////////////////
//******************************************** REPEATS *******************************************//
////////////////////////////////////////////////////////////////////////////////////////////////////

//Generates photomosaic best fits using identical settings multiple times
//Expects all the best fits to be identical
TEST_F(GeneratorFixture, CONSISTENCY_RGB_EUCLIDEAN_Detail_50_Repeats)
{
    setColourDifference(ColourDifference::Type::RGB_EUCLIDEAN);
    setColourScheme(ColourScheme::Type::NONE);
    setRepeat(20, 10000);
    CreateCellGroup(128, 0, 50);
    LoadImageLibrary();
    LoadRandomImage(0.5);

    ASSERT_TRUE(TestBestFitsConsistency());
}

//Generates photomosaic best fits using identical settings multiple times
//Expects all the best fits to be identical
TEST_F(GeneratorFixture, CONSISTENCY_CIE76_Detail_50_Repeats)
{
    setColourDifference(ColourDifference::Type::CIE76);
    setColourScheme(ColourScheme::Type::NONE);
    setRepeat(20, 10000);
    CreateCellGroup(128, 0, 50);
    LoadImageLibrary();
    LoadRandomImage(0.5);

    ASSERT_TRUE(TestBestFitsConsistency());
}

//Generates photomosaic best fits using identical settings multiple times
//Expects all the best fits to be identical
TEST_F(GeneratorFixture, CONSISTENCY_CIEDE2000_Detail_50_Repeats)
{
    setColourDifference(ColourDifference::Type::CIEDE2000);
    setColourScheme(ColourScheme::Type::NONE);
    setRepeat(20, 10000);
    CreateCellGroup(128, 0, 50);
    LoadImageLibrary();
    LoadRandomImage(0.5);

    ASSERT_TRUE(TestBestFitsConsistency());
}


////////////////////////////////////////////////////////////////////////////////////////////////////
//****************************************** SIZE STEPS ******************************************//
////////////////////////////////////////////////////////////////////////////////////////////////////

//Generates photomosaic best fits using identical settings multiple times
//Expects all the best fits to be identical
TEST_F(GeneratorFixture, CONSISTENCY_RGB_EUCLIDEAN_Size_Steps)
{
    setColourDifference(ColourDifference::Type::RGB_EUCLIDEAN);
    setColourScheme(ColourScheme::Type::NONE);
    setRepeat(0, 0);
    CreateCellGroup(128, 1, 50);
    LoadImageLibrary();
    LoadRandomImage(0.5);

    ASSERT_TRUE(TestBestFitsConsistency());
}

//Generates photomosaic best fits using identical settings multiple times
//Expects all the best fits to be identical
TEST_F(GeneratorFixture, CONSISTENCY_CIE76_Size_Steps)
{
    setColourDifference(ColourDifference::Type::CIE76);
    setColourScheme(ColourScheme::Type::NONE);
    setRepeat(0, 0);
    CreateCellGroup(128, 1, 50);
    LoadImageLibrary();
    LoadRandomImage(0.5);

    ASSERT_TRUE(TestBestFitsConsistency());
}

//Generates photomosaic best fits using identical settings multiple times
//Expects all the best fits to be identical
TEST_F(GeneratorFixture, CONSISTENCY_CIEDE2000_Size_Steps)
{
    setColourDifference(ColourDifference::Type::CIEDE2000);
    setColourScheme(ColourScheme::Type::NONE);
    setRepeat(0, 0);
    CreateCellGroup(128, 1, 50);
    LoadImageLibrary();
    LoadRandomImage(0.5);

    ASSERT_TRUE(TestBestFitsConsistency());
}


////////////////////////////////////////////////////////////////////////////////////////////////////
//****************************************** CELL SHAPES *****************************************//
////////////////////////////////////////////////////////////////////////////////////////////////////

//Generates photomosaic best fits using identical settings multiple times
//Expects all the best fits to be identical
TEST_F(GeneratorFixture, CONSISTENCY_RGB_EUCLIDEAN_Cell_Shapes)
{
    setColourDifference(ColourDifference::Type::RGB_EUCLIDEAN);
    setColourScheme(ColourScheme::Type::NONE);
    setRepeat(0, 0);
    CreateHexagonCellGroup(128, 0, 50);
    LoadImageLibrary();
    LoadRandomImage(0.5);

    ASSERT_TRUE(TestBestFitsConsistency());
}

//Generates photomosaic best fits using identical settings multiple times
//Expects all the best fits to be identical
TEST_F(GeneratorFixture, CONSISTENCY_CIE76_Cell_Shapes)
{
    setColourDifference(ColourDifference::Type::CIE76);
    setColourScheme(ColourScheme::Type::NONE);
    setRepeat(0, 0);
    CreateHexagonCellGroup(128, 0, 50);
    LoadImageLibrary();
    LoadRandomImage(0.5);

    ASSERT_TRUE(TestBestFitsConsistency());
}

//Generates photomosaic best fits using identical settings multiple times
//Expects all the best fits to be identical
TEST_F(GeneratorFixture, CONSISTENCY_CIEDE2000_Cell_Shapes)
{
    setColourDifference(ColourDifference::Type::CIEDE2000);
    setColourScheme(ColourScheme::Type::NONE);
    setRepeat(0, 0);
    CreateHexagonCellGroup(128, 0, 50);
    LoadImageLibrary();
    LoadRandomImage(0.5);

    ASSERT_TRUE(TestBestFitsConsistency());
}