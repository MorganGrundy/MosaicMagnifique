#pragma once

#include <gtest/gtest.h>
#include <gmock/gmock-matchers.h>
#include <QDir>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <cuda_runtime.h>

#include "..\src\Photomosaic\CPUPhotomosaicGenerator.h"
#include "..\src\Photomosaic\CUDA\CUDAPhotomosaicGenerator.h"
#include "testutility.h"
#include "..\src\Grid\GridGenerator.h"
#include "..\src\ImageLibrary\ImageLibrary.h"
#include "tst_Generator.h"
#include "..\src\Other\TimingLogger.h"

class CUDAGeneratorFixture : public ::testing::Test, public CUDAPhotomosaicGenerator
{
protected:
    CUDAGeneratorFixture() : CUDAPhotomosaicGenerator() {}

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
        //Tell TimingLogger to output files to subdir %test_suite_name%/%test_case_name%
        QString testSuite(::testing::UnitTest::GetInstance()->current_test_info()->test_suite_name());
        TimingLogger::SetSubdir(testSuite + "/" + ::testing::UnitTest::GetInstance()->current_test_info()->name());

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

class CPUvsCUDAGeneratorFixture : public ::testing::Test
{
protected:
    CPUvsCUDAGeneratorFixture() {}

    void setColourDifference(const ColourDifference::Type t_colourDiff)
    {
        cpuGenerator.setColourDifference(t_colourDiff);
        cudaGenerator.setColourDifference(t_colourDiff);
    }

    void setColourScheme(const ColourScheme::Type t_colourScheme)
    {
        cpuGenerator.setColourScheme(t_colourScheme);
        cudaGenerator.setColourScheme(t_colourScheme);
    }

    void setRepeat(const int t_repeatRange, const int t_repeatAddition)
    {
        cpuGenerator.setRepeat(t_repeatRange, t_repeatAddition);
        cudaGenerator.setRepeat(t_repeatRange, t_repeatAddition);
    }

    void CreateCellGroup(const size_t t_cellSize = 128, const size_t t_steps = 0, const size_t t_detail = 100)
    {
        CellGroup cellGroup;
        cellGroup.setCellShape(CellShape(t_cellSize));
        cellGroup.setSizeSteps(t_steps);
        cellGroup.setDetail(t_detail);

        cpuGenerator.setCellGroup(cellGroup);
        cudaGenerator.setCellGroup(cellGroup);
    }

    void CreateHexagonCellGroup(const size_t t_cellSize = 128, const size_t t_steps = 0, const size_t t_detail = 100)
    {
        CellGroup cellGroup;
        CellShape cellShape;
        cellShape.loadFromFile("E:/Desktop/MosaicMagnifique/MosaicMagnifique/Cells/Hexagon.mcs");
        cellGroup.setCellShape(cellShape.resized(t_cellSize));
        cellGroup.setSizeSteps(t_steps);
        cellGroup.setDetail(t_detail);

        cpuGenerator.setCellGroup(cellGroup);
        cudaGenerator.setCellGroup(cellGroup);
    }

    void LoadImageLibrary()
    {
        ImageLibrary lib(cpuGenerator.getCellGroup().getCellSize(0, false));
        lib.loadFromFile("E:/Desktop/MosaicMagnifique/MosaicMagnifique/Library/lib.mil");

        cpuGenerator.setLibrary(lib.getImages());
        cudaGenerator.setLibrary(lib.getImages());
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

        cpuGenerator.setMainImage(mainImage);
        cudaGenerator.setMainImage(mainImage);
    }

    ::testing::AssertionResult TestCompare()
    {
        //Tell TimingLogger to output files to subdir %test_suite_name%/%test_case_name%
        QString testSuite(::testing::UnitTest::GetInstance()->current_test_info()->test_suite_name());
        TimingLogger::SetSubdir(testSuite + "/" + ::testing::UnitTest::GetInstance()->current_test_info()->name());

        //Create folder for saving photomosaics
        QDir resultFolder(QDir::currentPath() + "/testcases/generator");
        if (!resultFolder.exists())
            resultFolder.mkpath(".");

        //Folder containing multiple images
        QDir imageFolder("E:/Desktop/MosaicMagnifique/NewLib");
        QStringList images = imageFolder.entryList(QDir::Filter::Files);

        //Iterate over all images
        for (auto it = images.cbegin(); it != images.cend() && it != images.cbegin() + TST_Generator::ITERATIONS; ++it)
        {
            //Load and set main image
            cv::Mat mainImage = cv::imread((imageFolder.path() + "/" + *it).toStdString());
            cv::resize(mainImage, mainImage, cv::Size(), 0.5, 0.5);
            cpuGenerator.setMainImage(mainImage);
            cudaGenerator.setMainImage(mainImage);

            //Generate grid state
            const GridUtility::MosaicBestFit gridState = GridGenerator::getGridState(cpuGenerator.getCellGroup(), mainImage, mainImage.rows, mainImage.cols);
            cpuGenerator.setGridState(gridState);
            cudaGenerator.setGridState(gridState);

            //Generate best fits
            cpuGenerator.generateBestFits();
            cudaGenerator.generateBestFits();

            //Compare best fits
            if (::testing::AssertionResult compareResult = TST_Generator::CompareBestFits(cpuGenerator.getBestFits(), cudaGenerator.getBestFits()); !compareResult)
            {
                //Build photomosaics
                cv::Mat result = cpuGenerator.buildPhotomosaic();
                cv::Mat resultCUDA = cudaGenerator.buildPhotomosaic();

                //Save photomosaics
                cv::imwrite((resultFolder.path() + "/" + *it).toStdString(), result);
                cv::imwrite((resultFolder.path() + "/CUDA-" + *it).toStdString(), resultCUDA);

                return compareResult;
            }
        }

        return ::testing::AssertionSuccess();
    }

private:
    CPUPhotomosaicGenerator cpuGenerator;
    CUDAPhotomosaicGenerator cudaGenerator;
};


////////////////////////////////////////////////////////////////////////////////////////////////////
//***************************************** CUDA DETAIL ******************************************//
////////////////////////////////////////////////////////////////////////////////////////////////////

//Generates (with CUDA) photomosaic best fits using identical settings multiple times
//Expects all the best fits to be identical
TEST_F(CUDAGeneratorFixture, CONSISTENCY_RGB_EUCLIDEAN_Detail_100)
{
    setColourDifference(ColourDifference::Type::RGB_EUCLIDEAN);
    setColourScheme(ColourScheme::Type::NONE);
    setRepeat(0, 0);
    CreateCellGroup(128, 0, 100);
    LoadImageLibrary();
    LoadRandomImage(0.5);

    ASSERT_TRUE(TestBestFitsConsistency());
}

//Generates (with CUDA) photomosaic best fits using identical settings multiple times
//Expects all the best fits to be identical
TEST_F(CUDAGeneratorFixture, CONSISTENCY_RGB_EUCLIDEAN_Detail_50)
{
    setColourDifference(ColourDifference::Type::RGB_EUCLIDEAN);
    setColourScheme(ColourScheme::Type::NONE);
    setRepeat(0, 0);
    CreateCellGroup(128, 0, 50);
    LoadImageLibrary();
    LoadRandomImage(0.5);

    ASSERT_TRUE(TestBestFitsConsistency());
}

//Generates (with CUDA) photomosaic best fits using identical settings multiple times
//Expects all the best fits to be identical
TEST_F(CUDAGeneratorFixture, CONSISTENCY_CIE76_Detail_100)
{
    setColourDifference(ColourDifference::Type::CIE76);
    setColourScheme(ColourScheme::Type::NONE);
    setRepeat(0, 0);
    CreateCellGroup(128, 0, 100);
    LoadImageLibrary();
    LoadRandomImage(0.5);

    ASSERT_TRUE(TestBestFitsConsistency());
}

//Generates (with CUDA) photomosaic best fits using identical settings multiple times
//Expects all the best fits to be identical
TEST_F(CUDAGeneratorFixture, CONSISTENCY_CIE76_Detail_50)
{
    setColourDifference(ColourDifference::Type::CIE76);
    setColourScheme(ColourScheme::Type::NONE);
    setRepeat(0, 0);
    CreateCellGroup(128, 0, 50);
    LoadImageLibrary();
    LoadRandomImage(0.5);

    ASSERT_TRUE(TestBestFitsConsistency());
}

//Generates (with CUDA) photomosaic best fits using identical settings multiple times
//Expects all the best fits to be identical
TEST_F(CUDAGeneratorFixture, CONSISTENCY_CIEDE2000_Detail_100)
{
    setColourDifference(ColourDifference::Type::CIEDE2000);
    setColourScheme(ColourScheme::Type::NONE);
    setRepeat(0, 0);
    CreateCellGroup(128, 0, 100);
    LoadImageLibrary();
    LoadRandomImage(0.5);

    ASSERT_TRUE(TestBestFitsConsistency());
}

//Generates (with CUDA) photomosaic best fits using identical settings multiple times
//Expects all the best fits to be identical
TEST_F(CUDAGeneratorFixture, CONSISTENCY_CIEDE2000_Detail_50)
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
//***************************************** CUDA REPEATS *****************************************//
////////////////////////////////////////////////////////////////////////////////////////////////////

//Generates (with CUDA) photomosaic best fits using identical settings multiple times
//Expects all the best fits to be identical
TEST_F(CUDAGeneratorFixture, CONSISTENCY_RGB_EUCLIDEAN_Repeats)
{
    setColourDifference(ColourDifference::Type::RGB_EUCLIDEAN);
    setColourScheme(ColourScheme::Type::NONE);
    setRepeat(20, 10000);
    CreateCellGroup(128, 0, 50);
    LoadImageLibrary();
    LoadRandomImage(0.5);

    ASSERT_TRUE(TestBestFitsConsistency());
}

//Generates (with CUDA) photomosaic best fits using identical settings multiple times
//Expects all the best fits to be identical
TEST_F(CUDAGeneratorFixture, CONSISTENCY_CIE76_Repeats)
{
    setColourDifference(ColourDifference::Type::CIE76);
    setColourScheme(ColourScheme::Type::NONE);
    setRepeat(20, 10000);
    CreateCellGroup(128, 0, 50);
    LoadImageLibrary();
    LoadRandomImage(0.5);

    ASSERT_TRUE(TestBestFitsConsistency());
}

//Generates (with CUDA) photomosaic best fits using identical settings multiple times
//Expects all the best fits to be identical
TEST_F(CUDAGeneratorFixture, CONSISTENCY_CIEDE2000_Repeats)
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
//*************************************** CUDA SIZE STEPS ****************************************//
////////////////////////////////////////////////////////////////////////////////////////////////////

//Generates photomosaic best fits using identical settings multiple times
//Expects all the best fits to be identical
TEST_F(CUDAGeneratorFixture, CONSISTENCY_RGB_EUCLIDEAN_Size_Steps)
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
TEST_F(CUDAGeneratorFixture, CONSISTENCY_CIE76_Size_Steps)
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
TEST_F(CUDAGeneratorFixture, CONSISTENCY_CIEDE2000_Size_Steps)
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
//*************************************** CUDA CELL SHAPES ***************************************//
////////////////////////////////////////////////////////////////////////////////////////////////////

//Generates photomosaic best fits using identical settings multiple times
//Expects all the best fits to be identical
TEST_F(CUDAGeneratorFixture, CONSISTENCY_RGB_EUCLIDEAN_Cell_Shapes)
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
TEST_F(CUDAGeneratorFixture, CONSISTENCY_CIE76_Cell_Shapes)
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
TEST_F(CUDAGeneratorFixture, CONSISTENCY_CIEDE2000_Cell_Shapes)
{
    setColourDifference(ColourDifference::Type::CIEDE2000);
    setColourScheme(ColourScheme::Type::NONE);
    setRepeat(0, 0);
    CreateHexagonCellGroup(128, 0, 50);
    LoadImageLibrary();
    LoadRandomImage(0.5);

    ASSERT_TRUE(TestBestFitsConsistency());
}


////////////////////////////////////////////////////////////////////////////////////////////////////
//************************************** DETAIL CPU vs CUDA **************************************//
////////////////////////////////////////////////////////////////////////////////////////////////////

//Generates photomosaic best fits using identical settings with and without CUDA
//Expects the best fits to be identical
//Tests with multiple main images
TEST_F(CPUvsCUDAGeneratorFixture, COMPARE_RGB_EUCLIDEAN_Detail_100)
{
    setColourDifference(ColourDifference::Type::RGB_EUCLIDEAN);
    setColourScheme(ColourScheme::Type::NONE);
    setRepeat(0, 0);
    CreateCellGroup(128, 0, 100);
    LoadImageLibrary();

    ASSERT_TRUE(TestCompare());
}

//Generates photomosaic best fits using identical settings with and without CUDA
//Expects the best fits to be identical
//Tests with multiple main images
TEST_F(CPUvsCUDAGeneratorFixture, COMPARE_RGB_EUCLIDEAN_Detail_50)
{
    setColourDifference(ColourDifference::Type::RGB_EUCLIDEAN);
    setColourScheme(ColourScheme::Type::NONE);
    setRepeat(0, 0);
    CreateCellGroup(128, 0, 50);
    LoadImageLibrary();

    ASSERT_TRUE(TestCompare());
}

//Generates photomosaic best fits using identical settings with and without CUDA
//Expects the best fits to be identical
//Tests with multiple main images
TEST_F(CPUvsCUDAGeneratorFixture, COMPARE_CIE76_Detail_100)
{
    setColourDifference(ColourDifference::Type::CIE76);
    setColourScheme(ColourScheme::Type::NONE);
    setRepeat(0, 0);
    CreateCellGroup(128, 0, 100);
    LoadImageLibrary();

    ASSERT_TRUE(TestCompare());
}

//Generates photomosaic best fits using identical settings with and without CUDA
//Expects the best fits to be identical
//Tests with multiple main images
TEST_F(CPUvsCUDAGeneratorFixture, COMPARE_CIE76_Detail_50)
{
    setColourDifference(ColourDifference::Type::CIE76);
    setColourScheme(ColourScheme::Type::NONE);
    setRepeat(0, 0);
    CreateCellGroup(128, 0, 50);
    LoadImageLibrary();

    ASSERT_TRUE(TestCompare());
}

//Generates photomosaic best fits using identical settings with and without CUDA
//Expects the best fits to be identical
//Tests with multiple main images
TEST_F(CPUvsCUDAGeneratorFixture, COMPARE_CIEDE2000_Detail_100)
{
    setColourDifference(ColourDifference::Type::CIEDE2000);
    setColourScheme(ColourScheme::Type::NONE);
    setRepeat(0, 0);
    CreateCellGroup(128, 0, 100);
    LoadImageLibrary();

    ASSERT_TRUE(TestCompare());
}

//Generates photomosaic best fits using identical settings with and without CUDA
//Expects the best fits to be identical
//Tests with multiple main images
TEST_F(CPUvsCUDAGeneratorFixture, COMPARE_CIEDE2000_Detail_50)
{
    setColourDifference(ColourDifference::Type::CIEDE2000);
    setColourScheme(ColourScheme::Type::NONE);
    setRepeat(0, 0);
    CreateCellGroup(128, 0, 50);
    LoadImageLibrary();

    ASSERT_TRUE(TestCompare());
}


////////////////////////////////////////////////////////////////////////////////////////////////////
//************************************* REPEATS CPU vs CUDA **************************************//
////////////////////////////////////////////////////////////////////////////////////////////////////

//Generates photomosaic best fits using identical settings with and without CUDA
//Expects the best fits to be identical
//Tests with multiple main images
TEST_F(CPUvsCUDAGeneratorFixture, COMPARE_RGB_EUCLIDEAN_Repeats)
{
    setColourDifference(ColourDifference::Type::RGB_EUCLIDEAN);
    setColourScheme(ColourScheme::Type::NONE);
    setRepeat(20, 10000);
    CreateCellGroup(128, 0, 50);
    LoadImageLibrary();

    ASSERT_TRUE(TestCompare());
}

//Generates photomosaic best fits using identical settings with and without CUDA
//Expects the best fits to be identical
//Tests with multiple main images
TEST_F(CPUvsCUDAGeneratorFixture, COMPARE_CIE76_Repeats)
{
    setColourDifference(ColourDifference::Type::CIE76);
    setColourScheme(ColourScheme::Type::NONE);
    setRepeat(20, 10000);
    CreateCellGroup(128, 0, 50);
    LoadImageLibrary();

    ASSERT_TRUE(TestCompare());
}

//Generates photomosaic best fits using identical settings with and without CUDA
//Expects the best fits to be identical
//Tests with multiple main images
TEST_F(CPUvsCUDAGeneratorFixture, COMPARE_CIEDE2000_Repeats)
{
    setColourDifference(ColourDifference::Type::CIEDE2000);
    setColourScheme(ColourScheme::Type::NONE);
    setRepeat(20, 10000);
    CreateCellGroup(128, 0, 50);
    LoadImageLibrary();

    ASSERT_TRUE(TestCompare());
}


////////////////////////////////////////////////////////////////////////////////////////////////////
//************************************ SIZE STEPS CPU vs CUDA ************************************//
////////////////////////////////////////////////////////////////////////////////////////////////////

//Generates photomosaic best fits using identical settings with and without CUDA
//Expects the best fits to be identical
//Tests with multiple main images
TEST_F(CPUvsCUDAGeneratorFixture, COMPARE_RGB_EUCLIDEAN_Size_Steps)
{
    setColourDifference(ColourDifference::Type::RGB_EUCLIDEAN);
    setColourScheme(ColourScheme::Type::NONE);
    setRepeat(0, 0);
    CreateCellGroup(128, 1, 50);
    LoadImageLibrary();

    ASSERT_TRUE(TestCompare());
}

//Generates photomosaic best fits using identical settings with and without CUDA
//Expects the best fits to be identical
//Tests with multiple main images
TEST_F(CPUvsCUDAGeneratorFixture, COMPARE_CIE76_Size_Steps)
{
    setColourDifference(ColourDifference::Type::CIE76);
    setColourScheme(ColourScheme::Type::NONE);
    setRepeat(0, 0);
    CreateCellGroup(128, 1, 50);
    LoadImageLibrary();

    ASSERT_TRUE(TestCompare());
}

//Generates photomosaic best fits using identical settings with and without CUDA
//Expects the best fits to be identical
//Tests with multiple main images
TEST_F(CPUvsCUDAGeneratorFixture, COMPARE_CIEDE2000_Size_Steps)
{
    setColourDifference(ColourDifference::Type::CIEDE2000);
    setColourScheme(ColourScheme::Type::NONE);
    setRepeat(0, 0);
    CreateCellGroup(128, 1, 50);
    LoadImageLibrary();

    ASSERT_TRUE(TestCompare());
}


////////////////////////////////////////////////////////////////////////////////////////////////////
//************************************ CELL SHAPES CPU vs CUDA ***********************************//
////////////////////////////////////////////////////////////////////////////////////////////////////

//Generates photomosaic best fits using identical settings with and without CUDA
//Expects the best fits to be identical
//Tests with multiple main images
TEST_F(CPUvsCUDAGeneratorFixture, COMPARE_RGB_EUCLIDEAN_Cell_Shapes)
{
    setColourDifference(ColourDifference::Type::RGB_EUCLIDEAN);
    setColourScheme(ColourScheme::Type::NONE);
    setRepeat(0, 0);
    CreateHexagonCellGroup(128, 0, 50);
    LoadImageLibrary();

    ASSERT_TRUE(TestCompare());
}

//Generates photomosaic best fits using identical settings with and without CUDA
//Expects the best fits to be identical
//Tests with multiple main images
TEST_F(CPUvsCUDAGeneratorFixture, COMPARE_CIE76_Cell_Shapes)
{
    setColourDifference(ColourDifference::Type::CIE76);
    setColourScheme(ColourScheme::Type::NONE);
    setRepeat(0, 0);
    CreateHexagonCellGroup(128, 0, 50);
    LoadImageLibrary();

    ASSERT_TRUE(TestCompare());
}

//Generates photomosaic best fits using identical settings with and without CUDA
//Expects the best fits to be identical
//Tests with multiple main images
TEST_F(CPUvsCUDAGeneratorFixture, COMPARE_CIEDE2000_Cell_Shapes)
{
    setColourDifference(ColourDifference::Type::CIEDE2000);
    setColourScheme(ColourScheme::Type::NONE);
    setRepeat(0, 0);
    CreateHexagonCellGroup(128, 0, 50);
    LoadImageLibrary();

    ASSERT_TRUE(TestCompare());
}