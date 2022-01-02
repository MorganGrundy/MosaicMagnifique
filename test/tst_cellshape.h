#pragma once

#include <gtest/gtest.h>
#include <gmock/gmock-matchers.h>
#include <QDir>
#include <QFileInfo>

#include "cellshape.h"
#include "testutility.h"

//Creates a cell shape, saves it to a file
//Loads all test cell shapes (with different .mcs versions), and then compares it
TEST(CellShape, Static_SaveAndLoad)
{
    //Create cell shape
    const size_t cellSize = 64;
    CellShape cell(cellSize);

    //Check if test for current cell shape version exists
    QDir cellFolder(QDir::currentPath() + "/testcases/cellshape");
    QFileInfo currentVersionCell(cellFolder, "cell-v" + QString::number(static_cast<uint>(
                                                 CellShape::MCS_VERSION)) + ".mcs");

    if (!currentVersionCell.exists() || !currentVersionCell.isFile())
    {
        //Save test cell for current version
        if (!cellFolder.exists())
            cellFolder.mkpath(".");
        cell.saveToFile(currentVersionCell.filePath());
    }

    //Get all test cells
    const QStringList testCells = cellFolder.entryList(QDir::Filter::Files);

    //Iterate over test cases
    for (const auto &testCell: testCells)
    {
        //Skip random test
        if (testCell.contains("cell-v"))
        {
            //Load cell shape from file
            CellShape loadedCell(cellSize);
            loadedCell.loadFromFile(cellFolder.path() + "/" + testCell);

            //Compare cell shapes
            ASSERT_EQ(cell, loadedCell) << "File: " << testCell.toStdString();
        }
    }
}

//Creates a random cell shape, saves it to a file, loads it, and then compares it
TEST(CellShape, Random_SaveAndLoad)
{
    srand(static_cast<unsigned int>(time(NULL)));

    const size_t cellSize = 64;

    //Test multiple random cell shapes
    for (size_t i = 0; i < 20; ++i)
    {
        //Create cell shape
        CellShape cell(TestUtility::createRandomImage(cellSize, cellSize, true));

        //Save cell shape to file
        QDir cellFolder(QDir::currentPath() + "/testcases/cellshape");
        if (!cellFolder.exists())
            cellFolder.mkpath(".");
        QString cellFile = cellFolder.path() + "/randCell.mcs";
        cell.saveToFile(cellFile);

        //Load cell shape from file
        CellShape loadedCell(cellSize);
        loadedCell.loadFromFile(cellFile);

        //Compare cell shape
        ASSERT_EQ(cell, loadedCell);
    }
}
