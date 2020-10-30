#ifndef TST_IMAGELIBRARY_H
#define TST_IMAGELIBRARY_H

#include <gtest/gtest.h>
#include <gmock/gmock-matchers.h>
#include <QDir>
#include <QFileInfo>

#include "imagelibrary.h"
#include "testutility.h"

//Creates an image library, saves it to a file
//Loads all test libraries (with different .mil versions), and then compares it
TEST(ImageLibrary, Static_SaveAndLoad)
{
    //Create image library
    const size_t imageSize = 64;
    const size_t noLibImages = 10;
    ImageLibrary lib(imageSize);
    for (size_t i = 0; i < noLibImages; ++i)
        lib.addImage(cv::Mat(imageSize, imageSize, CV_8UC3, cv::Scalar(50)));

    //Check if test for current image library version exists
    QDir libFolder(QDir::currentPath() + "/testcases/imagelibrary");
    QFileInfo currentVersionLib(libFolder, "lib-v" + QString::number(static_cast<uint>(
                                               ImageLibrary::MIL_VERSION)) + ".mil");

    if (!currentVersionLib.exists() || !currentVersionLib.isFile())
    {
        //Save test lib for current version
        if (!libFolder.exists())
            libFolder.mkpath(".");
        lib.saveToFile(currentVersionLib.filePath());
    }

    //Get all test libs
    const QStringList testLibs = libFolder.entryList(QDir::Filter::Files);

    //Iterate over test cases
    for (const auto &testLib: testLibs)
    {
        //Skip random test
        if (testLib.contains("lib-v"))
        {
            //Load library from file
            ImageLibrary loadedLib(imageSize);
            loadedLib.loadFromFile(libFolder.path() + "/" + testLib);

            //Compare library
            ASSERT_EQ(lib, loadedLib) << "File: " << testLib.toStdString();
        }
    }
}

//Creates a random image library, saves it to a file, loads it, and then compares it
TEST(ImageLibrary, Random_SaveAndLoad)
{
    srand(static_cast<unsigned int>(time(NULL)));

    const size_t imageSize = 64;
    const size_t noLibImages = 10;

    //Test multiple random libraries
    for (size_t i = 0; i < 20; ++i)
    {
        //Create library
        ImageLibrary lib(imageSize);
        for (size_t i = 0; i < noLibImages; ++i)
            lib.addImage(TestUtility::createRandomImage(imageSize, imageSize));

        //Save library to file
        QDir libFolder(QDir::currentPath() + "/testcases/imagelibrary");
        if (!libFolder.exists())
            libFolder.mkpath(".");
        QString libFile = libFolder.path() + "/randLib.mil";
        lib.saveToFile(libFile);

        //Load library from file
        ImageLibrary loadedLib(imageSize);
        loadedLib.loadFromFile(libFile);

        //Compare library
        ASSERT_EQ(lib, loadedLib);
    }
}

#endif // TST_IMAGELIBRARY_H
