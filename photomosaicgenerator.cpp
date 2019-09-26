#include "photomosaicgenerator.h"

#include <vector>
#include <cmath>
#include <climits>
#include <opencv2/core/mat.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <QDebug>

//Returns a Photomosaic of the main image made of the library images
//Mode is the colour difference formulae used
cv::Mat PhotomosaicGenerator::generate(cv::Mat &mainImage,
                                       const std::vector<cv::Mat> &library,
                                       Mode mode, QProgressDialog &progress)
{
    //If using a CIE mode converts all images to Lab colour space
    if (mode == Mode::CIE76 || mode == Mode::CIEDE2000)
    {
        progress.setMaximum(static_cast<int>(library.size()) + 1);
        progress.setValue(0);
        progress.setLabelText("Converting images to LAB colour space...");

        cv::cvtColor(mainImage, mainImage, cv::COLOR_BGR2Lab);
        progress.setValue(progress.value() + 1);

        for (auto image: library)
        {
            cv::cvtColor(image, image, cv::COLOR_BGR2Lab);
            progress.setValue(progress.value() + 1);
        }
    }

    cv::Point cellSize(library.front().cols, library.front().rows);
    cv::Point gridSize(mainImage.cols / cellSize.x, mainImage.rows / cellSize.y);

    progress.setMaximum(gridSize.x * gridSize.y);
    progress.setValue(0);
    progress.setLabelText("Finding best fits...");

    //Split main image into grid
    //Find best match for each cell in grid
    std::vector<std::vector<cv::Mat>> result(static_cast<size_t>(gridSize.x),
                                             std::vector<cv::Mat>(static_cast<size_t>(gridSize.y)));
    for (int x = 0; x < gridSize.x; ++x)
    {
        for (int y = 0; y < gridSize.y; ++y)
        {
            int xStart = x * cellSize.x, xEnd = (x+1) * cellSize.x;
            int yStart = y * cellSize.y, yEnd = (y+1) * cellSize.y;

            cv::Mat cell = mainImage(cv::Range(yStart, yEnd), cv::Range(xStart, xEnd));
            int index = findBestFitEuclidean(cell, library);
            if (index < 0 || index >= static_cast<int>(library.size()))
            {
                qDebug() << "Failed to find a best fit";
                index = 0;
            }

            result.at(static_cast<size_t>(x)).at(static_cast<size_t>(y)) =
                    library.at(static_cast<size_t>(index));
            progress.setValue(progress.value() + 1);
            if (progress.wasCanceled())
                return cv::Mat();
        }
    }

    //Combines all results into single image (mosaic)
    cv::Mat mosaic;
    std::vector<cv::Mat> mosaicRows(static_cast<size_t>(gridSize.x));
    for (size_t x = 0; x < static_cast<size_t>(gridSize.x); ++x)
        cv::vconcat(result.at(x), mosaicRows.at(x));
    cv::hconcat(mosaicRows, mosaic);

    if (mode == Mode::CIE76 || mode == Mode::CIEDE2000)
        cv::cvtColor(mosaic, mosaic, cv::COLOR_Lab2BGR);
    return mosaic;
}

//Compares pixels in the cell against the library images
//Returns the index of the library image with the smallest difference
//Used for mode RGB_EUCLIDEAN and CIE76
//(CIE76 is just a euclidean formulae in a different colour space)
int PhotomosaicGenerator::findBestFitEuclidean(const cv::Mat &cell,
                                        const std::vector<cv::Mat> &library)
{
    int bestFit = -1;
    long double bestVariant = LDBL_MAX;

    for (size_t i = 0; i < library.size(); ++i)
    {
        long double variant = 0;

        //For cell and library image compare the corresponding pixels
        //Sum all pixel differences for total image difference
        const uchar *p_main, *p_im;
        for(int row = 0; row < cell.rows && variant < bestVariant; ++row)
        {
            p_main = cell.ptr<uchar>(row);
            p_im = library.at(i).ptr<uchar>(row);
            for (int col = 0; col < cell.cols * cell.channels() && variant < bestVariant;
                 ++col)
            {
                variant += static_cast<long double>(sqrt(pow(p_main[col] - p_im[col], 2) +
                                                         pow(p_main[col + 1] - p_im[col + 1], 2) +
                                                         pow(p_main[col + 2] - p_im[col + 2], 2)));
            }
        }

        //If image difference is less than current lowest then replace
        if (variant < bestVariant)
        {
            bestVariant = variant;
            bestFit = static_cast<int>(i);
        }
    }
    return bestFit;
}
