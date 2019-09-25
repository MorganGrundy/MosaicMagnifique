#include "photomosaicgenerator.h"

#include <vector>
#include <cmath>
#include <climits>
#include <opencv2/core/mat.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <QDebug>

cv::Mat PhotomosaicGenerator::generate(const cv::Mat &mainImage,
                                       const std::vector<cv::Mat> &library,
                                       QProgressDialog *progress)
{
    cv::Point cellSize(library.front().cols, library.front().rows);
    cv::Point gridSize(mainImage.cols / cellSize.x, mainImage.rows / cellSize.y);
    progress->setMaximum(gridSize.x * gridSize.y);
    progress->setLabelText("Finding best fits...");

    //Split main image into grid
    //Find best match for each cell in grid
    std::vector<std::vector<cv::Mat>> result(gridSize.x, std::vector<cv::Mat>(gridSize.y));
    for (int x = 0; x < gridSize.x; ++x)
    {
        for (int y = 0; y < gridSize.y; ++y)
        {
            int xStart = x * cellSize.x, xEnd = (x+1) * cellSize.x;
            int yStart = y * cellSize.y, yEnd = (y+1) * cellSize.y;

            cv::Mat cell = mainImage(cv::Range(yStart, yEnd), cv::Range(xStart, xEnd));
            result.at(x).at(y) = library.at(findBestImage(cell, library));
            progress->setValue(progress->value() + 1);
            if (progress->wasCanceled())
                return cv::Mat();
        }
    }

    //Combines all results into single image (mosaic)
    cv::Mat mosaic;
    std::vector<cv::Mat> mosaicRows(gridSize.x);
    for (size_t x = 0; x < gridSize.x; ++x)
        cv::vconcat(result.at(x), mosaicRows.at(x));
    cv::hconcat(mosaicRows, mosaic);

    return mosaic;
}

int PhotomosaicGenerator::findBestImage(const cv::Mat &mainImage,
                                        const std::vector<cv::Mat> &library)
{
    int bestFit = -1;
    long double bestVariant = LDBL_MAX;

    for (size_t i = 0; i < library.size(); ++i)
    {
        long double variant = 0;

        const uchar *p_main, *p_im;
        for(int row = 0; row < mainImage.rows && variant < bestVariant; ++row)
        {
            p_main = mainImage.ptr<uchar>(row);
            p_im = library.at(i).ptr<uchar>(row);
            for (int col = 0; col < mainImage.cols * mainImage.channels() && variant < bestVariant;
                 ++col)
            {
                variant += static_cast<long double>(sqrt(pow(p_main[col] - p_im[col], 2) +
                                                         pow(p_main[col + 1] - p_im[col + 1], 2) +
                                                         pow(p_main[col + 2] - p_im[col + 2], 2)));
            }
        }

        if (variant < bestVariant)
        {
            bestVariant = variant;
            bestFit = static_cast<int>(i);
        }
    }
    return bestFit;
}
