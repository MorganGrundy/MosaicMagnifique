#include "photomosaicgenerator.h"

#define _USE_MATH_DEFINES

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
            int index;
            if (mode == Mode::CIEDE2000)
                index = findBestFitCIEDE2000(cell, library);
            else
                index = findBestFitEuclidean(cell, library);
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

//Compares pixels in the cell against the library images
//Returns the index of the library image with the smallest difference
//Used for mode CIEDE2000
int PhotomosaicGenerator::findBestFitCIEDE2000(const cv::Mat &cell,
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
                const double k_L = 1.0, k_C = 1.0, k_H = 1.0;
                const double deg360InRad = degToRad(360.0);
                const double deg180InRad = degToRad(180.0);
                const double pow25To7 = 6103515625.0; //pow(25, 7)

                double C1 = sqrt((p_main[col + 1] * p_main[col + 1]) + (p_main[col + 2] *
                        p_main[col + 2]));
                double C2 = sqrt((p_im[col + 1] * p_im[col + 1]) + (p_im[col + 2] * p_im[col + 2]));
                double barC = (C1 + C2) / 2.0;

                double G = 0.5 * (1 - sqrt(pow(barC, 7) / (pow(barC, 7) + pow25To7)));

                double a1Prime = (1.0 + G) * p_main[col + 1];
                double a2Prime = (1.0 + G) * p_im[col + 1];

                double CPrime1 = sqrt((a1Prime * a1Prime) + (p_main[col + 2] * p_main[col + 2]));
                double CPrime2 = sqrt((a2Prime * a2Prime) + (p_im[col + 2] * p_im[col + 2]));

                double hPrime1;
                if (p_main[col + 2] == 0 && a1Prime == 0.0)
                    hPrime1 = 0.0;
                else
                {
                    hPrime1 = atan2(p_main[col + 2], a1Prime);
                    //This must be converted to a hue angle in degrees between 0 and 360 by
                    //addition of 2 pi to negative hue angles.
                    if (hPrime1 < 0)
                        hPrime1 += deg360InRad;
                }

                double hPrime2;
                if (p_im[col + 2] == 0 && a2Prime == 0.0)
                    hPrime2 = 0.0;
                else
                {
                    hPrime2 = atan2(p_im[col + 2], a2Prime);
                    //This must be converted to a hue angle in degrees between 0 and 360 by
                    //addition of 2pi to negative hue angles.
                    if (hPrime2 < 0)
                        hPrime2 += deg360InRad;
                }

                double deltaLPrime = p_im[col] - p_main[col];
                double deltaCPrime = CPrime2 - CPrime1;

                double deltahPrime;
                double CPrimeProduct = CPrime1 * CPrime2;
                if (CPrimeProduct == 0.0)
                    deltahPrime = 0;
                else
                {
                    //Avoid the fabs() call
                    deltahPrime = hPrime2 - hPrime1;
                    if (deltahPrime < -deg180InRad)
                        deltahPrime += deg360InRad;
                    else if (deltahPrime > deg180InRad)
                        deltahPrime -= deg360InRad;
                }

                double deltaHPrime = 2.0 * sqrt(CPrimeProduct) * sin(deltahPrime / 2.0);

                double barLPrime = (p_main[col] + p_im[col]) / 2.0;
                double barCPrime = (CPrime1 + CPrime2) / 2.0;

                double barhPrime, hPrimeSum = hPrime1 + hPrime2;
                if (CPrime1 * CPrime2 == 0.0)
                    barhPrime = hPrimeSum;
                else
                {
                    if (fabs(hPrime1 - hPrime2) <= deg180InRad)
                        barhPrime = hPrimeSum / 2.0;
                    else
                    {
                        if (hPrimeSum < deg360InRad)
                            barhPrime = (hPrimeSum + deg360InRad) / 2.0;
                        else
                            barhPrime = (hPrimeSum - deg360InRad) / 2.0;
                    }
                }

                double T = 1.0 - (0.17 * cos(barhPrime - degToRad(30.0))) +
                        (0.24 * cos(2.0 * barhPrime)) +
                        (0.32 * cos((3.0 * barhPrime) + degToRad(6.0))) -
                        (0.20 * cos((4.0 * barhPrime) - degToRad(63.0)));

                double deltaTheta = degToRad(30.0) *
                        exp(-pow((barhPrime - degToRad(275.0)) / degToRad(25.0), 2.0));

                double R_C = 2.0 * sqrt(pow(barCPrime, 7.0) / (pow(barCPrime, 7.0) + pow25To7));

                double S_L = 1 +((0.015 * pow(barLPrime - 50.0, 2.0)) /
                                 sqrt(20 + pow(barLPrime - 50.0, 2.0)));
                double S_C = 1 + (0.045 * barCPrime);
                double S_H = 1 + (0.015 * barCPrime * T);

                double R_T = (-sin(2.0 * deltaTheta)) * R_C;


                variant += static_cast<long double>(sqrt(pow(deltaLPrime / (k_L * S_L), 2.0) +
                                                         pow(deltaCPrime / (k_C * S_C), 2.0) +
                                                         pow(deltaHPrime / (k_H * S_H), 2.0) +
                                                         (R_T * (deltaCPrime / (k_C * S_C)) *
                                                          (deltaHPrime / (k_H * S_H)))));

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

double PhotomosaicGenerator::degToRad(const double deg)
{
    return (deg * M_PI) / 180;
}
