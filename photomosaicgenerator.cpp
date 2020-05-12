#include "photomosaicgenerator.h"

#include <cmath>
#include <vector>
#include <climits>
#include <algorithm>
#include <opencv2/core/mat.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <QDebug>

#ifdef OPENCV_W_CUDA
#include <opencv2/cudawarping.hpp>
#include <opencv2/cudaimgproc.hpp>
#endif

#ifdef CUDA
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <math_constants.h>

#include "cudaphotomosaicdata.h"
#endif

PhotomosaicGenerator::PhotomosaicGenerator(QWidget *t_parent)
    : QProgressDialog{t_parent}, m_img{}, m_lib{}, m_detail{100}, m_mode{Mode::RGB_EUCLIDEAN},
      m_repeatRange{0}, m_repeatAddition{0}
{
    setWindowModality(Qt::WindowModal);
}

PhotomosaicGenerator::~PhotomosaicGenerator() {}

void PhotomosaicGenerator::setMainImage(const cv::Mat &t_img)
{
    m_img = t_img;
}

void PhotomosaicGenerator::setLibrary(const std::vector<cv::Mat> &t_lib)
{
    m_lib = t_lib;
}

void PhotomosaicGenerator::setDetail(const int t_detail)
{
    m_detail = (t_detail < 1) ? 0.01 : t_detail / 100.0;
}

void PhotomosaicGenerator::setMode(const Mode t_mode)
{
    m_mode = t_mode;
}

void PhotomosaicGenerator::setCellShape(const CellShape &t_cellShape)
{
    m_cellShape = t_cellShape;
}

void PhotomosaicGenerator::setRepeat(int t_repeatRange, int t_repeatAddition)
{
    m_repeatRange = t_repeatRange;
    m_repeatAddition = t_repeatAddition;
}

//Resizes main image and library images based on detail level
//Also converts images to Lab colour space if needed
//Returns results
std::pair<cv::Mat, std::vector<cv::Mat>> PhotomosaicGenerator::resizeAndCvtColor()
{
    cv::Mat resultMain;
    std::vector<cv::Mat> result(m_lib.size(), cv::Mat());

    //Use INTER_AREA for decreasing, INTER_CUBIC for increasing
    cv::InterpolationFlags flags = (m_detail < 1) ? cv::INTER_AREA : cv::INTER_CUBIC;

#ifdef OPENCV_W_CUDA
    cv::cuda::Stream stream;
    //Main image
    cv::cuda::GpuMat mainSrc, mainDst;
    mainSrc.upload(m_img, stream);
    cv::cuda::resize(mainSrc, mainDst, cv::Size(static_cast<int>(m_detail * mainSrc.cols),
                                                static_cast<int>(m_detail * mainSrc.rows)),
                     0, 0, flags, stream);
    if (m_mode == Mode::CIE76 || m_mode == Mode::CIEDE2000)
        cv::cuda::cvtColor(mainDst, mainDst, cv::COLOR_BGR2Lab, 0, stream);
    mainDst.download(resultMain, stream);

    //Library image
    std::vector<cv::cuda::GpuMat> src(m_lib.size()), dst(m_lib.size());
    for (size_t i = 0; i < m_lib.size(); ++i)
    {
        //Resize image
        src.at(i).upload(m_lib.at(i), stream);
        cv::cuda::resize(src.at(i), dst.at(i),
                         cv::Size(static_cast<int>(m_detail * src.at(i).cols),
                                  static_cast<int>(m_detail * src.at(i).rows)),
                         0, 0, flags, stream);
        if (m_mode == Mode::CIE76 || m_mode == Mode::CIEDE2000)
            cv::cuda::cvtColor(dst.at(i), dst.at(i), cv::COLOR_BGR2Lab, 0, stream);
        dst.at(i).download(result.at(i), stream);
    }
    stream.waitForCompletion();
#else
    //Main image
    cv::resize(m_img, resultMain, cv::Size(static_cast<int>(m_detail * m_img.cols),
                                                   static_cast<int>(m_detail * m_img.rows)),
                                          0, 0, flags);
    cv::cvtColor(resultMain, resultMain, cv::COLOR_BGR2Lab);
    //Library image
    for (size_t i = 0; i < m_lib.size(); ++i)
    {
        cv::resize(m_lib.at(i), result.at(i),
                   cv::Size(static_cast<int>(m_detail * m_lib.at(i).cols),
                            static_cast<int>(m_detail * m_lib.at(i).rows)), 0, 0, flags);
        cv::cvtColor(result.at(i), result.at(i), cv::COLOR_BGR2Lab);
    }


#endif

    return std::make_pair(resultMain, result);
}

#ifdef CUDA

size_t differenceGPU(CUDAPhotomosaicData &photomosaicData);

//Returns a Photomosaic of the main image made of the library images
//Generates using CUDA
cv::Mat PhotomosaicGenerator::cudaGenerate()
{
    const bool padGrid = !m_cellShape.empty();
    //Resizes main image and library based on detail level and converts colour space
    auto [resizedImg, resizedLib] = resizeAndCvtColor();

    //If no cell shape provided then use default square cell, else resize given cell shape
    CellShape resizedCellShape;
    if (m_cellShape.empty())
        resizedCellShape = CellShape(cv::Mat(resizedLib.front().rows, resizedLib.front().cols,
                                             CV_8UC1, cv::Scalar(255)));
    else
        resizedCellShape = m_cellShape.resized(resizedLib.front().cols, resizedLib.front().rows);

    //Library images are square so rows == cols
    const int cellSize = resizedLib.front().cols;
    const cv::Point gridSize(resizedImg.cols / resizedCellShape.getColSpacing() + padGrid,
                             resizedImg.rows / resizedCellShape.getRowSpacing() + padGrid);

    setMaximum(gridSize.x * gridSize.y);
    setValue(0);
    setLabelText("Finding best fits...");

    //Allocate memory on GPU and copy data from CPU to GPU
    CUDAPhotomosaicData photomosaicData(cellSize, resizedLib.front().channels(),
                                        gridSize.x, gridSize.y, resizedLib.size(),
                                        m_mode != PhotomosaicGenerator::Mode::CIEDE2000,
                                        m_repeatRange, m_repeatAddition);
    if (!photomosaicData.mallocData())
        return cv::Mat();

    //Move library images to GPU
    photomosaicData.setLibraryImages(resizedLib);

    //Move mask image to GPU
    photomosaicData.setMaskImage(resizedCellShape.getCellMask());

    //Split main image into grid
    //Find best match for each cell in grid
    std::vector<std::vector<cv::Mat>> result(static_cast<size_t>(gridSize.x),
                                             std::vector<cv::Mat>(static_cast<size_t>(gridSize.y)));
    std::vector<std::vector<size_t>> grid(static_cast<size_t>(gridSize.x),
                                          std::vector<size_t>(static_cast<size_t>(gridSize.y), 0));
    cv::Mat cell(cellSize, cellSize, m_img.type(), cv::Scalar(0));
    for (int x = -static_cast<int>(padGrid); x < gridSize.x - padGrid; ++x)
    {
        for (int y = -static_cast<int>(padGrid); y < gridSize.y - padGrid; ++y)
        {
            //If user hits cancel in QProgressDialog then return empty mat
            if (wasCanceled())
                return cv::Mat();

            const cv::Rect unboundedRect = resizedCellShape.getRectAt(x, y);

            //Cell bounded positions (in mosaic area)
            const int yStart = std::clamp(unboundedRect.tl().y, 0, resizedImg.rows);
            const int yEnd = std::clamp(unboundedRect.br().y, 0, resizedImg.rows);
            const int xStart = std::clamp(unboundedRect.tl().x, 0, resizedImg.cols);
            const int xEnd = std::clamp(unboundedRect.br().x, 0, resizedImg.cols);

            //Cell completely out of bounds, just skip
            if (yStart == yEnd || xStart == xEnd)
            {
                result.at(static_cast<size_t>(x + padGrid)).at(static_cast<size_t>(y + padGrid)) =
                        m_lib.front();
                setValue(value() + 1);
                continue;
            }
            const size_t index = (y + padGrid) * gridSize.x + (x + padGrid);

            size_t targetArea[4] = {static_cast<size_t>(yStart - unboundedRect.y),
                                    static_cast<size_t>(yEnd - unboundedRect.y),
                                    static_cast<size_t>(xStart - unboundedRect.x),
                                    static_cast<size_t>(xEnd - unboundedRect.x)};
            photomosaicData.setTargetArea(targetArea, index);

            //Copies visible part of main image to cell
            resizedImg(cv::Range(yStart, yEnd), cv::Range(xStart, xEnd)).
                    copyTo(cell(cv::Range(static_cast<int>(targetArea[0]),
                                          static_cast<int>(targetArea[1])),
                                cv::Range(static_cast<int>(targetArea[2]),
                                          static_cast<int>(targetArea[3]))));
            photomosaicData.setCellImage(cell, index);
        }
    }

    //Calculate differences
    differenceGPU(photomosaicData);
    size_t *results = photomosaicData.getResults();

    for (size_t x = 0; x < static_cast<size_t>(gridSize.x); ++x)
    {
        for (size_t y = 0; y < static_cast<size_t>(gridSize.y); ++y)
        {
            const size_t index = y * gridSize.x + x;
            if (results[index] >= resizedLib.size())
            {
                qDebug() << "Error: Failed to find a best fit";
                result.at(x).at(y) = m_lib.front();
                setValue(value() + 1);
                continue;
            }

            grid.at(x).at(y) = results[index];
            result.at(x).at(y) = m_lib.at(results[index]);
            setValue(value() + 1);
        }
    }

    //Deallocate memory on GPU and CPU
    photomosaicData.freeData();

    //Combines all results into single image (mosaic)
    return combineResults(gridSize, result);
}
#endif

//Returns a Photomosaic of the main image made of the library images
cv::Mat PhotomosaicGenerator::generate()
{
    bool padGrid = !m_cellShape.empty();
    //Resizes main image and library based on detail level and converts colour space
    auto [resizedImg, resizedLib] = resizeAndCvtColor();

    //If no cell shape provided then use default square cell, else resize given cell shape
    CellShape resizedCellShape;
    if (m_cellShape.empty())
        resizedCellShape = CellShape(cv::Mat(resizedLib.front().rows,
                                             resizedLib.front().cols,
                                             CV_8UC1, cv::Scalar(255)));
    else
        resizedCellShape = m_cellShape.resized(resizedLib.front().cols,
                                               resizedLib.front().rows);

    const cv::Point cellSize(resizedLib.front().cols, resizedLib.front().rows);
    const cv::Point gridSize(resizedImg.cols / resizedCellShape.getColSpacing() + padGrid,
                             resizedImg.rows / resizedCellShape.getRowSpacing() + padGrid);

    setMaximum(gridSize.x * gridSize.y);
    setValue(0);
    setLabelText("Finding best fits...");

    //Split main image into grid
    //Find best match for each cell in grid
    std::vector<std::vector<cv::Mat>> result(static_cast<size_t>(gridSize.x),
                                             std::vector<cv::Mat>(static_cast<size_t>(gridSize.y)));
    std::vector<std::vector<size_t>> grid(static_cast<size_t>(gridSize.x),
                                          std::vector<size_t>(static_cast<size_t>(gridSize.y), 0));
    cv::Mat cell(cellSize.y, cellSize.x, m_img.type(), cv::Scalar(0));
    for (int x = -static_cast<int>(padGrid); x < gridSize.x - padGrid; ++x)
    {
        for (int y = -static_cast<int>(padGrid); y < gridSize.y - padGrid; ++y)
        {
            //If user hits cancel in QProgressDialog then return empty mat
            if (wasCanceled())
                return cv::Mat();

            const cv::Rect unboundedRect = resizedCellShape.getRectAt(x, y);

            //Cell bounded positions (in mosaic area)
            const int yStart = std::clamp(unboundedRect.tl().y, 0, resizedImg.rows);
            const int yEnd = std::clamp(unboundedRect.br().y, 0, resizedImg.rows);
            const int xStart = std::clamp(unboundedRect.tl().x, 0, resizedImg.cols);
            const int xEnd = std::clamp(unboundedRect.br().x, 0, resizedImg.cols);

            //Cell completely out of bounds, just skip
            if (yStart == yEnd || xStart == xEnd)
            {
                result.at(static_cast<size_t>(x + padGrid)).at(static_cast<size_t>(y + padGrid)) =
                        m_lib.front();
                setValue(value() + 1);
                continue;
            }

            //Copies visible part of main image to cell
            resizedImg(cv::Range(yStart, yEnd), cv::Range(xStart, xEnd)).
                    copyTo(cell(cv::Range(yStart - unboundedRect.y, yEnd - unboundedRect.y),
                                cv::Range(xStart - unboundedRect.x, xEnd - unboundedRect.x)));

            //Calculate repeat value of each library image in repeat range
            const std::map<size_t, int> repeats = calculateRepeats(grid, gridSize, x + padGrid,
                                                                   y + padGrid);

            int index = -1;
            if (m_mode == Mode::CIEDE2000)
                index = findBestFitCIEDE2000(cell, resizedCellShape.getCellMask(), resizedLib,
                                             repeats,
                                             yStart - unboundedRect.y, yEnd - unboundedRect.y,
                                             xStart - unboundedRect.x, xEnd - unboundedRect.x);
            else
                index = findBestFitEuclidean(cell, resizedCellShape.getCellMask(), resizedLib,
                                             repeats,
                                             yStart - unboundedRect.y, yEnd - unboundedRect.y,
                                             xStart - unboundedRect.x, xEnd - unboundedRect.x);
            if (index < 0 || index >= static_cast<int>(resizedLib.size()))
            {
                qDebug() << "Failed to find a best fit";
                result.at(static_cast<size_t>(x + padGrid)).at(static_cast<size_t>(y + padGrid)) =
                        m_lib.front();
                setValue(value() + 1);
                continue;
            }

            grid.at(static_cast<size_t>(x + padGrid)).at(static_cast<size_t>(y + padGrid)) =
                    static_cast<size_t>(index);
            result.at(static_cast<size_t>(x + padGrid)).at(static_cast<size_t>(y + padGrid)) =
                    m_lib.at(static_cast<size_t>(index));
            setValue(value() + 1);
        }
    }

    //Combines all results into single image (mosaic)
    return combineResults(gridSize, result);
}

//Calculates the repeat value of each library image in repeat range around x,y
//Only needs to look at first half of cells as the latter half are not yet used
std::map<size_t, int> PhotomosaicGenerator::calculateRepeats(
        const std::vector<std::vector<size_t>> &grid, const cv::Point &gridSize,
        const int x, const int y) const
{
    std::map<size_t, int> repeats;
    const int repeatStartX = std::clamp(x - m_repeatRange, 0, gridSize.x);
    const int repeatStartY = std::clamp(y - m_repeatRange, 0, gridSize.y);
    const int repeatEndX = std::clamp(x + m_repeatRange, 0, gridSize.x - 1);

    //Looks at cells above the current cell
    for (int repeatY = repeatStartY; repeatY < std::clamp(y, 0, gridSize.y); ++repeatY)
    {
        for (int repeatX = repeatStartX; repeatX <= repeatEndX; ++repeatX)
        {
            const auto it = repeats.find(grid.at(static_cast<size_t>(repeatX)).
                                   at(static_cast<size_t>(repeatY)));
            if (it != repeats.end())
                it->second += m_repeatAddition;
            else
                repeats.emplace(grid.at(static_cast<size_t>(repeatX)).
                                at(static_cast<size_t>(repeatY)), m_repeatAddition);
        }
    }

    //Looks at cells directly to the left of current cell
    for (int repeatX = repeatStartX; repeatX < x; ++repeatX)
    {
        const auto it = repeats.find(grid.at(static_cast<size_t>(repeatX)).
                               at(static_cast<size_t>(y)));
        if (it != repeats.end())
            it->second += m_repeatAddition;
        else
            repeats.emplace(grid.at(static_cast<size_t>(repeatX)).
                            at(static_cast<size_t>(y)), m_repeatAddition);
    }
    return repeats;
}

//Compares pixels in the cell against the library images
//Returns the index of the library image with the smallest difference
//Used for mode RGB_EUCLIDEAN and CIE76
//(CIE76 is just a euclidean formulae in a different colour space)
int PhotomosaicGenerator::findBestFitEuclidean(const cv::Mat &cell, const cv::Mat &mask,
                                               const std::vector<cv::Mat> &library,
                                               const std::map<size_t, int> &repeats,
                                               const int yStart, const int yEnd,
                                               const int xStart, const int xEnd) const
{
    int bestFit = -1;
    long double bestVariant = std::numeric_limits<long double>::max();

    for (size_t i = 0; i < library.size(); ++i)
    {
        const auto it = repeats.find(i);
        long double variant = (it != repeats.end()) ? it->second : 0;

        //For cell and library image compare the corresponding pixels
        //Sum all pixel differences for total image difference
        const uchar *p_main, *p_im, *p_mask;
        for (int row = yStart; row < yEnd && row < cell.rows && variant < bestVariant; ++row)
        {
            p_main = cell.ptr<uchar>(row);
            p_im = library.at(i).ptr<uchar>(row);
            p_mask = mask.ptr<uchar>(row);
            for (int col = xStart * cell.channels();
                 col < xEnd * cell.channels() && variant < bestVariant;
                 col += cell.channels())
            {
                if (p_mask[col / cell.channels()] != 0)
                    variant += static_cast<long double>(
                                sqrt(pow(p_main[col] - p_im[col], 2) +
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
int PhotomosaicGenerator::findBestFitCIEDE2000(const cv::Mat &cell, const cv::Mat &mask,
                                               const std::vector<cv::Mat> &library,
                                               const std::map<size_t, int> &repeats,
                                               const int yStart, const int yEnd,
                                               const int xStart, const int xEnd) const
{
    int bestFit = -1;
    long double bestVariant = std::numeric_limits<long double>::max();

    for (size_t i = 0; i < library.size(); ++i)
    {
        const auto it = repeats.find(i);
        long double variant = (it != repeats.end()) ? it->second : 0;

        //For cell and library image compare the corresponding pixels
        //Sum all pixel differences for total image difference
        const uchar *p_main, *p_im, *p_mask;
        for (int row = yStart; row < yEnd && row < cell.rows && variant < bestVariant; ++row)
        {
            p_main = cell.ptr<uchar>(row);
            p_im = library.at(i).ptr<uchar>(row);
            p_mask = mask.ptr<uchar>(row);
            for (int col = xStart * cell.channels();
                 col < xEnd * cell.channels() && variant < bestVariant;
                 col += cell.channels())
            {
                if (p_mask[col / cell.channels()] == 0)
                    continue;

                const double k_L = 1.0, k_C = 1.0, k_H = 1.0;
                const double deg360InRad = degToRad(360.0);
                const double deg180InRad = degToRad(180.0);
                const double pow25To7 = 6103515625.0; //pow(25, 7)

                const double C1 = sqrt((p_main[col + 1] * p_main[col + 1]) +
                        (p_main[col + 2] * p_main[col + 2]));
                const double C2 = sqrt((p_im[col + 1] * p_im[col + 1]) +
                        (p_im[col + 2] * p_im[col + 2]));
                const double barC = (C1 + C2) / 2.0;

                const double G = 0.5 * (1 - sqrt(pow(barC, 7) / (pow(barC, 7) + pow25To7)));

                const double a1Prime = (1.0 + G) * p_main[col + 1];
                const double a2Prime = (1.0 + G) * p_im[col + 1];

                const double CPrime1 = sqrt((a1Prime * a1Prime) +
                                            (p_main[col + 2] * p_main[col + 2]));
                const double CPrime2 = sqrt((a2Prime * a2Prime) +(p_im[col + 2] * p_im[col + 2]));

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

                const double deltaLPrime = p_im[col] - p_main[col];
                const double deltaCPrime = CPrime2 - CPrime1;

                double deltahPrime;
                const double CPrimeProduct = CPrime1 * CPrime2;
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

                const double deltaHPrime = 2.0 * sqrt(CPrimeProduct) * sin(deltahPrime / 2.0);

                const double barLPrime = (p_main[col] + p_im[col]) / 2.0;
                const double barCPrime = (CPrime1 + CPrime2) / 2.0;

                double barhPrime;
                const double hPrimeSum = hPrime1 + hPrime2;
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

                const double T = 1.0 - (0.17 * cos(barhPrime - degToRad(30.0))) +
                        (0.24 * cos(2.0 * barhPrime)) +
                        (0.32 * cos((3.0 * barhPrime) + degToRad(6.0))) -
                        (0.20 * cos((4.0 * barhPrime) - degToRad(63.0)));

                const double deltaTheta = degToRad(30.0) *
                        exp(-pow((barhPrime - degToRad(275.0)) / degToRad(25.0), 2.0));

                const double R_C = 2.0 * sqrt(pow(barCPrime, 7.0) /
                                              (pow(barCPrime, 7.0) + pow25To7));

                const double S_L = 1 + ((0.015 * pow(barLPrime - 50.0, 2.0)) /
                                        sqrt(20 + pow(barLPrime - 50.0, 2.0)));
                const double S_C = 1 + (0.045 * barCPrime);
                const double S_H = 1 + (0.015 * barCPrime * T);

                const double R_T = (-sin(2.0 * deltaTheta)) * R_C;


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

//Converts degrees to radians
double PhotomosaicGenerator::degToRad(const double deg) const
{
    return (deg * M_PI) / 180;
}

//Combines results into a Photomosaic
cv::Mat PhotomosaicGenerator::combineResults(const cv::Point gridSize,
                                             const std::vector<std::vector<cv::Mat>> &result)
{
    cv::Mat mosaic;
    if (m_cellShape.empty())
    {
        std::vector<cv::Mat> mosaicRows(static_cast<size_t>(gridSize.x));
        for (size_t x = 0; x < static_cast<size_t>(gridSize.x); ++x)
            cv::vconcat(result.at(x), mosaicRows.at(x));
        cv::hconcat(mosaicRows, mosaic);
    }
    else
    {
        const bool padGrid = !m_cellShape.empty();

        //Resizes cell shape to size of library images
        CellShape resizedCellShape = m_cellShape.resized(m_lib.front().cols, m_lib.front().rows);
        //Convert cell mask to grayscale (need single channel for use in copyTo()

        mosaic = cv::Mat((gridSize.y - padGrid) * resizedCellShape.getRowSpacing(),
                         (gridSize.x - padGrid) * resizedCellShape.getColSpacing(), m_img.type(),
                         cv::Scalar(0));

        //For all cells
        for (int y = -static_cast<int>(padGrid); y < gridSize.y - padGrid; ++y)
        {
            for (int x = -static_cast<int>(padGrid); x < gridSize.x - padGrid; ++x)
            {
                const cv::Rect unboundedRect = resizedCellShape.getRectAt(x, y);

                //Cell bounded positions (in mosaic area)
                const int yStart = std::clamp(unboundedRect.tl().y, 0, mosaic.rows);
                const int yEnd = std::clamp(unboundedRect.br().y, 0, mosaic.rows);
                const int xStart = std::clamp(unboundedRect.tl().x, 0, mosaic.cols);
                const int xEnd = std::clamp(unboundedRect.br().x, 0, mosaic.cols);

                //Cell completely out of bounds, just skip
                if (yStart == yEnd || xStart == xEnd)
                    continue;

                //Creates a mat that is the cell area actually visible in mosaic
                cv::Mat cellBounded(result.at(static_cast<size_t>(x + padGrid)).
                                    at(static_cast<size_t>(y + padGrid)),
                                    cv::Range(yStart - unboundedRect.y, yEnd - unboundedRect.y),
                                    cv::Range(xStart - unboundedRect.x, xEnd - unboundedRect.x));

                //Creates mask bounded same as cell
                cv::Mat maskBounded(resizedCellShape.getCellMask(),
                                    cv::Range(yStart - unboundedRect.y, yEnd - unboundedRect.y),
                                    cv::Range(xStart - unboundedRect.x, xEnd - unboundedRect.x));

                //Copys cell to mosaic using mask
                cellBounded.copyTo(mosaic(cv::Range(yStart, yEnd), cv::Range(xStart, xEnd)),
                                   maskBounded);
            }
        }
    }

    return mosaic;
}
