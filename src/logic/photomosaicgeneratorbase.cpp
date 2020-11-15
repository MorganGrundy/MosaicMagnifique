/*
	Copyright © 2018-2020, Morgan Grundy

	This file is part of Mosaic Magnifique.

    Mosaic Magnifique is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    Mosaic Magnifique is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with Mosaic Magnifique.  If not, see <https://www.gnu.org/licenses/>.
*/

#include "photomosaicgeneratorbase.h"

#include <opencv2/imgproc.hpp>
#include <QDebug>

#ifdef OPENCV_W_CUDA
#include <opencv2/cudawarping.hpp>
#include <opencv2/cudaimgproc.hpp>
#endif

#include "gridutility.h"
#include "gridbounds.h"

PhotomosaicGeneratorBase::PhotomosaicGeneratorBase()
    : m_progress{0}, m_wasCanceled{false}, m_img{}, m_lib{}, m_mode{Mode::RGB_EUCLIDEAN},
      m_repeatRange{0}, m_repeatAddition{0}
{}

PhotomosaicGeneratorBase::~PhotomosaicGeneratorBase() {}

//Sets main image
void PhotomosaicGeneratorBase::setMainImage(const cv::Mat &t_img)
{
    m_img = t_img;
}

//Sets library images
void PhotomosaicGeneratorBase::setLibrary(const std::vector<cv::Mat> &t_lib)
{
    m_lib = t_lib;
}

//Sets photomosaic mode
void PhotomosaicGeneratorBase::setMode(const Mode t_mode)
{
    m_mode = t_mode;
}

//Sets cell group
void PhotomosaicGeneratorBase::setCellGroup(const CellGroup &t_cellGroup)
{
    m_cells = t_cellGroup;
}

//Sets grid state
void PhotomosaicGeneratorBase::setGridState(const GridUtility::MosaicBestFit &t_gridState)
{
    m_bestFits = t_gridState;
}

//Sets repeat range and addition
void PhotomosaicGeneratorBase::setRepeat(int t_repeatRange, int t_repeatAddition)
{
    m_repeatRange = t_repeatRange;
    m_repeatAddition = t_repeatAddition;
}

//Generate best fits for Photomosaic cells
//Returns true if successful
bool PhotomosaicGeneratorBase::generateBestFits()
{
    qDebug() << "PhotomosaicGeneratorBase::generateBestFits() - "
             << "This is a base class";
    return false;
}

//Returns best fits
GridUtility::MosaicBestFit PhotomosaicGeneratorBase::getBestFits()
{
    return m_bestFits;
}

//Builds photomosaic from mosaic state
cv::Mat PhotomosaicGeneratorBase::buildPhotomosaic(const cv::Scalar &t_backgroundColour) const
{
    cv::Mat mosaic = cv::Mat(m_img.rows, m_img.cols, CV_8UC4, t_backgroundColour);
    cv::Mat mosaicStep;

    cv::Mat mosaicMask = cv::Mat::zeros(m_img.rows, m_img.cols, CV_8UC1);
    cv::Mat mosaicMaskStep;

    //Stores library images, halved at each size step
    std::vector<cv::Mat> libImg(m_lib);
    ImageUtility::addAlphaChannel(libImg);

    //For all size steps in results
    for (size_t step = 0; step < m_bestFits.size(); ++step)
    {
        //Halve library images on each size step
        if (step != 0)
            ImageUtility::batchResizeMat(libImg);

        const CellShape &normalCellShape = m_cells.getCell(step);

        mosaicStep = cv::Mat(m_img.rows, m_img.cols, CV_8UC4, t_backgroundColour);
        mosaicMaskStep = cv::Mat::zeros(m_img.rows, m_img.cols, CV_8UC1);

        //For all cells
        for (int y = -GridUtility::PAD_GRID; y < static_cast<int>(m_bestFits.at(step).size())
                                                     - GridUtility::PAD_GRID; ++y)
        {
            for (int x = -GridUtility::PAD_GRID;
                 x < static_cast<int>(m_bestFits.at(step).at(y + GridUtility::PAD_GRID).size())
                         - GridUtility::PAD_GRID; ++x)
            {
                const GridUtility::CellBestFit &cellData =
                    m_bestFits.at(step).at(static_cast<size_t>(y + GridUtility::PAD_GRID)).
                    at(static_cast<size_t>(x + GridUtility::PAD_GRID));

                //Skip invalid cells
                if (!cellData.has_value())
                {
                    continue;
                }

                //Gets bounds of cell in global space
                const cv::Rect cellGlobalBound = GridUtility::getRectAt(normalCellShape, x, y);

                //Bound cell in image area
                const int yStart = std::clamp(cellGlobalBound.tl().y, 0, m_img.rows);
                const int yEnd = std::clamp(cellGlobalBound.br().y, 0, m_img.rows);
                const int xStart = std::clamp(cellGlobalBound.tl().x, 0, m_img.cols);
                const int xEnd = std::clamp(cellGlobalBound.br().x, 0, m_img.cols);

                //Bounds of cell in local space
                const cv::Rect cellLocalBound(xStart - cellGlobalBound.x,
                                              yStart - cellGlobalBound.y,
                                              xEnd - xStart, yEnd - yStart);

                //Calculate if and how current cell is flipped
                const auto flipState = GridUtility::getFlipStateAt(normalCellShape, x, y);

                //Creates mask bounded
                const cv::Mat maskBounded(normalCellShape.getCellMask(flipState.horizontal,
                                                                      flipState.vertical),
                                          cellLocalBound);

                //Adds cells to mosaic step
                const cv::Mat libBounded(libImg.at(cellData.value()), cellLocalBound);
                libBounded.copyTo(mosaicStep(cv::Range(yStart, yEnd), cv::Range(xStart, xEnd)),
                                  maskBounded);

                //Adds cell mask to mosaic mask step
                cv::Mat mosaicMaskPart(mosaicMaskStep, cv::Range(yStart, yEnd),
                                       cv::Range(xStart, xEnd));
                cv::bitwise_or(mosaicMaskPart, maskBounded, mosaicMaskPart);

            }
        }

        //Combine mosaic step into mosaic
        if (step != 0)
        {
            cv::Mat mask;
            cv::bitwise_not(mosaicMask, mask);
            mosaicStep.copyTo(mosaic, mask);
            mosaicMaskStep.copyTo(mosaicMask, mask);
            //CopyTo is a shallow copy, clone to make a deep copy
            mosaic = mosaic.clone();
            mosaicMask = mosaicMask.clone();
        }
        else
        {
            mosaic = mosaicStep.clone();
            mosaicMask = mosaicMaskStep.clone();
        }
    }

    return mosaic;
}

//Returns maximum progress
int PhotomosaicGeneratorBase::getMaxProgress()
{
    return std::pow(4, m_bestFits.size() - 1) * (m_bestFits.size()) *
           m_bestFits.at(0).at(0).size() * m_bestFits.at(0).size();
}

//Cancel generation
void PhotomosaicGeneratorBase::cancel()
{
    m_wasCanceled = true;
}

//Converts colour space of main image and library images
//Resizes library based on detail level
//Returns results
std::pair<cv::Mat, std::vector<cv::Mat>> PhotomosaicGeneratorBase::resizeAndCvtColor()
{
    cv::Mat resultMain;
    std::vector<cv::Mat> result(m_lib.size(), cv::Mat());

    //Use INTER_AREA for decreasing, INTER_CUBIC for increasing
    cv::InterpolationFlags flags = (m_cells.getDetail() < 1) ? cv::INTER_AREA : cv::INTER_CUBIC;

#ifdef OPENCV_W_CUDA
    cv::cuda::Stream stream;
    //Main image
    if (m_mode == Mode::CIE76 || m_mode == Mode::CIEDE2000)
    {
        cv::cuda::GpuMat main, mainConverted;
        main.upload(m_img, stream);

        //Convert 8U [0..255] to 32F [0..1]
        //CUDA in-place convertTo doesn't work
        main.convertTo(mainConverted, CV_32F, 1 / 255.0, stream);
        //Convert BGR to Lab
        cv::cuda::cvtColor(mainConverted, main, cv::COLOR_BGR2Lab, 0, stream);

        main.download(resultMain, stream);
    }
    else
        //Convert 8U [0..255] to 32F [0..255]
        m_img.convertTo(resultMain, CV_32F);

    //Library image
    std::vector<cv::cuda::GpuMat> src(m_lib.size());
    //CUDA in-place convertTo doesn't work
    cv::cuda::GpuMat tmpConvertTo;
    for (size_t i = 0; i < m_lib.size(); ++i)
    {
        src.at(i).upload(m_lib.at(i), stream);

        //Resize image
        if (m_cells.getDetail() != 1)
            cv::cuda::resize(src.at(i), src.at(i),
                             cv::Size(std::round(m_cells.getCellSize(0, true)),
                                      std::round(m_cells.getCellSize(0, true))),
                             0, 0, flags, stream);

        if (m_mode == Mode::CIE76 || m_mode == Mode::CIEDE2000)
        {
            //Convert 8U [0..255] to 32F [0..1]
            src.at(i).convertTo(tmpConvertTo, CV_32F, 1 / 255.0, stream);
            //Convert BGR to Lab
            cv::cuda::cvtColor(tmpConvertTo, tmpConvertTo, cv::COLOR_BGR2Lab, 0, stream);
        }
        else
            //Convert 8U [0..255] to 32F [0..255]
            src.at(i).convertTo(tmpConvertTo, CV_32F, stream);

        tmpConvertTo.download(result.at(i), stream);
    }
    stream.waitForCompletion();
#else
    //Main image
    if (m_mode == Mode::CIE76 || m_mode == Mode::CIEDE2000)
    {
        //Convert 8U [0..255] to 32F [0..1]
        m_img.convertTo(resultMain, CV_32F, 1 / 255.0);
        //Convert BGR to Lab
        cv::cvtColor(resultMain, resultMain, cv::COLOR_BGR2Lab);
    }
    else
        //Convert 8U [0..255] to 32F [0..255]
        m_img.convertTo(resultMain, CV_32F);

    //Library image
    for (size_t i = 0; i < m_lib.size(); ++i)
    {
        if (m_cells.getDetail() != 1)
            cv::resize(m_lib.at(i), result.at(i),
                       cv::Size(std::round(m_cells.getCellSize(0, true)),
                                std::round(m_cells.getCellSize(0, true))), 0, 0, flags);
        else
            result.at(i) = m_lib.at(i).clone();

        if (m_mode == Mode::CIE76 || m_mode == Mode::CIEDE2000)
        {
            //Convert 8U [0..255] to 32F [0..1]
            result.at(i).convertTo(result.at(i), CV_32F, 1 / 255.0);
            //Convert BGR to Lab
            cv::cvtColor(result.at(i), result.at(i), cv::COLOR_BGR2Lab);
        }
        else
            //Convert 8U [0..255] to 32F [0..255]
            result.at(i).convertTo(result.at(i), CV_32F);
    }


#endif

    return {resultMain, result};
}

//Returns the cell image at given position and it's local bounds
std::pair<cv::Mat, cv::Rect> PhotomosaicGeneratorBase::getCellAt(
    const CellShape &t_cellShape, const CellShape &t_detailCellShape,
    const int x, const int y, const cv::Mat &t_image) const
{
    //Gets bounds of cell in global space
    const cv::Rect cellGlobalBound = GridUtility::getRectAt(t_cellShape, x, y);

    //Bound cell in image area
    const int yStart = std::clamp(cellGlobalBound.y, 0, t_image.rows);
    const int yEnd = std::clamp(cellGlobalBound.br().y, 0, t_image.rows);
    const int xStart = std::clamp(cellGlobalBound.x, 0, t_image.cols);
    const int xEnd = std::clamp(cellGlobalBound.br().x, 0, t_image.cols);

    //Bounds of cell in local space
    const cv::Rect cellLocalBound(xStart - cellGlobalBound.x, yStart - cellGlobalBound.y,
                                  xEnd - xStart, yEnd - yStart);

    //Copies visible part of main image to cell
    cv::Mat cell(cellGlobalBound.height, cellGlobalBound.width, t_image.type(), cv::Scalar(0));
    t_image(cv::Range(yStart, yEnd), cv::Range(xStart, xEnd)).copyTo(cell(cellLocalBound));

    //Resize image cell to detail level
    cell = ImageUtility::resizeImage(cell, t_detailCellShape.getSize(), t_detailCellShape.getSize(),
                                     ImageUtility::ResizeType::EXACT);

    //Resizes bounds of cell in local space to detail level
    cv::Rect detailCellLocalBound(
        std::min(t_detailCellShape.getSize() - 1,
                 static_cast<int>(cellLocalBound.x * m_cells.getDetail())),
        std::min(t_detailCellShape.getSize() - 1,
                 static_cast<int>(cellLocalBound.y * m_cells.getDetail())),
        std::max(1, static_cast<int>(cellLocalBound.width * m_cells.getDetail())),
        std::max(1, static_cast<int>(cellLocalBound.height * m_cells.getDetail())));

    detailCellLocalBound.width = std::min(t_detailCellShape.getSize() - detailCellLocalBound.x,
                                          detailCellLocalBound.width);
    detailCellLocalBound.height = std::min(t_detailCellShape.getSize() - detailCellLocalBound.y,
                                           detailCellLocalBound.height);

    return {cell, detailCellLocalBound};
}
