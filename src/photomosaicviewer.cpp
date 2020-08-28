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

#include "photomosaicviewer.h"
#include "ui_photomosaicviewer.h"

#include <QFileDialog>
#include <QLabel>
#include <opencv2/highgui.hpp>

#include "imageutility.h"
#include "customgraphicsview.h"

PhotomosaicViewer::PhotomosaicViewer(QWidget *t_parent, const cv::Mat &t_img,
                               const std::vector<cv::Mat> &t_lib, const CellGroup &t_cells,
                               const GridUtility::MosaicBestFit &t_mosaicState,
                               const double t_duration)
    : QMainWindow{t_parent}, ui{new Ui::PhotomosaicViewer}, m_img{t_img.clone()}, m_lib{t_lib},
    m_cells{t_cells}, m_mosaicState{t_mosaicState}
{
    ui->setupUi(this);

    //Connects buttons to appropriate methods
    connect(ui->saveButton, &QPushButton::released, this, &PhotomosaicViewer::savePhotomosaic);
    connect(ui->fitButton, &QPushButton::released,
            ui->graphicsView, &CustomGraphicsView::fitToView);

    if (t_duration > 0)
    {
        //Displays the generation time on statusbar
        QLabel *label = new QLabel();
        label->setText(tr("Generated in ") + QString::number(t_duration) + "s");
        ui->statusbar->addPermanentWidget(label);
    }

    buildPhotomosaic();

    //Adds image to view
    const QPixmap pixmap = ImageUtility::matToQPixmap(m_photomosaic);
    scene = new QGraphicsScene();
    scene->setItemIndexMethod(QGraphicsScene::NoIndex);
    scene->addPixmap(pixmap);
    ui->graphicsView->setScene(scene);
    ui->graphicsView->setSceneRect(pixmap.rect());
}

PhotomosaicViewer::PhotomosaicViewer(QWidget *t_parent)
    : QMainWindow{t_parent}, ui{new Ui::PhotomosaicViewer}
{
    ui->setupUi(this);
}

PhotomosaicViewer::~PhotomosaicViewer()
{
    delete ui;
}

//Allows user to save the Photomosaic as an image file
void PhotomosaicViewer::savePhotomosaic()
{
    QString filename = QFileDialog::getSaveFileName(this, tr("Save Photomosaic"), "",
                                                    "Image Files (*.bmp *.dib *.jpeg *.jpg "
                                                    "*.jpe *.jp2 *.png *.pbm *.pgm *.ppm "
                                                    "*.pxm *.pnm *.sr *.ras *.tiff *.tif "
                                                    "*.hdr *.pic)");
    cv::imwrite(filename.toStdString(), m_photomosaic);
}

//Builds photomosaic from mosaic state
void PhotomosaicViewer::buildPhotomosaic()
{
    cv::Mat mosaic = cv::Mat::zeros(m_img.rows, m_img.cols, m_img.type());
    cv::Mat mosaicStep;

    cv::Mat mosaicMask = cv::Mat::zeros(m_img.rows, m_img.cols, CV_8UC1);
    cv::Mat mosaicMaskStep;

    //Stores library images, halved at each size step
    std::vector<cv::Mat> libImg(m_lib);

    //For all size steps in results
    for (size_t step = 0; step < m_mosaicState.size(); ++step)
    {
        //Halve library images on each size step
        if (step != 0)
            ImageUtility::batchResizeMat(libImg);

        const CellShape &normalCellShape = m_cells.getCell(step);

        mosaicStep = cv::Mat::zeros(m_img.rows, m_img.cols, m_img.type());
        mosaicMaskStep = cv::Mat::zeros(m_img.rows, m_img.cols, CV_8UC1);

        //For all cells
        for (int y = -GridUtility::PAD_GRID; y < static_cast<int>(m_mosaicState.at(step).size())
                                                     - GridUtility::PAD_GRID; ++y)
        {
            for (int x = -GridUtility::PAD_GRID;
                 x < static_cast<int>(m_mosaicState.at(step).at(y + GridUtility::PAD_GRID).size())
                         - GridUtility::PAD_GRID; ++x)
            {
                const GridUtility::CellBestFit &cellData =
                    m_mosaicState.at(step).at(static_cast<size_t>(y + GridUtility::PAD_GRID)).
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
                const auto flipState = GridUtility::getFlipStateAt(normalCellShape, x, y,
                                                                   GridUtility::PAD_GRID);

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

    m_photomosaic = mosaic;
}
