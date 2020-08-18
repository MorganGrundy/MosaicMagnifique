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

#include "mainwindow.h"
#include "ui_mainwindow.h"

#include <QFileDialog>
#include <memory>
#include <QStringList>
#include <QPixmap>
#include <QImage>
#include <QImageReader>
#include <QDebug>
#include <QMessageBox>
#include <QPainter>
#include <opencv2/core/mat.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <chrono>

#include "imageutility.h"
#include "photomosaicgenerator.h"
#include "imageviewer.h"
#include "colourvisualisation.h"

#ifdef CUDA
#include <cuda_runtime.h>
#include "cudaphotomosaicdata.h"
#endif

#ifdef OPENCV_W_CUDA
#include <opencv2/cudawarping.hpp>
#endif

MainWindow::MainWindow(QWidget *t_parent)
    : QMainWindow{t_parent}, ui{new Ui::MainWindow}
{
    ui->setupUi(this);

    //Setup progress bar in status bar
    progressBar = new QProgressBar(ui->statusbar);
    progressBar->setRange(0, 0);
    progressBar->setValue(0);
    progressBar->setFormat("%v/%m");
    progressBar->setStyleSheet("QProgressBar {"
                               "border: 1px solid black;"
                               "border-radius: 10px;"
                               "text-align: center;"
                               "}"
                               "QProgressBar::chunk {"
                               "background-color: #05B8CC;"
                               "border-radius: 10px;"
                               "}");
    progressBar->setVisible(false);
    ui->statusbar->addPermanentWidget(progressBar);
    ui->statusbar->setSizeGripEnabled(false);

    connect(ui->tabWidget, SIGNAL(currentChanged(int)), this, SLOT(tabChanged(int)));

    photomosaicSizeRatio = static_cast<double>(ui->spinPhotomosaicWidth->value()) /
                           ui->spinPhotomosaicHeight->value();

    //Image Library Editor
    ui->imageLibraryEditor->setProgressBar(progressBar);
    connect(ui->imageLibraryEditor, SIGNAL(imageLibraryChanged(int)),
            this, SLOT(updateImageLibraryCount(int)));

    //Connects generator settings to appropriate methods
    connect(ui->buttonMainImage, SIGNAL(released()), this, SLOT(selectMainImage()));
    connect(ui->buttonCompareColours, SIGNAL(released()), this, SLOT(compareColours()));
    connect(ui->buttonPhotomosaicSizeLink, SIGNAL(released()), this, SLOT(photomosaicSizeLink()));
    connect(ui->spinPhotomosaicWidth, SIGNAL(valueChanged(int)), this,
            SLOT(photomosaicWidthChanged(int)));
    connect(ui->spinPhotomosaicHeight, SIGNAL(valueChanged(int)), this,
            SLOT(photomosaicHeightChanged(int)));
    connect(ui->buttonPhotomosaicSize, SIGNAL(released()), this, SLOT(loadImageSize()));

    connect(ui->spinDetail, SIGNAL(valueChanged(int)), this, SLOT(photomosaicDetailChanged(int)));

    connect(ui->spinCellSize, SIGNAL(valueChanged(int)), this, SLOT(cellSizeChanged(int)));
    connect(ui->spinMinCellSize, SIGNAL(valueChanged(int)),
            this, SLOT(minimumCellSizeChanged(int)));
    connect(ui->checkCellShape, SIGNAL(clicked(bool)), this, SLOT(enableCellShape(bool)));

    connect(ui->buttonGenerate, SIGNAL(released()), this, SLOT(generatePhotomosaic()));

#ifdef CUDA
    int deviceCount, device;
    int gpuDeviceCount = 0;
    cudaDeviceProp properties;
    cudaError_t cudaResultCode = cudaGetDeviceCount(&deviceCount);
    if (cudaResultCode != cudaSuccess)
        deviceCount = 0;

    //Check devices are not emulation only (9999)
    for (device = 0; device < deviceCount; ++device) {
        gpuErrchk(cudaGetDeviceProperties(&properties, device));
        if (properties.major != 9999)
        {
            ++gpuDeviceCount;
            //Add device name to combo box
            ui->comboCUDA->addItem(properties.name);
        }
    }

    //No devices so disable CUDA controls
    if (gpuDeviceCount == 0)
    {
        ui->checkCUDA->setChecked(false);
        ui->checkCUDA->setEnabled(false);
        ui->comboCUDA->setEnabled(false);
    }
    else
    {
        connect(ui->comboCUDA, SIGNAL(currentIndexChanged(int)),
                this, SLOT(CUDADeviceChanged(int)));
        CUDADeviceChanged(0);

        //Initialise CUDA
        int *deviceInit;
        gpuErrchk(cudaMalloc(&deviceInit, 0 * sizeof(int)));
    }
#else
    ui->checkCUDA->hide();
    ui->comboCUDA->hide();
#endif

    //Sets default cell size
    ui->spinCellSize->setValue(CellShape::DEFAULT_CELL_SIZE);

    //tabWidget starts on Generator Settings tab
    ui->tabWidget->setCurrentIndex(2);

    //Sets default detail level
    ui->spinDetail->setValue(100);
}

MainWindow::~MainWindow()
{
    delete ui;
}

//Updates grid preview in generator sett
void MainWindow::tabChanged(int t_index)
{
    //Generator settings tab
    if (t_index == 2)
    {
        ui->lineCellShape->setText(ui->cellShapeEditor->getCellShapeName());
        if (ui->checkCellShape->isChecked() && ui->cellShapeEditor->isCellShapeChanged())
        {
            ui->widgetGridPreview->getCellGroup().setCellShape(
                ui->cellShapeEditor->getCellShape()
                    .resized(ui->spinCellSize->value(), ui->spinCellSize->value()));

            ui->widgetGridPreview->updateGrid();
        }
    }
}

//Updates image library count in tab widget
void MainWindow::updateImageLibraryCount(int t_newSize)
{
    ui->tabWidget->setTabText(1, tr("Image Library (") + QString::number(t_newSize) + ")");
}

//Prompts user for a main image
void MainWindow::selectMainImage()
{
    QString filename = QFileDialog::getOpenFileName(this, tr("Select main image"), "",
                                                    "Image Files (*.bmp *.dib *.jpeg *.jpg "
                                                    "*.jpe *.jp2 *.png *.pbm *.pgm *.ppm "
                                                    "*.pxm *.pnm *.sr *.ras *.tiff *.tif "
                                                    "*.hdr *.pic)");
    if (!filename.isNull())
    {
        ui->lineMainImage->setText(filename);

        //Load main image and check is valid
        mainImage = cv::imread(filename.toStdString());
        if (mainImage.empty())
        {
            ui->widgetGridPreview->setBackground(cv::Mat());
            ui->widgetGridPreview->updateGrid();
            QMessageBox msgBox;
            msgBox.setText(tr("The main image \"") + ui->lineMainImage->text() +
                           tr("\" failed to load"));
            msgBox.exec();
            return;
        }

        ui->spinPhotomosaicHeight->blockSignals(true);
        ui->spinPhotomosaicWidth->blockSignals(true);
        ui->spinPhotomosaicHeight->setValue(mainImage.rows);
        ui->spinPhotomosaicWidth->setValue(mainImage.cols);
        ui->spinPhotomosaicHeight->blockSignals(false);
        ui->spinPhotomosaicWidth->blockSignals(false);
        photomosaicSizeRatio = static_cast<double>(mainImage.cols) / mainImage.rows;

        //Gives main image to grid preview
        ui->widgetGridPreview->setBackground(
                    ImageUtility::resizeImage(mainImage, mainImage.rows, mainImage.cols,
                                              ImageUtility::ResizeType::INCLUSIVE));

        ui->widgetGridPreview->updateGrid();
    }
}

//Opens colour visualisation window
void MainWindow::compareColours()
{
    if (ui->imageLibraryEditor->getImageLibrarySize() == 0 || mainImage.empty())
        return;

    ColourVisualisation *colourVisualisation =
        new ColourVisualisation(this, mainImage, ui->imageLibraryEditor->getImageLibrary());
    colourVisualisation->show();
}

//Changes icon of photomosaic size link button and saves ratio
void MainWindow::photomosaicSizeLink()
{
    if (ui->buttonPhotomosaicSizeLink->isChecked())
    {
        ui->buttonPhotomosaicSizeLink->setIcon(QIcon(":/img/LinkIcon.png"));
        photomosaicSizeRatio = static_cast<double>(ui->spinPhotomosaicWidth->value()) /
                ui->spinPhotomosaicHeight->value();
    }
    else
    {
        ui->buttonPhotomosaicSizeLink->setIcon(QIcon(":/img/UnlinkIcon.png"));
    }
}

//If link is active then when Photomosaic width changes updates height
void MainWindow::photomosaicWidthChanged(int i)
{
    if (ui->buttonPhotomosaicSizeLink->isChecked())
    {
        //Blocks signals while changing value to prevent infinite loop
        ui->spinPhotomosaicHeight->blockSignals(true);
        ui->spinPhotomosaicHeight->setValue(std::round(i / photomosaicSizeRatio));
        ui->spinPhotomosaicHeight->blockSignals(false);

        //Updates image size in grid preview
        if (!mainImage.empty())
        {
            ui->widgetGridPreview->setBackground(
                        ImageUtility::resizeImage(mainImage, ui->spinPhotomosaicHeight->value(),
                                                  ui->spinPhotomosaicWidth->value(),
                                                  ImageUtility::ResizeType::INCLUSIVE));
            ui->widgetGridPreview->updateGrid();
        }
    }
}

//If link is active then when Photomosaic height changes updates width
void MainWindow::photomosaicHeightChanged(int i)
{
    if (ui->buttonPhotomosaicSizeLink->isChecked())
    {
        //Blocks signals while changing value to prevent infinite loop
        ui->spinPhotomosaicWidth->blockSignals(true);
        ui->spinPhotomosaicWidth->setValue(std::floor(i * photomosaicSizeRatio));
        ui->spinPhotomosaicWidth->blockSignals(false);
    }

    //Updates image size in grid preview
    if (!mainImage.empty())
    {
        ui->widgetGridPreview->setBackground(
                    ImageUtility::resizeImage(mainImage, ui->spinPhotomosaicHeight->value(),
                                              ui->spinPhotomosaicWidth->value(),
                                              ImageUtility::ResizeType::INCLUSIVE));
        ui->widgetGridPreview->updateGrid();
    }
}

//If a main image has been entered sets the Photomosaic size spinboxes to the image size
void MainWindow::loadImageSize()
{
    if (!mainImage.empty())
    {
        //Blocks signals while changing value
        ui->spinPhotomosaicWidth->blockSignals(true);
        ui->spinPhotomosaicHeight->blockSignals(true);
        ui->spinPhotomosaicWidth->setValue(mainImage.cols);
        ui->spinPhotomosaicHeight->setValue(mainImage.rows);
        ui->spinPhotomosaicWidth->blockSignals(false);
        ui->spinPhotomosaicHeight->blockSignals(false);
        //Update size ratio
        photomosaicSizeRatio = static_cast<double>(ui->spinPhotomosaicWidth->value()) /
                ui->spinPhotomosaicHeight->value();

        //Resize main image to user entered size
        ui->widgetGridPreview->setBackground(
                    ImageUtility::resizeImage(mainImage, mainImage.rows, mainImage.cols,
                                              ImageUtility::ResizeType::INCLUSIVE));
        ui->widgetGridPreview->updateGrid();
    }
}

//Updates grid preview when detail level changes
void MainWindow::photomosaicDetailChanged(int /*i*/)
{
    clampDetail();

    ui->widgetGridPreview->getCellGroup().setDetail(ui->spinDetail->value());
    if (!mainImage.empty())
        ui->widgetGridPreview->updateGrid();
}

//Updates grid preview and minimum cell size spin box
void MainWindow::cellSizeChanged(int t_value)
{
    ui->spinMinCellSize->blockSignals(true);
    ui->spinMinCellSize->setMaximum(t_value);
    ui->spinMinCellSize->stepBy(0);
    ui->spinMinCellSize->blockSignals(false);

    clampDetail();

    if (ui->checkCellShape->isChecked())
    {
        ui->widgetGridPreview->getCellGroup().setCellShape(
            ui->cellShapeEditor->getCellShape().resized(t_value, t_value));
    }
    else
    {
        ui->widgetGridPreview->getCellGroup().setCellShape(CellShape(CellShape::DEFAULT_CELL_SIZE));
    }

    ui->widgetGridPreview->getCellGroup().setSizeSteps(ui->spinMinCellSize->getHalveSteps());

    ui->widgetGridPreview->updateGrid();
}

//Updates grid preview
void MainWindow::minimumCellSizeChanged(int /*t_value*/)
{
    clampDetail();

    ui->widgetGridPreview->getCellGroup().setSizeSteps(ui->spinMinCellSize->getHalveSteps());
    ui->widgetGridPreview->updateGrid();
}

//Enables/disables non-square cell shapes, GUI widgets for choosing
void MainWindow::enableCellShape(bool t_state)
{
    ui->lineCellShape->setEnabled(t_state);
    if (t_state)
    {
        ui->widgetGridPreview->getCellGroup().setCellShape(
            ui->cellShapeEditor->getCellShape().resized(ui->spinCellSize->value(),
                                                        ui->spinCellSize->value()));
    }
    else
    {
        ui->widgetGridPreview->getCellGroup().setCellShape(CellShape(ui->spinCellSize->value()));
    }
    ui->widgetGridPreview->updateGrid();
}

#ifdef CUDA
//Changes CUDA device
void MainWindow::CUDADeviceChanged(int t_index)
{
    gpuErrchk(cudaSetDevice(t_index));
}
#endif

//Attempts to generate and display a Photomosaic for current settings
void MainWindow::generatePhotomosaic()
{
    //Check library contains images
    if (ui->imageLibraryEditor->getImageLibrarySize() == 0)
    {
        QMessageBox msgBox;
        msgBox.setText(tr("The library is empty, please add some images"));
        msgBox.exec();
        return;
    }

    //Load main image and check is valid
    cv::Mat mainImage = cv::imread(ui->lineMainImage->text().toStdString());
    if (mainImage.empty())
    {
        QMessageBox msgBox;
        msgBox.setText(tr("The main image \"") + ui->lineMainImage->text()
                       + tr("\" failed to load"));
        msgBox.exec();
        return;
    }
    //Resize main image to user entered size
    mainImage = ImageUtility::resizeImage(mainImage, ui->spinPhotomosaicHeight->value(),
                                          ui->spinPhotomosaicWidth->value(),
                                          ImageUtility::ResizeType::INCLUSIVE);

    std::vector<cv::Mat> library = ui->imageLibraryEditor->getImageLibrary();

    if (library.front().cols != ui->spinCellSize->value())
        library = ImageUtility::batchResizeMat(library, ui->spinCellSize->value(),
                                               ui->spinCellSize->value(),
                                               ImageUtility::ResizeType::EXACT, progressBar);

    //Generate Photomosaic
    PhotomosaicGenerator generator(this);
    generator.setMainImage(mainImage);
    generator.setLibrary(library);
    if (ui->comboMode->currentText() == "RGB Euclidean")
        generator.setMode(PhotomosaicGenerator::Mode::RGB_EUCLIDEAN);
    else if (ui->comboMode->currentText() == "CIE76")
        generator.setMode(PhotomosaicGenerator::Mode::CIE76);
    else if (ui->comboMode->currentText() == "CIEDE2000")
        generator.setMode(PhotomosaicGenerator::Mode::CIEDE2000);

    generator.setCellGroup(ui->widgetGridPreview->getCellGroup());

    generator.setGridState(ui->widgetGridPreview->getGridState());

    generator.setRepeat(ui->spinRepeatRange->value(), ui->spinRepeatAddition->value());

    const auto startTime = std::chrono::high_resolution_clock::now();
#ifdef CUDA
    const cv::Mat mosaic = (ui->checkCUDA->isChecked()) ? generator.cudaGenerate()
                                                        : generator.generate();
#else
    const cv::Mat mosaic = generator.generate();
#endif
    const double duration = std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::high_resolution_clock::now() - startTime).count() / 1000.0;
    qDebug() << "Generator time: " << duration << "s";

    //Displays Photomosaic
    ImageViewer *imageViewer = new ImageViewer(this, mosaic, duration);
    imageViewer->show();
}

//Clamps detail level so that cell size never reaches 0px
//Returns if detail was clamped
void MainWindow::clampDetail()
{
    const int minCellSize = ui->spinMinCellSize->value();
    const double detailLevel = ui->spinDetail->value() / 100.0;
    if (std::floor(minCellSize * detailLevel) < 1)
    {
        const int minDetail = std::ceil(100.0 / minCellSize);
        ui->spinDetail->setValue(minDetail);
    }
}
