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

#include "utilityfuncs.h"
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
    : QMainWindow(t_parent), ui(new Ui::MainWindow)
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

    cellShapeChanged = false;
    //Connects custom cell shapes tab to appropriate methods
    connect(ui->buttonSaveCustomCell, SIGNAL(released()), this, SLOT(saveCellShape()));
    connect(ui->buttonLoadCustomCell, SIGNAL(released()), this, SLOT(loadCellShape()));
    connect(ui->lineCustomCellName, SIGNAL(textChanged(const QString &)), this,
            SLOT(cellNameChanged(const QString &)));
    connect(ui->buttonCellMask, SIGNAL(released()), this, SLOT(selectCellMask()));
    //Cell spacing
    connect(ui->spinCustomCellSpacingCol, SIGNAL(valueChanged(int)), this,
            SLOT(cellSpacingColChanged(int)));
    connect(ui->spinCustomCellSpacingRow, SIGNAL(valueChanged(int)), this,
            SLOT(cellSpacingRowChanged(int)));
    //Cell alternate offset
    connect(ui->spinCustomCellAlternateColOffset, SIGNAL(valueChanged(int)), this,
            SLOT(cellAlternateColOffsetChanged(int)));
    connect(ui->spinCustomCellAlternateRowOffset, SIGNAL(valueChanged(int)), this,
            SLOT(cellAlternateRowOffsetChanged(int)));
    //Cell flipping
    connect(ui->checkCellColFlipH, SIGNAL(clicked(bool)), this,
            SLOT(cellColumnFlipHorizontalChanged(bool)));
    connect(ui->checkCellColFlipV, SIGNAL(clicked(bool)), this,
            SLOT(cellColumnFlipVerticalChanged(bool)));
    connect(ui->checkCellRowFlipH, SIGNAL(clicked(bool)), this,
            SLOT(cellRowFlipHorizontalChanged(bool)));
    connect(ui->checkCellRowFlipV, SIGNAL(clicked(bool)), this,
            SLOT(cellRowFlipVerticalChanged(bool)));
    //Cell alternate spacing
    connect(ui->checkCellAlternateRowSpacing, SIGNAL(toggled(bool)), this,
            SLOT(enableCellAlternateRowSpacing(bool)));
    connect(ui->checkCellAlternateColSpacing, SIGNAL(toggled(bool)), this,
            SLOT(enableCellAlternateColSpacing(bool)));
    connect(ui->spinCellAlternateRowSpacing, SIGNAL(valueChanged(int)), this,
            SLOT(cellAlternateRowSpacingChanged(int)));
    connect(ui->spinCellAlternateColSpacing, SIGNAL(valueChanged(int)), this,
            SLOT(cellAlternateColSpacingChanged(int)));

    //Setup image library list
    ui->listPhoto->setResizeMode(QListWidget::ResizeMode::Adjust);
    ui->listPhoto->setIconSize(ui->listPhoto->gridSize() - QSize(14, 14));
    imageSize = ui->spinLibCellSize->value();

    //Connects image library tab buttons to appropriate methods
    connect(ui->buttonAdd, SIGNAL(released()), this, SLOT(addImages()));
    connect(ui->buttonDelete, SIGNAL(released()), this, SLOT(deleteImages()));
    connect(ui->buttonLibCellSize, SIGNAL(released()), this, SLOT(updateCellSize()));
    connect(ui->buttonSave, SIGNAL(released()), this, SLOT(saveLibrary()));
    connect(ui->buttonLoad, SIGNAL(released()), this, SLOT(loadLibrary()));

    photomosaicSizeRatio = static_cast<double>(ui->spinPhotomosaicWidth->value()) /
            ui->spinPhotomosaicHeight->value();

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
    ui->spinCellSize->setValue(128);

    //tabWidget starts on Generator Settings tab
    ui->tabWidget->setCurrentIndex(2);

    ui->spinDetail->setValue(128);
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
        if (ui->checkCellShape->isChecked() && cellShapeChanged)
        {
            ui->widgetGridPreview->setCellShape(ui->widgetCellShapeViewer->getCellShape()
                                                .resized(ui->spinCellSize->value(),
                                                         ui->spinCellSize->value()));
            cellShapeChanged = false;
            ui->widgetGridPreview->updateGrid();
        }
    }
}

//Saves the custom cell shape to a file
void MainWindow::saveCellShape()
{
    CellShape &cellShape = ui->widgetCellShapeViewer->getCellShape();
    if (cellShape.getCellMask(0, 0).empty())
    {
        QMessageBox::information(this, tr("Failed to save custom cell shape"),
                                 tr("No cell mask was provided"));
        return;
    }

    //mcs = Mosaic Cell Shape
    QString filename = QFileDialog::getExistingDirectory(this, tr("Save custom cell shape"));

    if (!filename.isNull())
    {
        filename += '/' + ui->lineCustomCellName->text() + ".mcs";
        qDebug() << filename;
        QFile file(filename);
        file.open(QIODevice::WriteOnly);
        if (file.isWritable())
        {
            QDataStream out(&file);
            //Write header with "magic number" and version
            out << static_cast<quint32>(CellShape::MCS_MAGIC);
            out << static_cast<qint32>(CellShape::MCS_VERSION);

            out.setVersion(out.version());

            //Write cell mask and offsets
            out << cellShape;

            file.close();
        }
    }
}

//Loads a custom cell shape from a file for editing
void MainWindow::loadCellShape()
{
    //mcs = Mosaic Cell Shape
    QString filename = QFileDialog::getOpenFileName(this, tr("Select custom cell shape to load"),
                                                    "", tr("Mosaic Cell Shape") + " (*.mcs)");

    if (!filename.isNull())
    {
        QFile file(filename);
        file.open(QIODevice::ReadOnly);
        if (file.isReadable())
        {
            QDataStream in(&file);

            //Read and check magic number
            quint32 magic;
            in >> magic;
            if (magic != CellShape::MCS_MAGIC)
            {
                QMessageBox msgBox;
                msgBox.setText(filename + tr(" is not a valid .mcs file"));
                msgBox.exec();
                return;
            }

            //Read the version
            qint32 version;
            in >> version;
            if (version <= CellShape::MCS_VERSION && version >= 4)
                in.setVersion(in.version());
            else
            {
                QMessageBox msgBox;
                if (version < UtilityFuncs::VERSION_NO)
                    msgBox.setText(filename + tr(" uses an outdated file version"));
                else
                    msgBox.setText(filename + tr(" uses a newer file version"));
                msgBox.exec();
                return;
            }

            //Read cell mask and offsets
            CellShape cellShape;
            std::pair<CellShape &, const int> cellAndVersion(cellShape, version);
            in >> cellAndVersion;
            //Blocks signals to prevent grid update until all values loaded
            //Update cell spacing
            ui->spinCustomCellSpacingCol->blockSignals(true);
            ui->spinCustomCellSpacingRow->blockSignals(true);
            ui->spinCustomCellSpacingCol->setValue(cellShape.getColSpacing());
            ui->spinCustomCellSpacingRow->setValue(cellShape.getRowSpacing());
            ui->spinCustomCellSpacingCol->blockSignals(false);
            ui->spinCustomCellSpacingRow->blockSignals(false);

            //Update cell alternate spacing
            ui->checkCellAlternateColSpacing->setChecked(
                        cellShape.getAlternateColSpacing() != cellShape.getColSpacing());
            ui->checkCellAlternateRowSpacing->setChecked(
                        cellShape.getAlternateRowSpacing() != cellShape.getRowSpacing());
            ui->spinCellAlternateColSpacing->blockSignals(true);
            ui->spinCellAlternateRowSpacing->blockSignals(true);
            ui->spinCellAlternateColSpacing->setValue(cellShape.getAlternateColSpacing());
            ui->spinCellAlternateRowSpacing->setValue(cellShape.getAlternateRowSpacing());
            ui->spinCellAlternateColSpacing->blockSignals(false);
            ui->spinCellAlternateRowSpacing->blockSignals(false);

            //Update cell alternate offset
            ui->spinCustomCellAlternateColOffset->blockSignals(true);
            ui->spinCustomCellAlternateRowOffset->blockSignals(true);
            ui->spinCustomCellAlternateColOffset->setValue(cellShape.getAlternateColOffset());
            ui->spinCustomCellAlternateRowOffset->setValue(cellShape.getAlternateRowOffset());
            ui->spinCustomCellAlternateColOffset->blockSignals(false);
            ui->spinCustomCellAlternateRowOffset->blockSignals(false);

            //Update cell flip states
            //No need to block signals as setChecked does not trigger clicked signal
            ui->checkCellColFlipH->setChecked(cellShape.getColFlipHorizontal());
            ui->checkCellColFlipV->setChecked(cellShape.getColFlipVertical());
            ui->checkCellRowFlipH->setChecked(cellShape.getRowFlipHorizontal());
            ui->checkCellRowFlipV->setChecked(cellShape.getRowFlipVertical());

            //Extract cell name from filename
            QString cellName = filename.right(filename.size() - filename.lastIndexOf('/') - 1);
            cellName.chop(4);
            ui->lineCustomCellName->setText(cellName);

            file.close();

            ui->widgetCellShapeViewer->setCellShape(cellShape);
            ui->widgetCellShapeViewer->updateGrid();
            cellShapeChanged = true;
        }
    }
}

//Copies cell name from cell shape tab to generator settings tab
void MainWindow::cellNameChanged(const QString &text)
{
    ui->lineCellShape->setText(text);
}

//Prompts user for a cell mask image
//If an image is given converts to a binary image
void MainWindow::selectCellMask()
{
    QString filename = QFileDialog::getOpenFileName(this, tr("Select cell mask"), "",
                                                    "Image Files (*.bmp *.dib *.jpeg *.jpg "
                                                    "*.jpe *.jp2 *.png *.pbm *.pgm *.ppm "
                                                    "*.pxm *.pnm *.sr *.ras *.tiff *.tif "
                                                    "*.hdr *.pic)");
    cv::Mat tmp = cv::imread(filename.toStdString(), cv::IMREAD_GRAYSCALE);
    if (!tmp.empty())
    {
        CellShape cellShape(tmp);

        ui->widgetCellShapeViewer->setCellShape(cellShape);
        ui->widgetCellShapeViewer->updateGrid();
        ui->lineCellMask->setText(filename);

        ui->spinCustomCellSpacingCol->blockSignals(true);
        ui->spinCustomCellSpacingRow->blockSignals(true);
        ui->spinCustomCellAlternateColOffset->blockSignals(true);
        ui->spinCustomCellAlternateRowOffset->blockSignals(true);
        ui->spinCustomCellSpacingCol->setValue(tmp.cols);
        ui->spinCustomCellSpacingRow->setValue(tmp.rows);
        ui->spinCustomCellAlternateColOffset->setValue(0);
        ui->spinCustomCellAlternateRowOffset->setValue(0);
        ui->spinCustomCellSpacingCol->blockSignals(false);
        ui->spinCustomCellSpacingRow->blockSignals(false);
        ui->spinCustomCellAlternateColOffset->blockSignals(false);
        ui->spinCustomCellAlternateRowOffset->blockSignals(false);

        cellShapeChanged = true;
    }
}

//Update custom cell column spacing
void MainWindow::cellSpacingColChanged(int t_value)
{
    if (!ui->spinCellAlternateColSpacing->isEnabled())
        ui->spinCellAlternateColSpacing->setValue(t_value);
    ui->widgetCellShapeViewer->getCellShape().setColSpacing(t_value);
    ui->widgetCellShapeViewer->updateGrid();

    cellShapeChanged = true;
}

//Update custom cell column offset y
void MainWindow::cellSpacingRowChanged(int t_value)
{
    if (!ui->spinCellAlternateRowSpacing->isEnabled())
        ui->spinCellAlternateRowSpacing->setValue(t_value);
    ui->widgetCellShapeViewer->getCellShape().setRowSpacing(t_value);
    ui->widgetCellShapeViewer->updateGrid();

    cellShapeChanged = true;
}

//Update custom cell row offset x
void MainWindow::cellAlternateColOffsetChanged(int t_value)
{
    ui->widgetCellShapeViewer->getCellShape().setAlternateColOffset(t_value);
    ui->widgetCellShapeViewer->updateGrid();

    cellShapeChanged = true;
}

//Update custom cell row offset y
void MainWindow::cellAlternateRowOffsetChanged(int t_value)
{
    ui->widgetCellShapeViewer->getCellShape().setAlternateRowOffset(t_value);
    ui->widgetCellShapeViewer->updateGrid();

    cellShapeChanged = true;
}

//Update custom cell alternate column horizontal flipping
void MainWindow::cellColumnFlipHorizontalChanged(bool t_state)
{
    ui->widgetCellShapeViewer->getCellShape().setColFlipHorizontal(t_state);
    ui->widgetCellShapeViewer->updateGrid();

    cellShapeChanged = true;
}

//Update custom cell alternate column vertical flipping
void MainWindow::cellColumnFlipVerticalChanged(bool t_state)
{
    ui->widgetCellShapeViewer->getCellShape().setColFlipVertical(t_state);
    ui->widgetCellShapeViewer->updateGrid();

    cellShapeChanged = true;
}

//Update custom cell alternate row horizontal flipping
void MainWindow::cellRowFlipHorizontalChanged(bool t_state)
{
    ui->widgetCellShapeViewer->getCellShape().setRowFlipHorizontal(t_state);
    ui->widgetCellShapeViewer->updateGrid();

    cellShapeChanged = true;
}

//Update custom cell alternate row vertical flipping
void MainWindow::cellRowFlipVerticalChanged(bool t_state)
{
    ui->widgetCellShapeViewer->getCellShape().setRowFlipVertical(t_state);
    ui->widgetCellShapeViewer->updateGrid();

    cellShapeChanged = true;
}

//Enables/disables cell alternate row spacing
void MainWindow::enableCellAlternateRowSpacing(bool t_state)
{
    if (!t_state)
        ui->spinCellAlternateRowSpacing->setValue(ui->spinCustomCellSpacingRow->value());

    ui->spinCellAlternateRowSpacing->setEnabled(t_state);
    cellShapeChanged = true;
}

//Enables/disables cell alternate column spacing
void MainWindow::enableCellAlternateColSpacing(bool t_state)
{
    if (!t_state)
        ui->spinCellAlternateColSpacing->setValue(ui->spinCustomCellSpacingCol->value());

    ui->spinCellAlternateColSpacing->setEnabled(t_state);
    cellShapeChanged = true;
}

//Updates cell alternate row spacing
void MainWindow::cellAlternateRowSpacingChanged(int t_value)
{
    if (ui->checkCellAlternateRowSpacing->isEnabled())
    {
        ui->widgetCellShapeViewer->getCellShape().setAlternateRowSpacing(t_value);
        ui->widgetCellShapeViewer->updateGrid();

        cellShapeChanged = true;
    }
}

//Updates cell alternate column spacing
void MainWindow::cellAlternateColSpacingChanged(int t_value)
{
    if (ui->checkCellAlternateColSpacing->isEnabled())
    {
        ui->widgetCellShapeViewer->getCellShape().setAlternateColSpacing(t_value);
        ui->widgetCellShapeViewer->updateGrid();

        cellShapeChanged = true;
    }
}

//Used to add images to the image library
//Creates a QFileDialog allowing the user to select images
void MainWindow::addImages()
{
    QStringList filenames = QFileDialog::getOpenFileNames(this, tr("Select image to add"), "",
                                                          "Image Files (*.bmp *.dib *.jpeg *.jpg "
                                                          "*.jpe *.jp2 *.png *.pbm *.pgm *.ppm "
                                                          "*.pxm *.pnm *.sr *.ras *.tiff *.tif "
                                                          "*.hdr *.pic)");

    progressBar->setMaximum(filenames.size());
    progressBar->setValue(0);
    progressBar->setVisible(true);
    std::vector<cv::Mat> originalImages;
    std::vector<cv::Mat> images;
    std::vector<QString> names;
    //For all files selected by user load and add to library
    for (auto filename: filenames)
    {
        cv::Mat image = cv::imread(filename.toStdString());
        if (image.empty())
        {
            qDebug() << "Could not open or find the image";
            continue;
        }

        //Square image with focus on center and resize to image size
        UtilityFuncs::imageToSquare(image, UtilityFuncs::SquareMethod::CROP);
        originalImages.push_back(image);

        //Extracts filename and extension from full path
        QString name;
        for (auto it = filename.cbegin() + filename.lastIndexOf('/') + 1, end = filename.cend();
             it != end; ++it)
        {
            name += (*it);
        }
        names.push_back(name);
        progressBar->setValue(progressBar->value() + 1);
    }

    //Resize to images to current library size
    images = UtilityFuncs::batchResizeMat(originalImages, imageSize, imageSize,
                                          UtilityFuncs::ResizeType::EXACT, progressBar);

    auto nameIt = names.cbegin();
    auto imageIt = images.cbegin();
    for (auto image: originalImages)
    {
        //Add image to library
        QListWidgetItem listItem(QIcon(UtilityFuncs::matToQPixmap(*imageIt)), *nameIt);
        ui->listPhoto->addItem(new QListWidgetItem(listItem));

        //Store QListWidgetItem with resized and original OpenCV Mat
        allImages.insert(listItem, {*imageIt, image});

        ++nameIt;
        ++imageIt;
    }

    //Update tab widget to show new image count next to Image Library
    ui->tabWidget->setTabText(1, tr("Image Library (") + QString::number(allImages.size()) + ")");
}

//Deletes selected images
void MainWindow::deleteImages()
{
    auto selectedItems = ui->listPhoto->selectedItems();
    for (auto item: selectedItems)
        allImages.remove(*item);

    qDeleteAll(selectedItems);

    //Update tab widget to show new image count next to Image Library
    ui->tabWidget->setTabText(1, tr("Image Library (") + QString::number(allImages.size()) + ")");
}

//Reads the cell size from ui spin box, then resizes all images
void MainWindow::updateCellSize()
{
    auto t1 = std::chrono::high_resolution_clock::now();

    if (imageSize == ui->spinLibCellSize->value())
        return;

    imageSize = ui->spinLibCellSize->value();
    ui->spinCellSize->setValue(imageSize);
    ui->listPhoto->clear();

    std::vector<cv::Mat> images;
    for (auto image: allImages.values())
        images.push_back(image.second);

    images = UtilityFuncs::batchResizeMat(images, imageSize, imageSize,
                                          UtilityFuncs::ResizeType::EXACT, progressBar);

    auto it = images.cbegin();
    for (auto listItem: allImages.keys())
    {
        allImages[listItem].first = *it;
        listItem.setIcon(QIcon(UtilityFuncs::matToQPixmap(*it)));
        ui->listPhoto->addItem(new QListWidgetItem(listItem));

        ++it;
    }

    qDebug() << "Time: " << std::chrono::duration_cast<std::chrono::microseconds>(
                    std::chrono::high_resolution_clock::now() - t1).count() << "\xC2\xB5s";
}

//Saves the image library to a file
void MainWindow::saveLibrary()
{
    //mil = Mosaic Image Library
    QString filename = QFileDialog::getSaveFileName(this, tr("Save image library"), "",
                                                    tr("Mosaic Image Library") + " (*.mil)");

    if (!filename.isNull())
    {
        QFile file(filename);
        file.open(QIODevice::WriteOnly);
        if (file.isWritable())
        {
            QDataStream out(&file);
            //Write header with "magic number" and version
            out << static_cast<quint32>(UtilityFuncs::MIL_MAGIC);
            out << static_cast<qint32>(UtilityFuncs::MIL_VERSION);

            out.setVersion(out.version());
            //Write images and names
            out << imageSize;
            out << allImages.size();
            progressBar->setRange(0, allImages.size());
            progressBar->setValue(0);
            progressBar->setVisible(true);
            for (auto listItem: allImages.keys())
            {
                out << allImages[listItem].first;
                out << listItem.text();
                progressBar->setValue(progressBar->value() + 1);
            }
            progressBar->setVisible(false);
            file.close();
        }
    }
}

//Loads an image library from a file
void MainWindow::loadLibrary()
{
    // mil = Mosaic Image Library
    QString filename = QFileDialog::getOpenFileName(this, tr("Select image library to load"), "",
                                                    tr("Mosaic Image Library") + " (*.mil)");

    if (!filename.isNull())
    {
        QFile file(filename);
        file.open(QIODevice::ReadOnly);
        if (file.isReadable())
        {
            QDataStream in(&file);

            //Read and check magic number
            quint32 magic;
            in >> magic;
            if (magic != UtilityFuncs::MIL_MAGIC)
            {
                QMessageBox msgBox;
                msgBox.setText(filename + tr(" is not a valid .mil file"));
                msgBox.exec();
                return;
            }

            //Read the version
            qint32 version;
            in >> version;
            if (version == UtilityFuncs::MIL_VERSION)
                in.setVersion(in.version());
            else
            {
                QMessageBox msgBox;
                if (version < UtilityFuncs::VERSION_NO)
                    msgBox.setText(filename + tr(" uses an outdated file version"));
                else
                    msgBox.setText(filename + tr(" uses a newer file version"));
                msgBox.exec();
                return;
            }

            allImages.clear();
            ui->listPhoto->clear();

            //Read images and names
            in >> imageSize;
            ui->spinLibCellSize->setValue(imageSize);
            ui->spinCellSize->setValue(imageSize);
            int numberOfImage;
            in >> numberOfImage;
            progressBar->setRange(0, numberOfImage);
            progressBar->setValue(0);
            progressBar->setVisible(true);
            while (numberOfImage > 0)
            {
                --numberOfImage;
                cv::Mat image;
                in >> image;

                QString name;
                in >> name;

                QListWidgetItem listItem(QIcon(UtilityFuncs::matToQPixmap(image)), name);
                ui->listPhoto->addItem(new QListWidgetItem(listItem));

                allImages.insert(listItem, {image, image});
                progressBar->setValue(progressBar->value() + 1);
            }
            progressBar->setVisible(false);

            file.close();

            //Update tab widget to show new image count next to Image Library
            ui->tabWidget->setTabText(1, tr("Image Library (")
                                      + QString::number(allImages.size()) + ")");
        }
    }
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
                    UtilityFuncs::resizeImage(mainImage, mainImage.rows, mainImage.cols,
                                              UtilityFuncs::ResizeType::INCLUSIVE));

        ui->widgetGridPreview->updateGrid();
    }
}

//Opens colour visualisation window
void MainWindow::compareColours()
{
    if (allImages.size() == 0 || mainImage.empty())
        return;

    std::vector<cv::Mat> libImages;
    for (auto pair: allImages.values())
        libImages.push_back(pair.first);

    ColourVisualisation *colourVisualisation = new ColourVisualisation(this, mainImage,
                                                                       libImages.data(),
                                                                       libImages.size());
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
                        UtilityFuncs::resizeImage(mainImage, ui->spinPhotomosaicHeight->value(),
                                                  ui->spinPhotomosaicWidth->value(),
                                                  UtilityFuncs::ResizeType::INCLUSIVE));
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
                    UtilityFuncs::resizeImage(mainImage, ui->spinPhotomosaicHeight->value(),
                                              ui->spinPhotomosaicWidth->value(),
                                              UtilityFuncs::ResizeType::INCLUSIVE));
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
                    UtilityFuncs::resizeImage(mainImage, mainImage.rows, mainImage.cols,
                                              UtilityFuncs::ResizeType::INCLUSIVE));
        ui->widgetGridPreview->updateGrid();
    }
}

//Updates grid preview when detail level changes
void MainWindow::photomosaicDetailChanged(int /*i*/)
{
    clampDetail();

    ui->widgetGridPreview->setDetail(ui->spinDetail->value());
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
        ui->widgetGridPreview->setCellShape(ui->widgetCellShapeViewer->getCellShape().
                                            resized(t_value, t_value));
    }
    else
    {
        ui->widgetGridPreview->setCellShape(CellShape(cv::Mat(t_value, t_value,
                                                              CV_8UC1, cv::Scalar(255))));
    }

    ui->widgetGridPreview->setSizeSteps(ui->spinMinCellSize->getHalveSteps());
    ui->widgetGridPreview->updateGrid();
}

//Updates grid preview
void MainWindow::minimumCellSizeChanged(int /*t_value*/)
{
    clampDetail();

    ui->widgetGridPreview->setSizeSteps(ui->spinMinCellSize->getHalveSteps());
    ui->widgetGridPreview->updateGrid();
}

//Enables/disables non-square cell shapes, GUI widgets for choosing
void MainWindow::enableCellShape(bool t_state)
{
    ui->lineCellShape->setEnabled(t_state);
    if (t_state)
    {
        ui->widgetGridPreview->setCellShape(ui->widgetCellShapeViewer->getCellShape().
                                            resized(ui->spinCellSize->value(),
                                                    ui->spinCellSize->value()));
    }
    else
    {
        ui->widgetGridPreview->setCellShape(CellShape(cv::Mat(ui->spinCellSize->value(),
                                                              ui->spinCellSize->value(),
                                                              CV_8UC1, cv::Scalar(255))));
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
    if (allImages.size() == 0)
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
    mainImage = UtilityFuncs::resizeImage(mainImage, ui->spinPhotomosaicHeight->value(),
                                          ui->spinPhotomosaicWidth->value(),
                                          UtilityFuncs::ResizeType::INCLUSIVE);

    std::vector<cv::Mat> library;
    for (auto pair: allImages.values())
        library.push_back(pair.first);

    if (library.front().cols != ui->spinCellSize->value())
        library = UtilityFuncs::batchResizeMat(library, ui->spinCellSize->value(),
                                               ui->spinCellSize->value(),
                                               UtilityFuncs::ResizeType::EXACT, progressBar);

    //Generate Photomosaic
    PhotomosaicGenerator generator(this);
    generator.setMainImage(mainImage);
    generator.setLibrary(library);
    generator.setDetail(ui->spinDetail->value());
    if (ui->comboMode->currentText() == "RGB Euclidean")
        generator.setMode(PhotomosaicGenerator::Mode::RGB_EUCLIDEAN);
    else if (ui->comboMode->currentText() == "CIE76")
        generator.setMode(PhotomosaicGenerator::Mode::CIE76);
    else if (ui->comboMode->currentText() == "CIEDE2000")
        generator.setMode(PhotomosaicGenerator::Mode::CIEDE2000);

    generator.setCellShape(ui->widgetGridPreview->getCellShape());

    generator.setGridState(ui->widgetGridPreview->getGridState());

    generator.setSizeSteps(ui->spinMinCellSize->getHalveSteps());

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
