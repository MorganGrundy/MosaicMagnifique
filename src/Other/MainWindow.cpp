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

#include "MainWindow.h"
#include "ui_MainWindow.h"
#include "Logger.h"

#include <QDesktopServices>
#include <QFileDialog>
#include <QPixmap>
#include <QMessageBox>
#include <QInputDialog>
#include <QProgressDialog>
#include <QThread>
#include <opencv2/imgcodecs.hpp>
#include <chrono>

#include "ImageUtility.h"
#include "..\Photomosaic\PhotomosaicViewer.h"
#include "..\ImageLibrary\ColourVisualisation.h"
#include "..\Photomosaic\CPUPhotomosaicGenerator.h"
#include "..\Grid\GridEditor.h"
#include "..\Resources\VersionInfo.h"

#ifdef CUDA
#include <cuda_runtime.h>
#include "..\Photomosaic\CUDA\CUDAUtility.h"
#include "..\Photomosaic\CUDA\CUDAPhotomosaicGenerator.h"
#endif

#ifdef CUDA
#include <opencv2/cudawarping.hpp>
#endif

MainWindow::MainWindow(QWidget *t_parent)
    : QMainWindow{t_parent}, ui{new Ui::MainWindow}
{
    ui->setupUi(this);

    setWindowIcon(QIcon(":/MosaicMagnifique.png"));

    //Populate colour difference and scheme combo boxes
    for (auto type : ColourDifference::Type_STR)
        ui->comboColourDifference->addItem(type);
    for (auto type : ColourScheme::Type_STR)
        ui->comboColourScheme->addItem(type);

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

    connect(ui->tabWidget, &QTabWidget::currentChanged, this, &MainWindow::tabChanged);

    photomosaicSizeRatio = static_cast<double>(ui->spinPhotomosaicWidth->value()) /
                           ui->spinPhotomosaicHeight->value();

    //Website action opens github pages site
    connect(ui->actionWebsite, &QAction::triggered, [&]([[maybe_unused]] const bool triggered)
            {
                QDesktopServices::openUrl(QUrl("https://morgangrundy.github.io/"));
            });
    //Github action opens github repository
    connect(ui->actionGithub, &QAction::triggered, [&]([[maybe_unused]] const bool triggered)
            {
                QDesktopServices::openUrl(QUrl("https://github.com/MorganGrundy/MosaicMagnifique"));
            });
    //About action displays Mosaic Magnifique version, and build date and time
    connect(ui->actionAbout, &QAction::triggered, [&]([[maybe_unused]] const bool triggered)
            {
                QMessageBox msgBox;
                msgBox.setWindowTitle("About Mosaic Magnifique");

                msgBox.setText("<b>Mosaic Magnifique " + QString("%1.%2.%3").arg(VERSION_MAJOR, VERSION_MINOR, VERSION_BUILD) + "</b>");
                msgBox.setInformativeText("Built on " + QStringLiteral(__DATE__) + " " + QStringLiteral(__TIME__));
                msgBox.setIconPixmap(QPixmap(":/MosaicMagnifique.png"));
                msgBox.setStandardButtons(QMessageBox::Close);
                msgBox.exec();
            });

    //Cell Shape Editor
    cellShapeChanged = false;
    newCellShape = CellShape(CellShape::DEFAULT_CELL_SIZE);

    connect(ui->cellShapeEditor, &CellShapeEditor::cellShapeChanged, this, &MainWindow::updateCellShape);
    connect(ui->cellShapeEditor, &CellShapeEditor::cellNameChanged, this, &MainWindow::updateCellName);

    //Image Library Editor
    ui->imageLibraryEditor->setProgressBar(progressBar);
    connect(ui->imageLibraryEditor, &ImageLibraryEditor::imageLibraryChanged, this, &MainWindow::updateImageLibraryCount);

    //Connects generator settings to appropriate methods
    connect(ui->buttonMainImage, &QPushButton::released, this, &MainWindow::selectMainImage);
    connect(ui->buttonCompareColours, &QPushButton::released, this, &MainWindow::compareColours);

    connect(ui->buttonPhotomosaicSizeLink, &QPushButton::released, this, &MainWindow::photomosaicSizeLink);
    connect(ui->spinPhotomosaicWidth, qOverload<int>(&QSpinBox::valueChanged), this, &MainWindow::photomosaicWidthChanged);
    connect(ui->spinPhotomosaicHeight, qOverload<int>(&QSpinBox::valueChanged), this, &MainWindow::photomosaicHeightChanged);
    connect(ui->buttonPhotomosaicSize, &QPushButton::released, this, &MainWindow::loadImageSize);

    connect(ui->spinDetail, qOverload<int>(&QSpinBox::valueChanged), this, &MainWindow::photomosaicDetailChanged);

    connect(ui->spinCellSize, qOverload<int>(&QSpinBox::valueChanged), this, &MainWindow::cellSizeChanged);
    connect(ui->spinSizeSteps, qOverload<int>(&QSpinBox::valueChanged), this, &MainWindow::sizeStepsChanged);
    connect(ui->checkCellShape, &QCheckBox::clicked, this, &MainWindow::enableCellShape);

    connect(ui->buttonEditGrid, &QPushButton::released, this, &MainWindow::editCellGrid);

    connect(ui->buttonGenerate, &QPushButton::released, this, &MainWindow::generatePhotomosaic);

#ifdef CUDA
    CUDAinit();
#else
    ui->checkCUDA->hide();
    ui->comboCUDA->hide();
#endif

    //Sets default cell size
    ui->spinSizeSteps->setValue(0);
    ui->spinCellSize->setMinimum(CellShape::MIN_CELL_SIZE);
    ui->spinCellSize->setValue(CellShape::DEFAULT_CELL_SIZE);
    minCellSize = CellShape::DEFAULT_CELL_SIZE;

    //tabWidget starts on Generator Settings tab
    ui->tabWidget->setCurrentIndex(2);

    //Sets default detail level
    ui->spinDetail->setValue(100);
}

#ifdef CUDA
//Initialise CUDA and relevant UI
void MainWindow::CUDAinit()
{
    LogInfo("Initialising CUDA...");
    int deviceCount, device;
    int gpuDeviceCount = 0;
    cudaDeviceProp properties;
    cudaError_t cudaResultCode = cudaGetDeviceCount(&deviceCount);
    if (cudaResultCode != cudaSuccess)
    {
        deviceCount = 0;
        LogWarn("No CUDA devices found.");
    }

    //Check devices are not emulation only (9999)
    for (device = 0; device < deviceCount; ++device) {
        gpuErrchk(cudaGetDeviceProperties(&properties, device));
        if (properties.major != 9999)
        {
            ++gpuDeviceCount;
            //Add device name to combo box
            ui->comboCUDA->addItem(properties.name);

            LogInfo(QString("Found valid CUDA device: %1").arg(properties.name));
        }
        else
        {
            LogInfo(QString("Ignoring emulation CUDA device: %1").arg(properties.name));
        }
    }

    //No devices so disable CUDA controls
    if (gpuDeviceCount == 0)
    {
        LogWarn("No valid CUDA devices, disabled CUDA functionlity.");
        ui->checkCUDA->setChecked(false);
        ui->checkCUDA->setEnabled(false);
        ui->comboCUDA->setEnabled(false);
    }
    else
    {
        connect(ui->comboCUDA, qOverload<int>(&QComboBox::currentIndexChanged),
                this, &MainWindow::CUDADeviceChanged);

        //Initialise primary CUDA device
        CUDADeviceChanged(0);
    }
}
#endif

MainWindow::~MainWindow()
{
    delete ui;
}

//Updates cell shape in grid preview
void MainWindow::tabChanged(int t_index)
{
    LogInfo(QString("Main Window tab changed to %1").arg(ui->tabWidget->tabText(t_index)));

    //Generator settings tab
    if (t_index == 2)
    {
        if (ui->checkCellShape->isChecked() && cellShapeChanged)
        {
            cellShapeChanged = false;

            ui->widgetGridPreview->getCellGroup().setCellShape(
                newCellShape.resized(ui->spinCellSize->value()));

            updateGridPreview();
        }
    }
}

//Updates cell shape
void MainWindow::updateCellShape(const CellShape &t_cellShape)
{
    newCellShape = t_cellShape;
    cellShapeChanged = true;
}

//Update cell shape name
void MainWindow::updateCellName(const QString &t_name)
{
    ui->lineCellShape->setText(t_name);
}

//Updates image library count in tab widget
void MainWindow::updateImageLibraryCount(size_t t_newSize)
{
    ui->tabWidget->setTabText(1, tr("Image Library (") + QString::number(t_newSize) + ")");
}

//Prompts user for a main image
void MainWindow::selectMainImage()
{
    LogInfo("Selecting main image...");

    const QString filename = QFileDialog::getOpenFileName(this, tr("Select main image"), "",
                                                    "Image Files (*.bmp *.dib *.jpeg *.jpg "
                                                    "*.jpe *.jp2 *.png *.pbm *.pgm *.ppm "
                                                    "*.pxm *.pnm *.sr *.ras *.tiff *.tif "
                                                    "*.hdr *.pic)");
    if (!filename.isNull())
    {
        LogInfo(QString("Selected main image: %1").arg(filename));

        //Load main image and check is valid
        mainImage = cv::imread(filename.toStdString());
        if (mainImage.empty())
        {
            LogInfo(QString("Failed to load main image: %1").arg(filename));

            ui->widgetGridPreview->setBackground(cv::Mat());
            updateGridPreview();
            QMessageBox msgBox;
            msgBox.setText(tr("The main image \"") + ui->lineMainImage->text() +
                           tr("\" failed to load"));
            msgBox.exec();
            return;
        }

        LogInfo(QString("Loaded main image: %1").arg(filename));

        ui->lineMainImage->setText(filename);

        //Update main image size
        ui->spinPhotomosaicHeight->blockSignals(true);
        ui->spinPhotomosaicWidth->blockSignals(true);
        ui->spinPhotomosaicHeight->setValue(mainImage.rows);
        ui->spinPhotomosaicWidth->setValue(mainImage.cols);
        ui->spinPhotomosaicHeight->blockSignals(false);
        ui->spinPhotomosaicWidth->blockSignals(false);
        photomosaicSizeRatio = static_cast<double>(mainImage.cols) / mainImage.rows;

        //Gives main image to grid preview
        ui->widgetGridPreview->setBackground(mainImage);
        updateGridPreview();
    }
}

//Opens colour visualisation window
void MainWindow::compareColours()
{
    if (ui->imageLibraryEditor->getImageLibrarySize() == 0 || mainImage.empty())
        return;

    LogInfo("Opening colour visualisation window.");

    ColourVisualisation *colourVisualisation =
        new ColourVisualisation(this, mainImage, ui->imageLibraryEditor->getImageLibrary());
    colourVisualisation->show();
}

//Links width and height of photomosaic so they scale together
//Updates link icon
void MainWindow::photomosaicSizeLink()
{
    if (ui->buttonPhotomosaicSizeLink->isChecked())
    {
        ui->buttonPhotomosaicSizeLink->setIcon(QIcon(":/img/LinkIcon.png"));
        //Gets ratio between current width and height
        photomosaicSizeRatio = static_cast<double>(ui->spinPhotomosaicWidth->value()) /
                ui->spinPhotomosaicHeight->value();

        LogInfo(QString("Photomosaic size ratio set (w%1/h%2)").arg(ui->spinPhotomosaicWidth->value(), ui->spinPhotomosaicHeight->value()));
    }
    else
    {
        ui->buttonPhotomosaicSizeLink->setIcon(QIcon(":/img/UnlinkIcon.png"));
        LogInfo("Photomosaic size ratio unset");
    }
}

//Updates photomosaic width
void MainWindow::photomosaicWidthChanged(int i)
{
    LogInfo(QString("Photomosaic width changed: %1").arg(i));

    //If size link active, height is scaled with width
    if (ui->buttonPhotomosaicSizeLink->isChecked())
    {
        const int newHeight = std::round(i / photomosaicSizeRatio);
        LogInfo(QString("Changing Photomosaic height based on size ratio: %1").arg(newHeight));

        //Blocks signals while changing value to prevent infinite loop
        ui->spinPhotomosaicHeight->blockSignals(true);
        ui->spinPhotomosaicHeight->setValue(newHeight);
        ui->spinPhotomosaicHeight->blockSignals(false);
    }

    //Updates image size in grid preview
    if (!mainImage.empty())
    {
        ui->widgetGridPreview->setBackground(
            ImageUtility::resizeImage(mainImage, ui->spinPhotomosaicHeight->value(),
                                      ui->spinPhotomosaicWidth->value(),
                                      ImageUtility::ResizeType::INCLUSIVE));
        updateGridPreview();
    }
}

//Updates photomosaic height
void MainWindow::photomosaicHeightChanged(int i)
{
    LogInfo(QString("Photomosaic height changed: %1").arg(i));

    //If size link active, width is scaled with height
    if (ui->buttonPhotomosaicSizeLink->isChecked())
    {
        const int newWidth = std::floor(i * photomosaicSizeRatio);
        LogInfo(QString("Changing Photomosaic width based on size ratio: %1").arg(newWidth));

        //Blocks signals while changing value to prevent infinite loop
        ui->spinPhotomosaicWidth->blockSignals(true);
        ui->spinPhotomosaicWidth->setValue(newWidth);
        ui->spinPhotomosaicWidth->blockSignals(false);
    }

    //Updates image size in grid preview
    if (!mainImage.empty())
    {
        ui->widgetGridPreview->setBackground(
            ImageUtility::resizeImage(mainImage, ui->spinPhotomosaicHeight->value(),
                                      ui->spinPhotomosaicWidth->value(),
                                      ImageUtility::ResizeType::INCLUSIVE));
        updateGridPreview();
    }
}

//Sets photomosaic size to current main image size
void MainWindow::loadImageSize()
{
    if (!mainImage.empty())
    {
        LogInfo(QString("Set Photomosaic size to main image size (w%1, h%2)").arg(mainImage.cols, mainImage.rows));

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
        updateGridPreview();
    }
}

//Updates detail level
void MainWindow::photomosaicDetailChanged([[maybe_unused]] int i)
{
    LogInfo(QString("Photomosaic detail level changed to %1").arg(ui->spinDetail->value()));

    ui->widgetGridPreview->getCellGroup().setDetail(ui->spinDetail->value());
    if (!mainImage.empty())
        updateGridPreview();
}

//Updates cell size
void MainWindow::cellSizeChanged(int t_value)
{
    LogInfo(QString("Cell size changed to %1").arg(t_value));

    //Updates minimum cell size
    updateCellSizes();

    if (ui->checkCellShape->isChecked())
        ui->widgetGridPreview->getCellGroup().setCellShape(newCellShape.resized(t_value));
    else
        ui->widgetGridPreview->getCellGroup().setCellShape(CellShape(t_value));

    updateGridPreview();
}

//Updates cell grid size steps
void MainWindow::sizeStepsChanged([[maybe_unused]] int t_value)
{
    LogInfo(QString("Size steps changed to %1").arg(t_value));

    //Updates minimum cell size
    updateCellSizes();

    ui->widgetGridPreview->getCellGroup().setSizeSteps(ui->spinSizeSteps->value());
    updateGridPreview();
}

//Enables/disables custom cell shapes
void MainWindow::enableCellShape(bool t_state)
{
    ui->lineCellShape->setEnabled(t_state);

    if (t_state)
    {
        ui->widgetGridPreview->getCellGroup().setCellShape(newCellShape.resized(ui->spinCellSize->value()));
        LogInfo("Enabled cell shape");
    }
    else
    {
        ui->widgetGridPreview->getCellGroup().setCellShape(CellShape(ui->spinCellSize->value()));
        LogInfo("Disabled cell shape");
    }

    updateGridPreview();
}

//Allows user to manually edit current cell grid
void MainWindow::editCellGrid()
{
    if (!mainImage.empty())
    {
        LogInfo("Opening grid editor");

        //Create grid editor
        GridEditor gridEditor(
            ImageUtility::resizeImage(mainImage, ui->spinPhotomosaicHeight->value(),
                                      ui->spinPhotomosaicWidth->value(),
                                      ImageUtility::ResizeType::INCLUSIVE),
            ui->widgetGridPreview->getCellGroup(), this);

        gridEditor.setWindowModality(Qt::WindowModality::ApplicationModal);

        //When grid editor is closed get the new grid state and give to grid preview
        connect(&gridEditor, &GridEditor::gridStateChanged,
                [&](const GridUtility::MosaicBestFit &t_gridState)
                {
                    ui->widgetGridPreview->setGridState(t_gridState);
                    ui->widgetGridPreview->updateView();
                });

        //Show grid editor
        gridEditor.show();

        //Wait till window returns
        QEventLoop loop;
        connect(&gridEditor, SIGNAL(destroyed()), &loop, SLOT(quit()));
        loop.exec();
        LogInfo("Closed grid editor");
    }
}

#ifdef CUDA
//Changes CUDA device
void MainWindow::CUDADeviceChanged(int t_index)
{
    cudaDeviceProp properties;
    gpuErrchk(cudaGetDeviceProperties(&properties, t_index));

    LogInfo(QString("CUDA device changed: %1").arg(properties.name));

    gpuErrchk(cudaSetDevice(t_index));

    //Initialise CUDA device
    int *deviceInit;
    gpuErrchk(cudaMalloc(&deviceInit, 0));
}
#endif

//Generate and display a Photomosaic for current settings
void MainWindow::generatePhotomosaic()
{
    //Check main image is loaded
    if (ui->widgetGridPreview->getBackground().empty())
    {
        QMessageBox msgBox;
        msgBox.setText(tr("No main image loaded, please load an image"));
        msgBox.exec();
        return;
    }

    //Check library contains images
    if (ui->imageLibraryEditor->getImageLibrarySize() == 0)
    {
        QMessageBox msgBox;
        msgBox.setText(tr("The library is empty, please add some images"));
        msgBox.exec();
        return;
    }

    //Resize image library
    std::vector<cv::Mat> library = ui->imageLibraryEditor->getImageLibrary();
    if (library.front().cols != ui->spinCellSize->value())
    {
        ImageUtility::batchResizeMat(library, library,
            ui->spinCellSize->value(), ui->spinCellSize->value(),
            ImageUtility::ResizeType::EXACT, progressBar);
    }

    //Generate Photomosaic
    std::shared_ptr<PhotomosaicGeneratorBase> generator;

    //Choose which generator to use
#ifdef CUDA
    if (ui->checkCUDA->isChecked())
    {
        LogInfo("Generating Photomosaic with CUDA...");
        generator = std::make_shared<CUDAPhotomosaicGenerator>();
    }
    else
    {
#endif
        LogInfo("Generating Photomosaic on CPU...");
        generator = std::make_shared<CPUPhotomosaicGenerator>();
#ifdef CUDA
    }
#endif

    //Set generator settings
    generator->setMainImage(ui->widgetGridPreview->getBackground());

    generator->setLibrary(library);

    generator->setColourDifference(ColourDifference::strToEnum(ui->comboColourDifference->currentText()));
    generator->setColourScheme(ColourScheme::strToEnum(ui->comboColourScheme->currentText()));

    generator->setCellGroup(ui->widgetGridPreview->getCellGroup());
    generator->setGridState(ui->widgetGridPreview->getGridState());
    generator->setRepeat(ui->spinRepeatRange->value(), ui->spinRepeatAddition->value());

    //Create progress dialog
    QProgressDialog progressDialog(this);
    progressDialog.setWindowModality(Qt::WindowModal);
    progressDialog.setMinimumDuration(0);
    progressDialog.setLabelText("Finding best fits...");
    progressDialog.setMaximum(generator->getMaxProgress());

    connect(&progressDialog, &QProgressDialog::canceled,
            generator.get(), &PhotomosaicGeneratorBase::cancel);
    connect(generator.get(), &PhotomosaicGeneratorBase::progress,
            &progressDialog, &QProgressDialog::setValue);

    progressDialog.show();
    QCoreApplication::processEvents(QEventLoop::ProcessEventsFlag::DialogExec);

    //Generate Photomosaic and measure time
    const auto startTime = std::chrono::high_resolution_clock::now();
    const bool generateSucceeded = generator->generateBestFits();
    const double duration = std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::high_resolution_clock::now() - startTime).count() / 1000.0;

    //Displays Photomosaic
    if (generateSucceeded)
    {
        PhotomosaicViewer *photomosaicViewer =
            new PhotomosaicViewer(this, generator, duration);
        photomosaicViewer->setAttribute(Qt::WA_DeleteOnClose);
        photomosaicViewer->show();
    }
}

//Update list of cell sizes
void MainWindow::updateCellSizes()
{
    LogInfo(QString("Updating cell sizes."));

    //Create string listing cell sizes for each size step
    QString cellSizes("Cell Sizes: ");
    int cellSize = ui->spinCellSize->value();
    cellSizes.append(QString::number(cellSize));
    int i = 0;
    //For each size step, stop if cell size will subseed minimum cell size
    for (; i < ui->spinSizeSteps->value() && cellSize > CellShape::MIN_CELL_SIZE * 2; ++i)
    {
        cellSize /= 2;

        cellSizes.append(", ");
        cellSizes.append(QString::number(cellSize));
    }

    //Size steps would subseed minimum cell size, instead limit size steps
    if (i != ui->spinSizeSteps->value())
    {
        ui->spinSizeSteps->blockSignals(true);
        ui->spinSizeSteps->setValue(i);
        ui->spinSizeSteps->blockSignals(false);
    }

    //Save minimum cell size
    minCellSize = cellSize;
    //Show list
    ui->labelCellSizesList->setText(cellSizes);
}

//Updates grid preview
void MainWindow::updateGridPreview()
{
    LogInfo("Updating grid preview...");

    //Save current status message
    QString savedMessage = ui->statusbar->currentMessage();
    //Set status message
    ui->statusbar->showMessage("Updating grid preview...");
    QCoreApplication::processEvents(QEventLoop::ProcessEventsFlag::ExcludeUserInputEvents);
    //Save focus widget
    QWidget *focusWidget = QApplication::focusWidget();
    //Disable window interactions
    setEnabled(false);

    //Update grid preview
    ui->widgetGridPreview->updateGrid();

    //Return saved status message
    ui->statusbar->showMessage(savedMessage);
    //Enable window interactions
    setEnabled(true);
    //Return focus to saved widget
    if (focusWidget)
        focusWidget->setFocus();
}
