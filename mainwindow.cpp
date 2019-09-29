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
#include <opencv2/core/mat.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <chrono>

#include "utilityfuncs.h"
#include "photomosaicgenerator.h"

#ifdef OPENCV_WITH_CUDA
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
    connect(ui->buttonPhotomosaicSizeLink, SIGNAL(released()), this, SLOT(photomosaicSizeLink()));
    connect(ui->spinPhotomosaicWidth, SIGNAL(valueChanged(int)), this,
            SLOT(photomosaicWidthChanged(int)));
    connect(ui->spinPhotomosaicHeight, SIGNAL(valueChanged(int)), this,
            SLOT(photomosaicHeightChanged(int)));
    connect(ui->buttonPhotomosaicSize, SIGNAL(released()), this, SLOT(loadImageSize()));

    connect(ui->buttonCellShape, SIGNAL(released()), this, SLOT(selectCellFolder()));
    connect(ui->checkCellShape, SIGNAL(stateChanged(int)), this, SLOT(enableCellShape(int)));

    connect(ui->buttonGenerate, SIGNAL(released()), this, SLOT(generatePhotomosaic()));
}

MainWindow::~MainWindow()
{
    delete ui;
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
        UtilityFuncs::imageToSquare(image);
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
                                          UtilityFuncs::ResizeType::EXCLUSIVE, progressBar);

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

    //Update status bar with new number of images
    ui->statusbar->showMessage(QString::number(allImages.size()) + tr(" images"));
}

//Deletes selected images
void MainWindow::deleteImages()
{
    auto selectedItems = ui->listPhoto->selectedItems();
    for (auto item: selectedItems)
        allImages.remove(*item);

    qDeleteAll(selectedItems);

    //Update status bar with new number of images
    ui->statusbar->showMessage(QString::number(allImages.size()) + tr(" images"));
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
                                          UtilityFuncs::ResizeType::EXCLUSIVE, progressBar);

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
    // mil = Mosaic Image Library
    QString filename = QFileDialog::getSaveFileName(this, tr("Save image library"), "",
                                                    tr("Image Files") + " (*.mil)");
    QFile file(filename);
    file.open(QIODevice::WriteOnly);
    if (file.isWritable())
    {
        QDataStream out(&file);
        //Write header with "magic number" and version
        out << static_cast<quint32>(0xADBE2480);
        out << static_cast<qint32>(UtilityFuncs::FILE_VERSION);

        out.setVersion(QDataStream::Qt_5_13);
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

//Loads an image library from a file
void MainWindow::loadLibrary()
{
    // mil = Mosaic Image Library
    QString filename = QFileDialog::getOpenFileName(this, tr("Select image library to load"), "",
                                                    tr("Image Files") + " (*.mil)");
    QFile file(filename);
    file.open(QIODevice::ReadOnly);
    if (file.isReadable())
    {
        QDataStream in(&file);

        //Read and check magic number
        quint32 magic;
        in >> magic;
        if (magic != 0xADBE2480)
        {
            QMessageBox msgBox;
            msgBox.setText(filename + tr(" is not a valid .mil file"));
            msgBox.exec();
            return;
        }

        //Read the version
        qint32 version;
        in >> version;
        if (version == UtilityFuncs::FILE_VERSION)
            in.setVersion(QDataStream::Qt_5_13);
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
    }
    //Update status bar with new number of images
    ui->statusbar->showMessage(QString::number(allImages.size()) + tr(" images"));
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
        ui->lineMainImage->setText(filename);
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
        ui->spinPhotomosaicHeight->setValue(static_cast<int>(i / photomosaicSizeRatio));
        ui->spinPhotomosaicHeight->blockSignals(false);
    }
}

//If link is active then when Photomosaic height changes updates width
void MainWindow::photomosaicHeightChanged(int i)
{
    if (ui->buttonPhotomosaicSizeLink->isChecked())
    {
        //Blocks signals while changing value to prevent infinite loop
        ui->spinPhotomosaicWidth->blockSignals(true);
        ui->spinPhotomosaicWidth->setValue(static_cast<int>(i * photomosaicSizeRatio));
        ui->spinPhotomosaicWidth->blockSignals(false);
    }
}

//If a main image has been entered sets the Photomosaic size spinboxes to the image size
void MainWindow::loadImageSize()
{
    //Load main image and check is valid
    cv::Mat mainImage = cv::imread(ui->lineMainImage->text().toStdString());
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
    }
}

//Enables/disables non-square cell shapes, GUI widgets for choosing
void MainWindow::enableCellShape(int t_state)
{
    ui->lineCellShape->setEnabled(t_state == Qt::Checked);
    ui->buttonCellShape->setEnabled(t_state == Qt::Checked);
}

//Prompts user for a cell folder
void MainWindow::selectCellFolder()
{
    QString folder = QFileDialog::getExistingDirectory(this, tr("Select cell folder"));

    if (!folder.isNull())
        ui->lineCellShape->setText(folder);
}

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
        msgBox.setText(tr("The main image \"") + ui->lineMainImage->text() + tr("\" failed to load"));
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
                                               UtilityFuncs::ResizeType::EXCLUSIVE, progressBar);

    //Generate Photomosaic
    PhotomosaicGenerator generator(this);
    if (ui->comboMode->currentText() == "RGB Euclidean")
        generator.m_mode = PhotomosaicGenerator::Mode::RGB_EUCLIDEAN;
    else if (ui->comboMode->currentText() == "CIE76")
        generator.m_mode = PhotomosaicGenerator::Mode::CIE76;
    else if (ui->comboMode->currentText() == "CIEDE2000")
        generator.m_mode = PhotomosaicGenerator::Mode::CIEDE2000;

    generator.m_repeatRange = ui->spinRepeatRange->value();
    generator.m_repeatAddition = ui->spinRepeatAddition->value();

    auto t1 = std::chrono::high_resolution_clock::now();
    cv::Mat mosaic = generator.generate(mainImage, library);
    qDebug() << "Generator time: " << std::chrono::duration_cast<std::chrono::seconds>(
                    std::chrono::high_resolution_clock::now() - t1).count() << "s";

    if (!mosaic.empty())
        cv::imshow("Mosaic", mosaic);
}

//Outputs a OpenCV mat to a QDataStream
//Can be used to save a OpenCV mat to a file
QDataStream &operator<<(QDataStream &t_out, const cv::Mat &t_mat)
{
    t_out << t_mat.type() << t_mat.rows << t_mat.cols;

    const int dataSize = t_mat.cols * t_mat.rows * static_cast<int>(t_mat.elemSize());
    QByteArray data = QByteArray::fromRawData(reinterpret_cast<const char *>(t_mat.ptr()),
                                              dataSize);
    t_out << data;

    return t_out;
}

//Inputs a OpenCV mat from a QDataStream
//Can be used to load a OpenCV mat from a file
QDataStream &operator>>(QDataStream &t_in, cv::Mat &t_mat)
{
    int type, rows, cols;
    QByteArray data;
    t_in >> type >> rows >> cols;
    t_in >> data;

    t_mat = cv::Mat(rows, cols, type, data.data()).clone();

    return t_in;
}
