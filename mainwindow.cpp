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

#ifdef OPENCV_WITH_CUDA
#include <opencv2/cudawarping.hpp>
#endif

#include "utilityfuncs.h"
#include "photomosaicgenerator.h"

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

    //Connects generator settings to appropriate methods
    connect(ui->toolMainImage, SIGNAL(released()), this, SLOT(selectMainImage()));
    connect(ui->toolCellShape, SIGNAL(released()), this, SLOT(selectCellFolder()));
    connect(ui->checkCellShape, SIGNAL(stateChanged(int)), this, SLOT(enableCellShape(int)));

    connect(ui->toolGenerate, SIGNAL(released()), this, SLOT(generatePhotomosaic()));
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

    progressBar->setRange(0, filenames.size());
    progressBar->setValue(0);
    progressBar->setVisible(true);
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
        cv::Mat resizedImage = UtilityFuncs::resizeImage(image, imageSize, imageSize,
                                                         UtilityFuncs::ResizeType::EXCLUSIVE);

        //Extracts filename and extension from full path
        QString name;
        for (auto it = filename.cbegin() + filename.lastIndexOf('/') + 1, end = filename.cend();
             it != end; ++it)
        {
            name += (*it);
        }

        //Add image to library
        QListWidgetItem listItem(QIcon(UtilityFuncs::matToQPixmap(resizedImage)), name);
        ui->listPhoto->addItem(new QListWidgetItem(listItem));

        //Store QListWidgetItem with resized and original OpenCV Mat
        allImages.insert(listItem, {resizedImage, image});

        progressBar->setValue(progressBar->value() + 1);
    }
    progressBar->setVisible(false);

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

    progressBar->setValue(0);
    progressBar->setVisible(true);

#ifdef OPENCV_WITH_CUDA
    progressBar->setRange(0, allImages.size() * 2);

    //Upload all Mat to GpuMat
    std::vector<cv::cuda::GpuMat> src, dst(static_cast<size_t>(allImages.size()));
    for (auto listItem: allImages.keys())
        src.push_back(cv::cuda::GpuMat(allImages[listItem].second));

    //Calculates resize factor
    double resizeFactor = static_cast<double>(imageSize) / src.front().rows;
    if (imageSize < resizeFactor * src.front().cols)
        resizeFactor = static_cast<double>(imageSize) / src.front().cols;

    //Use INTER_AREA for decreasing, INTER_CUBIC for increasing
    cv::InterpolationFlags flags = (resizeFactor < 1) ? cv::INTER_AREA : cv::INTER_CUBIC;

    //Resize GpuMat
    auto dstIt = dst.begin(), dstEnd = dst.end();
    for (auto srcIt = src.cbegin(), srcEnd = src.cend();
         srcIt != srcEnd; ++srcIt, ++dstIt)
    {
        cv::cuda::resize(*srcIt, *dstIt, cv::Size(static_cast<int>(resizeFactor * srcIt->cols),
                                                  static_cast<int>(resizeFactor * srcIt->rows)),
                         0, 0, flags);
        progressBar->setValue(progressBar->value() + 1);
    }

    //Download resized GpuMat to Mat and update GUI
    auto it = dst.cbegin();
    for (auto listItem: allImages.keys())
    {
        it->download(allImages[listItem].first);
        listItem.setIcon(QIcon(UtilityFuncs::matToQPixmap(allImages[listItem].first)));
        ui->listPhoto->addItem(new QListWidgetItem(listItem));
        progressBar->setValue(progressBar->value() + 1);
        ++it;
    }
#else
    progressBar->setRange(0, allImages.size());

    for (auto listItem: allImages.keys())
    {
        allImages[listItem].first = UtilityFuncs::resizeImage(allImages[listItem].second,
                                                         imageSize, imageSize,
                                                         UtilityFuncs::ResizeType::EXCLUSIVE);
        listItem.setIcon(QIcon(UtilityFuncs::matToQPixmap(allImages[listItem].first)));
        ui->listPhoto->addItem(new QListWidgetItem(listItem));
        progressBar->setValue(progressBar->value() + 1);
    }
#endif
    progressBar->setVisible(false);

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

//Enables/disables non-square cell shapes, GUI widgets for choosing
void MainWindow::enableCellShape(int t_state)
{
    ui->lineCellShape->setEnabled(t_state == Qt::Checked);
    ui->toolCellShape->setEnabled(t_state == Qt::Checked);
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

    //Generate Photomosaic
    auto t1 = std::chrono::high_resolution_clock::now();
    cv::Mat mosaic = PhotomosaicGenerator::generate(mainImage, library);
    qDebug() << "Generator time: " << std::chrono::duration_cast<std::chrono::seconds>(
                    std::chrono::high_resolution_clock::now() - t1).count() << "s";
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
