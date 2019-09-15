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

    //Creates a string of all supported image types for use with QFileDialog
    imageTypes += tr("Image Files (");
    auto supportedTypes = QImageReader::supportedImageFormats();
    for (auto type: supportedTypes)
        imageTypes += "*." + type + " ";
    imageTypes += ")";
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
                                                          imageTypes);

    progressBar->setRange(0, filenames.size());
    progressBar->setValue(0);
    progressBar->setVisible(true);
    //For all files selected by user load and add to library
    QImageReader imgReader;
    for (auto filename: filenames)
    {
        imgReader.setFileName(filename);
        if (QImage img = imgReader.read(); !img.isNull())
        {
            //Crops image to square, with center origin
            if (img.width() < img.height())
            {
                int size = img.width();
                img = img.copy(0, (img.height() - size)/2, size, size);
            }
            else
            {
                int size = img.height();
                img = img.copy((img.width() - size)/2, 0, size, size);
            }

            QPixmap pixmap = QPixmap::fromImage(img);

            //Extracts filename and extension from full path
            QString name;
            for (auto it = filename.cbegin() + filename.lastIndexOf('/') + 1, end = filename.cend();
                 it != end; ++it)
            {
                name += (*it);
            }

            //Resize image then add to library
            QListWidgetItem listItem(
                        QIcon(pixmap.scaled(QSize(imageSize, imageSize),
                                            Qt::AspectRatioMode::KeepAspectRatioByExpanding,
                                            Qt::TransformationMode::SmoothTransformation)), name);
            ui->listPhoto->addItem(new QListWidgetItem(listItem));

            //Store original image
            originalImages.insert(listItem, pixmap);
        }
        progressBar->setValue(progressBar->value() + 1);
    }
    progressBar->setVisible(false);

    //Update status bar with new number of images
    ui->statusbar->showMessage(QString::number(originalImages.size()) + tr(" images"));
}

//Deletes selected images
void MainWindow::deleteImages()
{
    auto selectedItems = ui->listPhoto->selectedItems();
    for (auto item: selectedItems)
        originalImages.remove(*item);

    qDeleteAll(selectedItems);

    //Update status bar with new number of images
    ui->statusbar->showMessage(QString::number(originalImages.size()) + tr(" images"));
}

void MainWindow::updateCellSize()
{
    imageSize = ui->spinLibCellSize->value();
    ui->listPhoto->clear();

    progressBar->setRange(0, originalImages.size());
    progressBar->setValue(0);
    progressBar->setVisible(true);
    for (auto image: originalImages.keys())
    {
        image.setIcon(QIcon(originalImages.value(image).
                            scaled(QSize(imageSize, imageSize),
                                   Qt::AspectRatioMode::KeepAspectRatioByExpanding,
                                   Qt::TransformationMode::SmoothTransformation)));
        ui->listPhoto->addItem(new QListWidgetItem(image));
        progressBar->setValue(progressBar->value() + 1);
    }
    progressBar->setVisible(false);
}

// MIL = Mosaic Image Library
void MainWindow::saveLibrary()
{
    QString filename = QFileDialog::getSaveFileName(this, tr("Save image library"), "",
                                                    tr("Image Files") + " (*.mil)");
    QFile file(filename);
    file.open(QIODevice::WriteOnly);
    if (file.isWritable())
    {
        QDataStream out(&file);
        //Write header with "magic number" and version
        out << static_cast<quint32>(0xADBE2480);
        out << static_cast<qint32>(100);

        out.setVersion(QDataStream::Qt_5_13);
        //Write images and names
        out << imageSize;
        out << originalImages.size();
        progressBar->setRange(0, originalImages.size());
        progressBar->setValue(0);
        progressBar->setVisible(true);
        for (auto image: originalImages.keys())
        {
            out << image.icon().pixmap(QSize(imageSize, imageSize));
            out << image.text();
            progressBar->setValue(progressBar->value() + 1);
        }
        progressBar->setVisible(false);
        file.close();
    }
}

void MainWindow::loadLibrary()
{
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
        if (version == 100)
            in.setVersion(QDataStream::Qt_5_13);
        else
            return;

        originalImages.clear();

        //Read images and names
        in >> imageSize;
        ui->spinLibCellSize->setValue(imageSize);
        int numberOfImage;
        in >> numberOfImage;
        progressBar->setRange(0, numberOfImage);
        progressBar->setValue(0);
        progressBar->setVisible(true);
        while (numberOfImage > 0)
        {
            --numberOfImage;
            QPixmap pixmap;
            in >> pixmap;

            QString name;
            in >> name;

            QListWidgetItem listItem(
                        QIcon(pixmap.scaled(QSize(imageSize, imageSize),
                                            Qt::AspectRatioMode::KeepAspectRatioByExpanding,
                                            Qt::TransformationMode::SmoothTransformation)), name);
            ui->listPhoto->addItem(new QListWidgetItem(listItem));

            //Store original image
            originalImages.insert(listItem, pixmap);
            progressBar->setValue(progressBar->value() + 1);
        }
        progressBar->setVisible(false);

        file.close();
    }
    //Update status bar with new number of images
    ui->statusbar->showMessage(QString::number(originalImages.size()) + tr(" images"));
}
