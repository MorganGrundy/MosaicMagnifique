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

#include "imagelibraryeditor.h"
#include "ui_imagelibraryeditor.h"

#include <QFileDialog>
#include <QDebug>
#include <QMessageBox>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "imageutility.h"
#include "cellshape.h"
#include "imagesquarer.h"

ImageLibraryEditor::ImageLibraryEditor(QWidget *parent) :
    QWidget(parent),
    ui{new Ui::ImageLibraryEditor}, m_progressBar{nullptr}, m_cropMode{CropMode::Center}
{
    ui->setupUi(this);

    //Setup image library list
    ui->listPhoto->setResizeMode(QListWidget::ResizeMode::Adjust);
    ui->listPhoto->setIconSize(ui->listPhoto->gridSize() - QSize(14, 14));
    m_imageSize = CellShape::DEFAULT_CELL_SIZE;
    ui->spinLibCellSize->setValue(m_imageSize);

    //Connects image library tab buttons to appropriate methods
    connect(ui->comboCropMode, &QComboBox::currentTextChanged,
            this, &ImageLibraryEditor::changeCropMode);
    connect(ui->buttonAdd, &QPushButton::released, this, &ImageLibraryEditor::addImages);
    connect(ui->buttonDelete, &QPushButton::released, this, &ImageLibraryEditor::deleteImages);
    connect(ui->buttonLibCellSize, &QPushButton::released,
            this, &ImageLibraryEditor::updateCellSize);
    connect(ui->buttonSave, &QPushButton::released, this, &ImageLibraryEditor::saveLibrary);
    connect(ui->buttonLoad, &QPushButton::released, this, &ImageLibraryEditor::loadLibrary);

    //Create manual image squarer
    m_imageSquarer = new ImageSquarer(this);
    m_imageSquarer->setWindowModality(Qt::WindowModality::ApplicationModal);
}

ImageLibraryEditor::~ImageLibraryEditor()
{
    delete ui;
}

//Sets pointer to progress bar
void ImageLibraryEditor::setProgressBar(QProgressBar *t_progressBar)
{
    m_progressBar = t_progressBar;
}

//Returns size of image library
size_t ImageLibraryEditor::getImageLibrarySize() const
{
    return m_images.size();
}

//Returns image library
const std::vector<cv::Mat> ImageLibraryEditor::getImageLibrary() const
{
    std::vector<cv::Mat> imageLibrary;

    for (const auto &image: m_images)
        imageLibrary.push_back(image.resizedImage);

    return imageLibrary;
}

//Changes the cropping mode used for library images
void ImageLibraryEditor::changeCropMode(const QString &t_mode)
{
    if (t_mode == "Manual")
        m_cropMode = CropMode::Manual;
    else if (t_mode == "Center")
        m_cropMode = CropMode::Center;
    else if (t_mode == "Features")
        m_cropMode = CropMode::Features;
    else if (t_mode == "Entropy")
        m_cropMode = CropMode::Entropy;
    else if (t_mode == "Cascade Classifier")
    {
        QDir defaultDir = QDir::current();
        defaultDir.cdUp();
        defaultDir.cd("CascadeClassifier");
        //Prompt user for cascade classifier file
        QString filename = QFileDialog::getOpenFileName(this, tr("Select cascade classifier file"),
                                                        defaultDir.path() +
                                                            "/haarcascade_frontalface_default.xml",
                                                        "Cascade Classifier (*.xml)");

        //Load cascade classifier file
        if (!filename.isNull())
        {
            if (!m_imageSquarer->loadCascade(filename.toStdString()))
            {
                //Failed to load, reset crop mode
                ui->comboCropMode->setCurrentIndex(0);
                QMessageBox msgBox;
                msgBox.setText(tr("The cascade classifier \"") + filename +
                               tr("\" failed to load"));
                msgBox.exec();
                return;
            }
        }
        else
        {
            //User exited prompt, reset crop mode
            ui->comboCropMode->setCurrentIndex(0);
            return;
        }

        //Cascade classifier successfully loaded
        m_cropMode = CropMode::CascadeClassifier;
    }
    else
        qDebug() << Q_FUNC_INFO << "Crop mode" << t_mode << "was not recognised";
}

//Loads images
void ImageLibraryEditor::addImages()
{
    //Get path to image files from user
    QStringList files = QFileDialog::getOpenFileNames(this, tr("Select image to add"), "",
                                                          "Image Files (*.bmp *.dib *.jpeg *.jpg "
                                                          "*.jpe *.jp2 *.png *.pbm *.pgm *.ppm "
                                                          "*.pxm *.pnm *.sr *.ras *.tiff *.tif "
                                                          "*.hdr *.pic)");

    //Initialise progress bar
    if (m_progressBar != nullptr)
    {
        m_progressBar->setMaximum(files.size());
        m_progressBar->setValue(0);
        m_progressBar->setVisible(true);
    }

    std::vector<cv::Mat> originalImages;
    std::vector<cv::Mat> resizedImages;
    std::vector<QString> names;
    //For all files selected by user load and add to library
    for (const auto &file: files)
    {
        //Load image
        cv::Mat image = cv::imread(file.toStdString());
        if (image.empty())
        {
            qDebug() << Q_FUNC_INFO << "Could not open or find the image.";
            continue;
        }

        //Square image
        bool imageWasSquared = false;
        if (m_cropMode == CropMode::Manual)
        {
            //Allow user to manually crop image
            if (!m_imageSquarer->squareManual(image, image))
            {
                //Image squarer window was closed, cancel adding images
                m_progressBar->setVisible(false);
                return;
            }

            imageWasSquared = true;
        }
        else if (m_cropMode == CropMode::Features)
            imageWasSquared = m_imageSquarer->squareToFeatures(image, image);
        else if (m_cropMode == CropMode::Entropy)
            imageWasSquared = m_imageSquarer->squareToEntropy(image, image);
        else if (m_cropMode == CropMode::CascadeClassifier)
            imageWasSquared = m_imageSquarer->squareToCascadeClassifier(image, image);

        //Center crop also used as fallback if other modes fail
        if (!imageWasSquared)
            ImageUtility::imageToSquare(image, ImageUtility::SquareMethod::CROP);

        if (!image.empty())
        {
            originalImages.push_back(image);

            //Extracts filename and extension from full path
            names.push_back(file.right(file.size() - file.lastIndexOf('/') - 1));
        }

        if (m_progressBar != nullptr)
            m_progressBar->setValue(m_progressBar->value() + 1);

        //Helps prevent 'not responding'
        QCoreApplication::processEvents();
    }
    m_imageSquarer->hide();

    //Resize images to current library size
    ImageUtility::batchResizeMat(originalImages, resizedImages, m_imageSize, m_imageSize,
                                 ImageUtility::ResizeType::EXACT, m_progressBar);

    //Add images to list widget
    auto nameIt = names.cbegin();
    auto imageIt = resizedImages.cbegin();
    for (const auto &image: originalImages)
    {
        //Add image to library
        m_images.push_back(LibraryImage(image, *imageIt, *nameIt));
        ui->listPhoto->addItem(m_images.back().listWidget.get());

        ++nameIt;
        ++imageIt;
    }

    //Emit new image library size
    emit imageLibraryChanged(m_images.size());
}

//Deletes selected images
void ImageLibraryEditor::deleteImages()
{
    auto selectedItems = ui->listPhoto->selectedItems();

    //Delete images
    for (auto item: selectedItems)
    {
        m_images.erase(std::remove_if(m_images.begin(), m_images.end(),
                                      [&item](const LibraryImage &t_libraryImage)
                                      {
                                          return t_libraryImage.listWidget.get() == item;
                                      }), m_images.end());
    }

    ui->listPhoto->update();

    //Emit new image library size
    emit imageLibraryChanged(m_images.size());
}

//Resizes image library
void ImageLibraryEditor::updateCellSize()
{
    //Size unchanged
    if (m_imageSize == ui->spinLibCellSize->value())
        return;

    m_imageSize = ui->spinLibCellSize->value();

    //Get original images
    std::vector<cv::Mat> images;
    for (const auto &image: m_images)
        images.push_back(image.resizedImage);

    //Resize images
    ImageUtility::batchResizeMat(images, images, m_imageSize, m_imageSize,
                                 ImageUtility::ResizeType::EXACT, m_progressBar);

    //Update image library with new resized images
    auto it = images.cbegin();
    for (auto &image: m_images)
    {
        image.resizedImage = *it;
        image.listWidget->setIcon(QIcon(ImageUtility::matToQPixmap(*it)));

        ++it;
    }

    ui->listPhoto->update();
}

//Saves the image library to a file
void ImageLibraryEditor::saveLibrary()
{
    //Get path to save file from user, mil = Mosaic Image Library
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
            out << static_cast<quint32>(ImageUtility::MIL_MAGIC);
            out << static_cast<qint32>(ImageUtility::MIL_VERSION);

            out.setVersion(QDataStream::Qt_5_0);

            //Write image size and library size
            out << m_imageSize;
            out << m_images.size();

            //Initialise progress bar
            if (m_progressBar != nullptr)
            {
                m_progressBar->setRange(0, static_cast<int>(m_images.size()));
                m_progressBar->setValue(0);
                m_progressBar->setVisible(true);
            }

            //Write images and names
            for (const auto &image: m_images)
            {
                out << image.resizedImage;
                out << image.listWidget->text();

                if (m_progressBar != nullptr)
                    m_progressBar->setValue(m_progressBar->value() + 1);
            }

            if (m_progressBar != nullptr)
                m_progressBar->setVisible(false);

            file.close();
        }
    }
}

//Loads an image library from a file
void ImageLibraryEditor::loadLibrary()
{
    //Get path to mil file from user
    QString filename = QFileDialog::getOpenFileName(this, tr("Select image library to load"), "",
                                                    tr("Mosaic Image Library") + " (*.mil)");

    if (!filename.isNull())
    {
        //Check for valid file
        QFile file(filename);
        file.open(QIODevice::ReadOnly);
        if (file.isReadable())
        {
            QDataStream in(&file);

            //Read and check magic number
            quint32 magic;
            in >> magic;
            if (magic != ImageUtility::MIL_MAGIC)
            {
                QMessageBox msgBox;
                msgBox.setText(filename + tr(" is not a valid .mil file"));
                msgBox.exec();
                return;
            }

            //Read the version
            qint32 version;
            in >> version;
            if (version <= ImageUtility::MIL_VERSION && version >= 2)
                in.setVersion(QDataStream::Qt_5_0);
            else
            {
                QMessageBox msgBox;
                if (version < ImageUtility::MIL_VERSION)
                    msgBox.setText(filename + tr(" uses an outdated file version"));
                else
                    msgBox.setText(filename + tr(" uses a newer file version"));
                msgBox.exec();
                return;
            }

            //Clear current image library
            m_images.clear();
            ui->listPhoto->clear();

            //Read image size
            in >> m_imageSize;
            ui->spinLibCellSize->setValue(m_imageSize);
            //Read library size
            size_t numberOfImage;
            //mil file version 2 used an int for library size instead of size_t, so need to cast
            if (version == 2)
            {
                int intNumberOfImage;
                in >> intNumberOfImage;
                numberOfImage = static_cast<size_t>(intNumberOfImage);
            }
            else
                in >> numberOfImage;

            //Initialise progress bar
            if (m_progressBar != nullptr)
            {
                m_progressBar->setRange(0, static_cast<int>(numberOfImage));
                m_progressBar->setValue(0);
                m_progressBar->setVisible(true);
            }

            //Read images and names
            while (numberOfImage > 0)
            {
                --numberOfImage;
                cv::Mat image;
                in >> image;

                QString name;
                in >> name;

                //Add to list widget
                m_images.push_back(LibraryImage(image, image, name));
                ui->listPhoto->addItem(m_images.back().listWidget.get());

                if (m_progressBar != nullptr)
                    m_progressBar->setValue(m_progressBar->value() + 1);
            }

            if (m_progressBar != nullptr)
                m_progressBar->setVisible(false);

            file.close();

            //Emit new image library size
            emit imageLibraryChanged(m_images.size());
        }
        else
            qDebug() << Q_FUNC_INFO << "Could not read file:" << filename;
    }
}
