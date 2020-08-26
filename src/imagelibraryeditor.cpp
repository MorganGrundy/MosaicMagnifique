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

#include "imageutility.h"
#include "cellshape.h"

ImageLibraryEditor::ImageLibraryEditor(QWidget *parent) :
    QWidget(parent),
    ui{new Ui::ImageLibraryEditor}, m_progressBar{nullptr}
{
    ui->setupUi(this);

    //Setup image library list
    ui->listPhoto->setResizeMode(QListWidget::ResizeMode::Adjust);
    ui->listPhoto->setIconSize(ui->listPhoto->gridSize() - QSize(14, 14));
    m_imageSize = CellShape::DEFAULT_CELL_SIZE;
    ui->spinLibCellSize->setValue(m_imageSize);

    //Connects image library tab buttons to appropriate methods
    connect(ui->buttonAdd, &QPushButton::released, this, &ImageLibraryEditor::addImages);
    connect(ui->buttonDelete, &QPushButton::released, this, &ImageLibraryEditor::deleteImages);
    connect(ui->buttonLibCellSize, &QPushButton::released,
            this, &ImageLibraryEditor::updateCellSize);
    connect(ui->buttonSave, &QPushButton::released, this, &ImageLibraryEditor::saveLibrary);
    connect(ui->buttonLoad, &QPushButton::released, this, &ImageLibraryEditor::loadLibrary);
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
    for (auto pair: m_images.values())
        imageLibrary.push_back(pair.first);

    return imageLibrary;
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
    for (auto file: files)
    {
        //Load image
        cv::Mat image = cv::imread(file.toStdString());
        if (image.empty())
        {
            qDebug() << "Could not open or find the image";
            continue;
        }

        //Square image
        ImageUtility::imageToSquare(image, ImageUtility::SquareMethod::CROP);
        originalImages.push_back(image);

        //Extracts filename and extension from full path
        names.push_back(file.right(file.size() - file.lastIndexOf('/') - 1));

        if (m_progressBar != nullptr)
            m_progressBar->setValue(m_progressBar->value() + 1);
    }

    //Resize images to current library size
    resizedImages = ImageUtility::batchResizeMat(originalImages, m_imageSize, m_imageSize,
                                                 ImageUtility::ResizeType::EXACT, m_progressBar);

    //Add images to list widget
    auto nameIt = names.cbegin();
    auto imageIt = resizedImages.cbegin();
    for (auto image: originalImages)
    {
        //Add image to library
        QListWidgetItem listItem(QIcon(ImageUtility::matToQPixmap(*imageIt)), *nameIt);
        ui->listPhoto->addItem(new QListWidgetItem(listItem));

        //Store QListWidgetItem with resized and original OpenCV Mat
        m_images.insert(listItem, {*imageIt, image});

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
        m_images.remove(*item);
    qDeleteAll(selectedItems);

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
    ui->listPhoto->clear();

    //Resize images
    std::vector<cv::Mat> images =
        ImageUtility::batchResizeMat(getImageLibrary(), m_imageSize, m_imageSize,
                                     ImageUtility::ResizeType::EXACT, m_progressBar);

    //Update list widget
    auto it = images.cbegin();
    for (auto listItem: m_images.keys())
    {
        m_images[listItem].first = *it;
        listItem.setIcon(QIcon(ImageUtility::matToQPixmap(*it)));
        ui->listPhoto->addItem(new QListWidgetItem(listItem));

        ++it;
    }
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
                m_progressBar->setRange(0, m_images.size());
                m_progressBar->setValue(0);
                m_progressBar->setVisible(true);
            }

            //Write images and names
            for (auto listItem: m_images.keys())
            {
                out << m_images[listItem].first;
                out << listItem.text();

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
            if (version == ImageUtility::MIL_VERSION)
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
            int numberOfImage;
            in >> numberOfImage;

            //Initialise progress bar
            if (m_progressBar != nullptr)
            {
                m_progressBar->setRange(0, numberOfImage);
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
                QListWidgetItem listItem(QIcon(ImageUtility::matToQPixmap(image)), name);
                ui->listPhoto->addItem(new QListWidgetItem(listItem));

                m_images.insert(listItem, {image, image});

                if (m_progressBar != nullptr)
                    m_progressBar->setValue(m_progressBar->value() + 1);
            }

            if (m_progressBar != nullptr)
                m_progressBar->setVisible(false);

            file.close();

            //Emit new image library size
            emit imageLibraryChanged(m_images.size());
        }
    }
}
