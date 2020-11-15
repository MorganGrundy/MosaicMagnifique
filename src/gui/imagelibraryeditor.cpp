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
    ui{new Ui::ImageLibraryEditor}, m_progressBar{nullptr}, m_cropMode{CropMode::Center},
    m_images{CellShape::DEFAULT_CELL_SIZE}
{
    ui->setupUi(this);

    //Setup image library list
    ui->listPhoto->setResizeMode(QListWidget::ResizeMode::Adjust);
    ui->listPhoto->setIconSize(ui->listPhoto->gridSize() - QSize(14, 14));
    ui->spinLibCellSize->setValue(CellShape::DEFAULT_CELL_SIZE);

    //Connects image library tab buttons to appropriate methods
    connect(ui->buttonSave, &QPushButton::released, this, &ImageLibraryEditor::saveLibrary);
    connect(ui->buttonLoad, &QPushButton::released, this, &ImageLibraryEditor::loadLibrary);

    connect(ui->comboCropMode, &QComboBox::currentTextChanged,
            this, &ImageLibraryEditor::changeCropMode);
    connect(ui->buttonAdd, &QPushButton::released, this, &ImageLibraryEditor::addImages);
    connect(ui->buttonDelete, &QPushButton::released, this, &ImageLibraryEditor::deleteImages);
    connect(ui->buttonClear, &QPushButton::released, this, &ImageLibraryEditor::clearLibrary);

    connect(ui->buttonLibCellSize, &QPushButton::released,
            this, &ImageLibraryEditor::updateCellSize);

    //Create manual image squarer
    m_imageSquarer = new ImageSquarer(this);
    m_imageSquarer->setWindowModality(Qt::WindowModality::ApplicationModal);

    m_featureDetector = cv::FastFeatureDetector::create();
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
    return m_images.getImages().size();
}

//Returns image library
const std::vector<cv::Mat> ImageLibraryEditor::getImageLibrary() const
{
    return m_images.getImages();
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
            if (!m_cascadeClassifier.load(filename.toStdString()))
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

        try
        {
            if (m_cropMode == CropMode::Manual)
            {
                //Allow user to manually crop image
                if (!m_imageSquarer->square(image, image))
                {
                    //Image squarer window was closed, cancel adding images
                    if (m_progressBar != nullptr)
                        m_progressBar->setVisible(false);
                    return;
                }
            }
            else if (m_cropMode == CropMode::Features)
                ImageUtility::squareToFeatures(image, image, m_featureDetector);
            else if (m_cropMode == CropMode::Entropy)
                ImageUtility::squareToEntropy(image, image);
            else if (m_cropMode == CropMode::CascadeClassifier)
                ImageUtility::squareToCascadeClassifier(image, image, m_cascadeClassifier);
        }
        catch (const std::invalid_argument &e)
        {
            qDebug() << Q_FUNC_INFO << e.what();
        }


        //Extracts filename and extension from full path
        QString imageName = file.right(file.size() - file.lastIndexOf('/') - 1);

        try
        {
            //Add image to library
            m_images.addImage(image, imageName);

            //Add image to list widget
            m_imageWidgets.push_back(std::make_shared<QListWidgetItem>(
                QIcon(ImageUtility::matToQPixmap(m_images.getImages().back())), imageName));
            ui->listPhoto->addItem(m_imageWidgets.back().get());
        }
        catch (const std::invalid_argument &e)
        {
            qDebug() << Q_FUNC_INFO << e.what();
        }

        if (m_progressBar != nullptr)
            m_progressBar->setValue(m_progressBar->value() + 1);

        QCoreApplication::processEvents(QEventLoop::ProcessEventsFlag::ExcludeUserInputEvents);
    }

    if (m_cropMode == CropMode::Manual)
        m_imageSquarer->hide();

    if (m_progressBar != nullptr)
        m_progressBar->setVisible(false);

    //Emit new image library size
    emit imageLibraryChanged(m_images.getImages().size());
}

//Deletes selected images
void ImageLibraryEditor::deleteImages()
{
    auto selectedItems = ui->listPhoto->selectedItems();

    //Get image indices
    std::vector<size_t> imageIndices;
    for (const auto item: selectedItems)
    {
        auto it = std::find_if(m_imageWidgets.cbegin(), m_imageWidgets.cend(),
                               [item](const std::shared_ptr<QListWidgetItem> t_item)
                               {
                                   return t_item.get() == item;
                               });
        if (it != m_imageWidgets.cend())
        {
            //Calculate item index
            imageIndices.push_back(std::distance(m_imageWidgets.cbegin(), it));
        }
    }

    //Sort indices in descending order
    std::sort(imageIndices.begin(), imageIndices.end(), std::greater<size_t>());

    //Remove from image library and widgets
    for (const auto index: imageIndices)
    {
        m_imageWidgets.erase(m_imageWidgets.begin() + index);
        m_images.removeAtIndex(index);
    }

    ui->listPhoto->update();

    //Emit new image library size
    emit imageLibraryChanged(m_images.getImages().size());
}

//Clears the image library
void ImageLibraryEditor::clearLibrary()
{
    m_images.clear();
    m_imageWidgets.clear();

    ui->listPhoto->update();

    //Emit new image library size
    emit imageLibraryChanged(m_images.getImages().size());
}

//Resizes image library
void ImageLibraryEditor::updateCellSize()
{
    //Size unchanged
    if (m_images.getImageSize() == static_cast<size_t>(ui->spinLibCellSize->value()))
        return;

    //Resize image library
    m_images.setImageSize(ui->spinLibCellSize->value());

    //Update image library with new resized images
    for (auto [widgetIt, imageIt] = std::pair{m_imageWidgets.begin(),
                                              m_images.getImages().begin()};
         widgetIt != m_imageWidgets.end(); ++widgetIt, ++imageIt)
    {
        widgetIt->get()->setIcon(QIcon(ImageUtility::matToQPixmap(*imageIt)));
    }

    ui->listPhoto->update();
}

//Saves the image library to a file
void ImageLibraryEditor::saveLibrary()
{
    //Get path to save file from user, mil = Mosaic Image Library
    QString filename = QFileDialog::getSaveFileName(this, tr("Save image library"), "",
                                                    tr("Mosaic Image Library") + " (*.mil)");

    try
    {
        m_images.saveToFile(filename);
    }
    catch (const std::invalid_argument &e)
    {
        QMessageBox msgBox;
        msgBox.setText(tr(e.what()));
        msgBox.exec();

        return;
    }
}

//Loads an image library from a file
void ImageLibraryEditor::loadLibrary()
{
    //Get path to mil file from user
    QString filename = QFileDialog::getOpenFileName(this, tr("Select image library to load"), "",
                                                    tr("Mosaic Image Library") + " (*.mil)");

    const size_t libSizeBefore = m_images.getImages().size();
    try
    {
        m_images.loadFromFile(filename);
    }
    catch (const std::invalid_argument &e)
    {
        QMessageBox msgBox;
        msgBox.setText(tr(e.what()));
        msgBox.exec();

        return;
    }

    ui->spinLibCellSize->setValue(static_cast<int>(m_images.getImageSize()));

    //Add images to list widget
    for (auto [imageIt, nameIt] = std::pair{m_images.getImages().cbegin() + libSizeBefore,
                                            m_images.getNames().cbegin() + libSizeBefore};
         imageIt != m_images.getImages().cend(); ++imageIt, ++nameIt)
    {
        m_imageWidgets.push_back(std::make_shared<QListWidgetItem>(
            QIcon(ImageUtility::matToQPixmap(*imageIt)), *nameIt));
        ui->listPhoto->addItem(m_imageWidgets.back().get());
    }

    //Emit new image library size
    emit imageLibraryChanged(m_images.getImages().size());
}
