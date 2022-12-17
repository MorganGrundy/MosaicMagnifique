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

#include "PhotomosaicViewer.h"
#include "ui_PhotomosaicViewer.h"

#include <QFileDialog>
#include <QErrorMessage>
#include <QLabel>
#include <QColorDialog>
#include <opencv2/imgcodecs.hpp>

#include "..\Other\ImageUtility.h"
#include "..\Other\Logger.h"
#include "..\Other\CustomGraphicsView.h"

PhotomosaicViewer::PhotomosaicViewer(
    QWidget *t_parent, std::shared_ptr<PhotomosaicGeneratorBase> t_photomosaicGenerator,
    const double t_duration)
    : QMainWindow{t_parent}, ui{new Ui::PhotomosaicViewer},
    m_photomosaicGenerator{t_photomosaicGenerator}, m_backgroundColor{Qt::black}
{
    ui->setupUi(this);

    //Connects buttons to appropriate methods
    connect(ui->saveButton, &QPushButton::released, this, &PhotomosaicViewer::savePhotomosaic);
    connect(ui->fitButton, &QPushButton::released,
            ui->graphicsView, &CustomGraphicsView::fitToView);

    //Setup background colour button
    connect(ui->pushBackgroundColour, &QPushButton::released,
            this, &PhotomosaicViewer::openColourSelector);
    //Change button colour
    ui->pushBackgroundColour->setStyleSheet("background-color: rgb(0, 0, 0)");

    if (t_duration > 0)
    {
        //Displays the generation time on statusbar
        QLabel *label = new QLabel();
        label->setText(tr("Generated in ") + QString::number(t_duration) + "s");
        ui->statusbar->addPermanentWidget(label);
    }

    //Create scene
    scene = new QGraphicsScene();
    scene->setItemIndexMethod(QGraphicsScene::NoIndex);
    ui->graphicsView->setScene(scene);
    //Show Photomosaic
    updatePhotomosaic();
}

PhotomosaicViewer::PhotomosaicViewer(QWidget *t_parent)
    : QMainWindow{t_parent}, ui{new Ui::PhotomosaicViewer}
{
    ui->setupUi(this);
}

PhotomosaicViewer::~PhotomosaicViewer()
{
    delete ui;
    LogInfo("Closed Photomosaic Viewer.");
}

//Allows user to save the Photomosaic as an image file
void PhotomosaicViewer::savePhotomosaic()
{
    QString filename = QFileDialog::getSaveFileName(this, tr("Save Photomosaic"), "",
                                                    "Image Files (*.bmp *.dib *.jpeg *.jpg "
                                                    "*.jpe *.jp2 *.png *.pbm *.pgm *.ppm "
                                                    "*.pxm *.pnm *.sr *.ras *.tiff *.tif "
                                                    "*.hdr *.pic)");
    if (!filename.isNull())
    {
        if (!cv::imwrite(filename.toStdString(), m_photomosaic))
        {
            QErrorMessage errorMsg(this);
            errorMsg.showMessage("Failed to save Photomosaic to: " + filename);
        }
    }
}

//Opens colour selector for setting background colour
void PhotomosaicViewer::openColourSelector()
{
    const QColor newColour = QColorDialog::getColor(
        m_backgroundColor, this, "Select a background colour:",
        QColorDialog::ColorDialogOption::ShowAlphaChannel);

    //Check for valid and different colour
    if (newColour.isValid() && newColour != m_backgroundColor)
    {
        m_backgroundColor = newColour;

        //Change button colour
        QString newStyle = QString("background-color: rgba(%1, %2, %3, %4)");
        newStyle = newStyle.arg(QString::number(m_backgroundColor.red()))
                           .arg(QString::number(m_backgroundColor.green()))
                           .arg(QString::number(m_backgroundColor.blue()))
                           .arg(QString::number(m_backgroundColor.alpha()));
        ui->pushBackgroundColour->setStyleSheet(newStyle);

        updatePhotomosaic();
    }
}

void PhotomosaicViewer::show()
{
    QMainWindow::show();
    LogInfo("Opened Photomosaic Viewer.");
}

//Creates Photomosaic and displays
void PhotomosaicViewer::updatePhotomosaic()
{
    //Build Photomosaic with new background colour
    cv::Scalar opencvColour(m_backgroundColor.blue(), m_backgroundColor.green(),
                            m_backgroundColor.red(), m_backgroundColor.alpha());
    m_photomosaic = m_photomosaicGenerator->buildPhotomosaic(opencvColour);

    //Update viewer
    scene->clear();
    const QPixmap pixmap = ImageUtility::matToQPixmap(m_photomosaic, QImage::Format_RGBA8888);
    scene->addPixmap(pixmap);
    ui->graphicsView->setSceneRect(pixmap.rect());
}
