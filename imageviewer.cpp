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

#include "imageviewer.h"
#include "ui_imageviewer.h"

#include <QFileDialog>
#include <QLabel>
#include <opencv2/highgui.hpp>

#include "utilityfuncs.h"
#include "customgraphicsview.h"

ImageViewer::ImageViewer(QWidget *t_parent)
    : QMainWindow(t_parent), ui(new Ui::ImageViewer)
{
    ui->setupUi(this);

    connect(ui->saveButton, SIGNAL(released()), this, SLOT(saveImage()));
    connect(ui->fitButton, SIGNAL(released()), ui->graphicsView, SLOT(fitToView()));
}

ImageViewer::ImageViewer(QWidget *t_parent, const cv::Mat &t_image, const double t_duration)
    : QMainWindow(t_parent), ui(new Ui::ImageViewer), image(t_image)
{
    ui->setupUi(this);
    //Displays the generation time on statusbar
    QLabel *label = new QLabel();
    label->setText(tr("Generated in ") + QString::number(t_duration) + "s");
    ui->statusbar->addPermanentWidget(label);

    //Connects buttons to appropriate methods
    connect(ui->saveButton, SIGNAL(released()), this, SLOT(saveImage()));
    connect(ui->fitButton, SIGNAL(released()), ui->graphicsView, SLOT(fitToView()));

    //Adds image to view
    const QPixmap pixmap = UtilityFuncs::matToQPixmap(t_image);
    QGraphicsScene *scene = new QGraphicsScene();
    scene->setItemIndexMethod(QGraphicsScene::NoIndex);
    scene->addPixmap(pixmap);
    ui->graphicsView->setScene(scene);
    ui->graphicsView->setSceneRect(pixmap.rect());
}

ImageViewer::~ImageViewer()
{
    delete ui;
}

//Allows user to save the Photomosaic as an image file
void ImageViewer::saveImage()
{
    QString filename = QFileDialog::getSaveFileName(this, tr("Save Photomosaic"), "",
                                                    "Image Files (*.bmp *.dib *.jpeg *.jpg "
                                                    "*.jpe *.jp2 *.png *.pbm *.pgm *.ppm "
                                                    "*.pxm *.pnm *.sr *.ras *.tiff *.tif "
                                                    "*.hdr *.pic)");
    cv::imwrite(filename.toStdString(), image);
}
