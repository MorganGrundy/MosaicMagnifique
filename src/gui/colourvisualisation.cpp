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

#include "colourvisualisation.h"
#include "ui_colourvisualisation.h"

#include "imagehistogramcompare.h"

ColourVisualisation::ColourVisualisation(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::ColourVisualisation)
{
    ui->setupUi(this);
}

ColourVisualisation::ColourVisualisation(QWidget *parent, const cv::Mat &t_image,
                                         const std::vector<cv::Mat> &t_libImages) :
    QMainWindow(parent),
    ui(new Ui::ColourVisualisation)
{
    ui->setupUi(this);

    ImageHistogramCompare::colourPriorityList colourPriority =
        ImageHistogramCompare::getColourPriorityList(t_image, t_libImages);

    //Add all colours to list
    for (auto data: colourPriority)
    {
        //Create square image of bin colour (using bin median colour)
        QPixmap colour(iconSize, iconSize);
        colour.fill(QColor(std::get<0>(data.first), std::get<1>(data.first),
                           std::get<2>(data.first)));

        QListWidgetItem *listItem = new QListWidgetItem(QIcon(colour), QString());
        ui->listWidget->addItem(listItem);
    }
}

ColourVisualisation::~ColourVisualisation()
{
    delete ui;
}
