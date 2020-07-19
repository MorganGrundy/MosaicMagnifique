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

#ifndef COLOURVISUALISATION_H
#define COLOURVISUALISATION_H

#include <QMainWindow>
#include <opencv2/core/mat.hpp>

namespace Ui {
class ColourVisualisation;
}

class ColourVisualisation : public QMainWindow
{
    Q_OBJECT

public:
    explicit ColourVisualisation(QWidget *parent = nullptr);
    explicit ColourVisualisation(QWidget *parent, const cv::Mat &t_image, cv::Mat *t_libImages,
                                 const size_t t_noLib);
    ~ColourVisualisation();

private:
    void createColourList();

    Ui::ColourVisualisation *ui;

    cv::Mat colours;

    cv::Mat mainHistogram, libraryHistogram;

    //Number of bins in histogram
    const int noOfBins = 30;
    const int histogramSize[3] = {noOfBins, noOfBins, noOfBins};

    //Histogram range
    const float RGBRanges[2] = {0, 256};
    const float *ranges[3] = {RGBRanges, RGBRanges, RGBRanges};

    const int channels[3] = {0, 1, 2};

    const int iconSize = 100;
};

#endif // COLOURVISUALISATION_H
