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

#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QProgressBar>
#include <opencv2/core/mat.hpp>

#include "cellshape.h"

QT_BEGIN_NAMESPACE
namespace Ui { class MainWindow; }
QT_END_NAMESPACE

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    MainWindow(QWidget *t_parent = nullptr);
    ~MainWindow();

public slots:
    //Updates cell shape in grid preview
    void tabChanged(int t_index);

    //Updates cell shape
    void updateCellShape(const CellShape &t_cellShape);
    //Update cell shape name
    void updateCellName(const QString &t_name);

    //Updates image library count in tab widget
    void updateImageLibraryCount(size_t t_newSize);

    //Prompts user for a main image
    void selectMainImage();
    //Opens colour visualisation window
    void compareColours();

    //Links width and height of photomosaic so they scale together
    //Updates link icon
    void photomosaicSizeLink();
    //Updates photomosaic width
    void photomosaicWidthChanged(int i);
    //Updates photomosaic height
    void photomosaicHeightChanged(int i);
    //Sets photomosaic size to current main image size
    void loadImageSize();

    //Updates detail level
    void photomosaicDetailChanged(int i);

    //Updates cell size
    void cellSizeChanged(int t_value);
    //Updates cell grid size steps
    void minimumCellSizeChanged(int t_value);
    //Enables/disables custom cell shapes
    void enableCellShape(bool t_state);

    //Allows user to manually edit current cell grid
    void editCellGrid();

#ifdef CUDA
    //Changes CUDA device
    void CUDADeviceChanged(int t_index);
#endif

    //Generate and display a Photomosaic for current settings
    void generatePhotomosaic();

private:
    //Clamps detail level so that cell size never reaches 0px
    void clampDetail();

    Ui::MainWindow *ui;
    QProgressBar *progressBar;

    bool cellShapeChanged;
    CellShape newCellShape;

    double photomosaicSizeRatio;

    cv::Mat mainImage;
};
#endif // MAINWINDOW_H
