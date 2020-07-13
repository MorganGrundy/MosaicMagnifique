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
    along with Foobar.  If not, see <https://www.gnu.org/licenses/>.
*/

#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <memory>
#include <QMap>
#include <QListWidgetItem>
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
    void tabChanged(int t_index);

    //Custom Cell Shapes tab
    void saveCellShape();
    void loadCellShape();
    void cellNameChanged(const QString &text);
    void selectCellMask();
    //Cell spacing
    void cellSpacingColChanged(int t_value);
    void cellSpacingRowChanged(int t_value);
    //Cell alternate offset
    void cellAlternateColOffsetChanged(int t_value);
    void cellAlternateRowOffsetChanged(int t_value);
    //Cell flipping
    void cellColumnFlipHorizontalChanged(bool t_state);
    void cellColumnFlipVerticalChanged(bool t_state);
    void cellRowFlipHorizontalChanged(bool t_state);
    void cellRowFlipVerticalChanged(bool t_state);
    //Cell alternate spacing
    void enableCellAlternateRowSpacing(bool t_state);
    void enableCellAlternateColSpacing(bool t_state);
    void cellAlternateRowSpacingChanged(int t_value);
    void cellAlternateColSpacingChanged(int t_value);

    //Image Library tab
    void addImages();
    void deleteImages();
    void updateCellSize();
    void saveLibrary();
    void loadLibrary();

    //Generator Settings tab
    void selectMainImage();
    void compareColours();
    void photomosaicSizeLink();
    void photomosaicWidthChanged(int i);
    void photomosaicHeightChanged(int i);
    void loadImageSize();
    void photomosaicDetailChanged(int i);
    void cellSizeChanged(int t_value);
    void minimumCellSizeChanged(int t_value);
    void enableCellShape(bool t_state);
    void CUDADeviceChanged(int t_index);

    void generatePhotomosaic();

private:
    void clampDetail();

    Ui::MainWindow *ui;
    QProgressBar *progressBar;

    bool cellShapeChanged;

    double photomosaicSizeRatio;

    int imageSize;
    QMap<QListWidgetItem, std::pair<cv::Mat, cv::Mat>> allImages;

    cv::Mat mainImage;
};
#endif // MAINWINDOW_H
