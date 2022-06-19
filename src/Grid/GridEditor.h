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

#pragma once

#include <QMainWindow>

#include "GridEditViewer.h"

namespace Ui {
class GridEditor;
}

class GridEditor : public QMainWindow
{
    Q_OBJECT

public:
    explicit GridEditor(QWidget *parent = nullptr);
    explicit GridEditor(const cv::Mat &t_background, const CellGroup &t_cellGroup,
                        QWidget *t_parent = nullptr);
    ~GridEditor();

signals:
    void gridStateChanged(const GridUtility::MosaicBestFit &t_gridState);

protected:
    //Updates grid
    void showEvent(QShowEvent *event) override;
    //Emit grid state when grid editor is closed
    void closeEvent(QCloseEvent *event) override;

private:
    Ui::GridEditor *ui;
};