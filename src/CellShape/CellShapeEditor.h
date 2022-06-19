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

#include <QWidget>

#include "CellShape.h"

namespace Ui {
class CellShapeEditor;
}

class CellShapeEditor : public QWidget
{
    Q_OBJECT

public:
    explicit CellShapeEditor(QWidget *parent = nullptr);
    ~CellShapeEditor();

public slots:
    //Saves the cell shape to a file
    void saveCellShape();
    //Loads a cell shape from a file
    void loadCellShape();

    //Loads a cell mask from a image file
    void loadCellMask();

    //Update cell shape column spacing
    void cellSpacingColChanged(int t_value);
    //Update cell shape column row spacing
    void cellSpacingRowChanged(int t_value);

    //Update cell shape alternate column offset
    void cellAlternateOffsetColChanged(int t_value);
    //Update cell shape alternate row offset
    void cellAlternateOffsetRowChanged(int t_value);

    //Update cell shape alternate column horizontal flipping
    void cellColumnFlipHorizontalChanged(bool t_state);
    //Update cell shape alternate column vertical flipping
    void cellColumnFlipVerticalChanged(bool t_state);
    //Update cell shape alternate row horizontal flipping
    void cellRowFlipHorizontalChanged(bool t_state);
    //Update cell shape alternate row vertical flipping
    void cellRowFlipVerticalChanged(bool t_state);

    //Enables/disables cell shape alternate row spacing
    void enableCellAlternateSpacingRow(bool t_state);
    //Enables/disables cell shape alternate column spacing
    void enableCellAlternateSpacingCol(bool t_state);
    //Updates cell alternate row spacing
    void cellAlternateSpacingRowChanged(int t_value);
    //Updates cell alternate column spacing
    void cellAlternateSpacingColChanged(int t_value);

signals:
    void cellShapeChanged(const CellShape &t_cellShape);
    void cellNameChanged(const QString &t_name);

private:
    //Loads settings from given cell shape
    void loadSettingsFromCellShape(const CellShape &t_cellShape);

    //Updates grid preview
    void updateGridPreview();

    Ui::CellShapeEditor *ui;
};