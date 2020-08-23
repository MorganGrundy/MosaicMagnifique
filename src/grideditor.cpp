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

#include "grideditor.h"
#include "ui_grideditor.h"

GridEditor::GridEditor(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::GridEditor)
{
    ui->setupUi(this);
}

GridEditor::~GridEditor()
{
    delete ui;
}

GridEditViewer *GridEditor::getGridEditViewer()
{
    return ui->gridEditViewer;
}

//Emit grid state when grid editor is closed
void GridEditor::closeEvent(QCloseEvent */*event*/)
{
    emit gridStateChanged(ui->gridEditViewer->getGridState());
}
