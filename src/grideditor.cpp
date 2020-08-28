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

    //Pass new size step to grid edit viewer
    connect(ui->spinSizeStep, qOverload<int>(&QSpinBox::valueChanged),
            [&](const int t_newValue){
                ui->gridEditViewer->setSizeStep(static_cast<size_t>(t_newValue));
            });

    //Tool buttons
    connect(ui->buttonGroupTools, qOverload<QAbstractButton *>(&QButtonGroup::buttonClicked),
            [&](QAbstractButton *button)
            {
                //Update grid edit viewer active tool
                if (button->text() == "Single")
                {
                    ui->gridEditViewer->setTool(GridEditViewer::Tool::Single);
                }
                else if (button->text() == "Selection")
                {
                    ui->gridEditViewer->setTool(GridEditViewer::Tool::Selection);
                }
            });

    //Activate single tool by default
    ui->toolSingle->click();
}

GridEditor::GridEditor(const cv::Mat &t_background, const CellGroup &t_cellGroup,
                       QWidget *t_parent) : GridEditor{t_parent}
{
    ui->gridEditViewer->setBackground(t_background);
    ui->gridEditViewer->setCellGroup(t_cellGroup);

    ui->spinSizeStep->setMaximum(static_cast<int>(t_cellGroup.getSizeSteps()));
}

GridEditor::~GridEditor()
{
    delete ui;
}

//Updates grid
void GridEditor::showEvent(QShowEvent * /*event*/)
{
    ui->gridEditViewer->updateGrid();
}

//Emit grid state when grid editor is closed
void GridEditor::closeEvent(QCloseEvent * /*event*/)
{
    emit gridStateChanged(ui->gridEditViewer->getGridState());
}
