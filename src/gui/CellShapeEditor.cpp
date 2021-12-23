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

#include "CellShapeEditor.h"
#include "ui_CellShapeEditor.h"

#include <QMessageBox>
#include <QFileDialog>
#include <QDebug>
#include <QLineEdit>
#include <opencv2/core/mat.hpp>
#include <opencv2/imgcodecs.hpp>

CellShapeEditor::CellShapeEditor(QWidget *parent) :
    QWidget(parent),
    ui(new Ui::CellShapeEditor)
{
    ui->setupUi(this);

    //Connects ui signals to slots
    //Save and load cell shape
    connect(ui->buttonSaveCell, &QPushButton::released, this, &CellShapeEditor::saveCellShape);
    connect(ui->buttonLoadCell, &QPushButton::released, this, &CellShapeEditor::loadCellShape);

    //Cell name changed
    connect(ui->lineCellName, &QLineEdit::editingFinished, this, [=]()
            {
                emit cellNameChanged(ui->lineCellName->text());
                ui->cellShapeViewer->getCellGroup().getCell(0).setName(ui->lineCellName->text());
            });

    //Cell mask button
    connect(ui->buttonCellMask, &QPushButton::released, this, &CellShapeEditor::loadCellMask);

    //Cell spacing changed
    connect(ui->spinCellSpacingCol, qOverload<int>(&QSpinBox::valueChanged),
            this, &CellShapeEditor::cellSpacingColChanged);
    connect(ui->spinCellSpacingRow, qOverload<int>(&QSpinBox::valueChanged),
            this, &CellShapeEditor::cellSpacingRowChanged);

    //Cell alternate offset changed
    connect(ui->spinCellAlternateOffsetCol, qOverload<int>(&QSpinBox::valueChanged),
            this, &CellShapeEditor::cellAlternateOffsetColChanged);
    connect(ui->spinCellAlternateOffsetRow, qOverload<int>(&QSpinBox::valueChanged),
            this, &CellShapeEditor::cellAlternateOffsetRowChanged);

    //Cell flip state changed
    connect(ui->checkCellColFlipH, &QCheckBox::clicked,
            this, &CellShapeEditor::cellColumnFlipHorizontalChanged);
    connect(ui->checkCellColFlipV, &QCheckBox::clicked,
            this, &CellShapeEditor::cellColumnFlipVerticalChanged);
    connect(ui->checkCellRowFlipH, &QCheckBox::clicked,
            this, &CellShapeEditor::cellRowFlipHorizontalChanged);
    connect(ui->checkCellRowFlipV, &QCheckBox::clicked,
            this, &CellShapeEditor::cellRowFlipVerticalChanged);

    //Cell alternate spacing toggles
    connect(ui->checkCellAlternateSpacingRow, &QCheckBox::toggled,
            this, &CellShapeEditor::enableCellAlternateSpacingRow);
    connect(ui->checkCellAlternateSpacingCol, &QCheckBox::toggled,
            this, &CellShapeEditor::enableCellAlternateSpacingCol);
    //Cell alternate spacing changed
    connect(ui->spinCellAlternateSpacingRow, qOverload<int>(&QSpinBox::valueChanged),
            this, &CellShapeEditor::cellAlternateSpacingRowChanged);
    connect(ui->spinCellAlternateSpacingCol, qOverload<int>(&QSpinBox::valueChanged),
            this, &CellShapeEditor::cellAlternateSpacingColChanged);

    //Sets cell shape editor to default cell shape
    CellShape defaultCellShape(CellShape::DEFAULT_CELL_SIZE);
    ui->cellShapeViewer->getCellGroup().setCellShape(defaultCellShape);
    ui->spinCellSpacingCol->setValue(CellShape::DEFAULT_CELL_SIZE);
    ui->spinCellSpacingRow->setValue(CellShape::DEFAULT_CELL_SIZE);
}

CellShapeEditor::~CellShapeEditor()
{
    delete ui;
}

//Saves the cell shape to a file
void CellShapeEditor::saveCellShape()
{
    //Checks for valid cell shape
    const CellShape &cellShape = ui->cellShapeViewer->getCellGroup().getCell(0);
    if (cellShape.getCellMask(0, 0).empty())
    {
        QMessageBox::information(this, tr("Failed to save custom cell shape"),
                                 tr("No cell mask was provided"));
        return;
    }

    //Get file directory to save cell shape at from user
    QString defaultDir = QDir::currentPath() + "/" +ui->lineCellName->text() + ".mcs";
    QString filename = QFileDialog::getSaveFileName(this, tr("Save cell shape"), defaultDir,
                                                    tr("Mosaic Cell Shape") + " (*.mcs)");

    try
    {
        ui->cellShapeViewer->getCellGroup().getCell(0).saveToFile(filename);
    }
    catch (const std::invalid_argument &e)
    {
        QMessageBox msgBox;
        msgBox.setText(tr(e.what()));
        msgBox.exec();

        return;
    }
}

//Loads a cell shape from a file
void CellShapeEditor::loadCellShape()
{
    //Get path to mcs file from user
    QString filename = QFileDialog::getOpenFileName(this, tr("Select custom cell shape to load"),
                                                    "", tr("Mosaic Cell Shape") + " (*.mcs)");

    try
    {
        CellShape tmpCellShape;
        tmpCellShape.loadFromFile(filename);

        //Give cell shape to grid preview
        ui->cellShapeViewer->getCellGroup().setCellShape(tmpCellShape);
        updateGridPreview();

        //Update cell settings in ui
        ui->lineCellName->setText(tmpCellShape.getName());
        loadSettingsFromCellShape(tmpCellShape);
        emit cellShapeChanged(tmpCellShape);
        emit cellNameChanged(ui->lineCellName->text());
    }
    catch (const std::invalid_argument &e)
    {
        QMessageBox msgBox;
        msgBox.setText(tr(e.what()));
        msgBox.exec();

        return;
    }
}

//Loads a cell mask from a image file
void CellShapeEditor::loadCellMask()
{
    //Get path to image file from user
    QString filename = QFileDialog::getOpenFileName(this, tr("Select cell mask"), "",
                                                    "Image Files (*.bmp *.dib *.jpeg *.jpg "
                                                    "*.jpe *.jp2 *.png *.pbm *.pgm *.ppm "
                                                    "*.pxm *.pnm *.sr *.ras *.tiff *.tif "
                                                    "*.hdr *.pic)");
    cv::Mat tmp = cv::imread(filename.toStdString(), cv::IMREAD_GRAYSCALE);
    if (!tmp.empty())
    {
        //Create cell shape from cell mask and update settings
        ui->lineCellMaskPath->setText(filename);
        CellShape cellShape(tmp);
        loadSettingsFromCellShape(cellShape);

        //Give cell shape to grid preview
        ui->cellShapeViewer->getCellGroup().setCellShape(cellShape);
        updateGridPreview();
        emit cellShapeChanged(ui->cellShapeViewer->getCellGroup().getCell(0));
    }
}

//Update cell shape column spacing
void CellShapeEditor::cellSpacingColChanged(int t_value)
{
    //Update alternate spacing
    if (!ui->spinCellAlternateSpacingCol->isEnabled())
        ui->spinCellAlternateSpacingCol->setValue(t_value);

    //Create new cell shape
    CellShape newCellShape = ui->cellShapeViewer->getCellGroup().getCell(0);
    newCellShape.setColSpacing(t_value);

    //Give cell shape to grid preview
    ui->cellShapeViewer->getCellGroup().setCellShape(newCellShape);
    updateGridPreview();
    emit cellShapeChanged(ui->cellShapeViewer->getCellGroup().getCell(0));
}

//Update cell shape column row spacing
void CellShapeEditor::cellSpacingRowChanged(int t_value)
{
    //Update alternate spacing
    if (!ui->spinCellAlternateSpacingRow->isEnabled())
        ui->spinCellAlternateSpacingRow->setValue(t_value);

    //Create new cell shape
    CellShape newCellShape = ui->cellShapeViewer->getCellGroup().getCell(0);
    newCellShape.setRowSpacing(t_value);

    //Give cell shape to grid preview
    ui->cellShapeViewer->getCellGroup().setCellShape(newCellShape);
    updateGridPreview();
    emit cellShapeChanged(ui->cellShapeViewer->getCellGroup().getCell(0));
}

//Update cell shape alternate column offset
void CellShapeEditor::cellAlternateOffsetColChanged(int t_value)
{
    //Create new cell shape
    CellShape newCellShape = ui->cellShapeViewer->getCellGroup().getCell(0);
    newCellShape.setAlternateColOffset(t_value);

    //Give cell shape to grid preview
    ui->cellShapeViewer->getCellGroup().setCellShape(newCellShape);
    updateGridPreview();
    emit cellShapeChanged(ui->cellShapeViewer->getCellGroup().getCell(0));
}

//Update cell shape alternate row offset
void CellShapeEditor::cellAlternateOffsetRowChanged(int t_value)
{
    //Create new cell shape
    CellShape newCellShape = ui->cellShapeViewer->getCellGroup().getCell(0);
    newCellShape.setAlternateRowOffset(t_value);

    //Give cell shape to grid preview
    ui->cellShapeViewer->getCellGroup().setCellShape(newCellShape);
    updateGridPreview();
    emit cellShapeChanged(ui->cellShapeViewer->getCellGroup().getCell(0));
}

//Update cell shape alternate column horizontal flipping
void CellShapeEditor::cellColumnFlipHorizontalChanged(bool t_state)
{
    //Create new cell shape
    CellShape newCellShape = ui->cellShapeViewer->getCellGroup().getCell(0);
    newCellShape.setAlternateColFlipHorizontal(t_state);

    //Give cell shape to grid preview
    ui->cellShapeViewer->getCellGroup().setCellShape(newCellShape);
    updateGridPreview();
    emit cellShapeChanged(ui->cellShapeViewer->getCellGroup().getCell(0));
}

//Update cell shape alternate column vertical flipping
void CellShapeEditor::cellColumnFlipVerticalChanged(bool t_state)
{
    //Create new cell shape
    CellShape newCellShape = ui->cellShapeViewer->getCellGroup().getCell(0);
    newCellShape.setAlternateColFlipVertical(t_state);

    //Give cell shape to grid preview
    ui->cellShapeViewer->getCellGroup().setCellShape(newCellShape);
    updateGridPreview();
    emit cellShapeChanged(ui->cellShapeViewer->getCellGroup().getCell(0));
}

//Update cell shape alternate row horizontal flipping
void CellShapeEditor::cellRowFlipHorizontalChanged(bool t_state)
{
    //Create new cell shape
    CellShape newCellShape = ui->cellShapeViewer->getCellGroup().getCell(0);
    newCellShape.setAlternateRowFlipHorizontal(t_state);

    //Give cell shape to grid preview
    ui->cellShapeViewer->getCellGroup().setCellShape(newCellShape);
    updateGridPreview();
    emit cellShapeChanged(ui->cellShapeViewer->getCellGroup().getCell(0));
}

//Update cell shape alternate row vertical flipping
void CellShapeEditor::cellRowFlipVerticalChanged(bool t_state)
{
    //Create new cell shape
    CellShape newCellShape = ui->cellShapeViewer->getCellGroup().getCell(0);
    newCellShape.setAlternateRowFlipVertical(t_state);

    //Give cell shape to grid preview
    ui->cellShapeViewer->getCellGroup().setCellShape(newCellShape);
    updateGridPreview();
    emit cellShapeChanged(ui->cellShapeViewer->getCellGroup().getCell(0));
}

//Enables/disables cell shape alternate row spacing
void CellShapeEditor::enableCellAlternateSpacingRow(bool t_state)
{
    //If disabling, reset value to row spacing
    if (!t_state)
        ui->spinCellAlternateSpacingRow->setValue(ui->spinCellSpacingRow->value());

    ui->spinCellAlternateSpacingRow->setEnabled(t_state);
}

//Enables/disables cell shape alternate column spacing
void CellShapeEditor::enableCellAlternateSpacingCol(bool t_state)
{
    //If disabling, reset value to column spacing
    if (!t_state)
        ui->spinCellAlternateSpacingCol->setValue(ui->spinCellSpacingCol->value());

    ui->spinCellAlternateSpacingCol->setEnabled(t_state);
}

//Updates cell alternate row spacing
void CellShapeEditor::cellAlternateSpacingRowChanged(int t_value)
{
    if (ui->checkCellAlternateSpacingRow->isEnabled())
    {
        //Create new cell shape
        CellShape newCellShape = ui->cellShapeViewer->getCellGroup().getCell(0);
        newCellShape.setAlternateRowSpacing(t_value);

        //Give cell shape to grid preview
        ui->cellShapeViewer->getCellGroup().setCellShape(newCellShape);
        updateGridPreview();
        emit cellShapeChanged(ui->cellShapeViewer->getCellGroup().getCell(0));
    }
}

//Updates cell alternate column spacing
void CellShapeEditor::cellAlternateSpacingColChanged(int t_value)
{
    if (ui->checkCellAlternateSpacingCol->isEnabled())
    {
        //Create new cell shape
        CellShape newCellShape = ui->cellShapeViewer->getCellGroup().getCell(0);
        newCellShape.setAlternateColSpacing(t_value);

        //Give cell shape to grid preview
        ui->cellShapeViewer->getCellGroup().setCellShape(newCellShape);
        updateGridPreview();
        emit cellShapeChanged(ui->cellShapeViewer->getCellGroup().getCell(0));
    }
}

//Loads settings from given cell shape
void CellShapeEditor::loadSettingsFromCellShape(const CellShape &t_cellShape)
{
    //Blocks signals to prevent grid update until all values loaded
    //Update cell spacing
    ui->spinCellSpacingCol->blockSignals(true);
    ui->spinCellSpacingRow->blockSignals(true);
    ui->spinCellSpacingCol->setValue(t_cellShape.getColSpacing());
    ui->spinCellSpacingRow->setValue(t_cellShape.getRowSpacing());
    ui->spinCellSpacingCol->blockSignals(false);
    ui->spinCellSpacingRow->blockSignals(false);

    //Update cell alternate spacing
    ui->spinCellAlternateSpacingCol->blockSignals(true);
    ui->spinCellAlternateSpacingRow->blockSignals(true);
    ui->checkCellAlternateSpacingCol->setChecked(
        t_cellShape.getAlternateColSpacing() != t_cellShape.getColSpacing());
    ui->checkCellAlternateSpacingRow->setChecked(
        t_cellShape.getAlternateRowSpacing() != t_cellShape.getRowSpacing());
    ui->spinCellAlternateSpacingCol->setValue(t_cellShape.getAlternateColSpacing());
    ui->spinCellAlternateSpacingRow->setValue(t_cellShape.getAlternateRowSpacing());
    ui->spinCellAlternateSpacingCol->blockSignals(false);
    ui->spinCellAlternateSpacingRow->blockSignals(false);

    //Update cell alternate offset
    ui->spinCellAlternateOffsetCol->blockSignals(true);
    ui->spinCellAlternateOffsetRow->blockSignals(true);
    ui->spinCellAlternateOffsetCol->setValue(t_cellShape.getAlternateColOffset());
    ui->spinCellAlternateOffsetRow->setValue(t_cellShape.getAlternateRowOffset());
    ui->spinCellAlternateOffsetCol->blockSignals(false);
    ui->spinCellAlternateOffsetRow->blockSignals(false);

    //Update cell flip states
    //No need to block signals as setChecked does not trigger clicked signal
    ui->checkCellColFlipH->setChecked(t_cellShape.getAlternateColFlipHorizontal());
    ui->checkCellColFlipV->setChecked(t_cellShape.getAlternateColFlipVertical());
    ui->checkCellRowFlipH->setChecked(t_cellShape.getAlternateRowFlipHorizontal());
    ui->checkCellRowFlipV->setChecked(t_cellShape.getAlternateRowFlipVertical());
}

//Updates grid preview
void CellShapeEditor::updateGridPreview()
{
    //Save focus widget
    QWidget *focusWidget = QApplication::focusWidget();
    //Disable window interactions
    setEnabled(false);

    //Update grid preview
    ui->cellShapeViewer->updateGrid();

    //Enable window interactions
    setEnabled(true);
    //Return focus to saved widget
    if (focusWidget)
        focusWidget->setFocus();
}
