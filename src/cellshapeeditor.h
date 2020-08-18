#ifndef CELLSHAPEEDITOR_H
#define CELLSHAPEEDITOR_H

#include <QWidget>

#include "cellshape.h"

namespace Ui {
class CellShapeEditor;
}

class CellShapeEditor : public QWidget
{
    Q_OBJECT

public:
    explicit CellShapeEditor(QWidget *parent = nullptr);
    ~CellShapeEditor();

    //Returns if the cell shape has been edited
    bool isCellShapeChanged() const;

    //Returns current cell shape
    const CellShape &getCellShape();

    //Returns cell shape name
    const QString getCellShapeName();

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

private:
    Ui::CellShapeEditor *ui;

    //Stores if the cell shape has been changed
    bool cellShapeChanged;

    //Loads settings from given cell shape
    void loadSettingsFromCellShape(const CellShape &t_cellShape);
};

#endif // CELLSHAPEEDITOR_H
