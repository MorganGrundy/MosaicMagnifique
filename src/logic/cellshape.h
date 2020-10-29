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

#ifndef CELLSHAPE_H
#define CELLSHAPE_H

#include <opencv2/core/mat.hpp>
#include <QDataStream>

class CellShape
{
public:
    //Current version number
    static const quint32 MCS_VERSION = 7;
    static const quint32 MCS_MAGIC = 0x87AECFB1;

    //Default cell size
    static const size_t DEFAULT_CELL_SIZE = 128;

    //Constructors
    CellShape();
    CellShape(const cv::Mat &t_cellMask);
    CellShape(const CellShape &t_cellShape);
    //Constructor for default cell shape
    CellShape(const size_t t_size);

    //Return if CellShape is equal to other
    bool operator==(const CellShape &t_other) const;

    void setName(const QString &t_name);
    QString getName() const;

    void setCellMask(const cv::Mat &t_cellMask);
    const cv::Mat &getCellMask(const bool t_flippedHorizontal, const bool t_flippedVertical) const;

    void setRowSpacing(const int t_rowSpacing);
    int getRowSpacing() const;

    void setColSpacing(const int t_colSpacing);
    int getColSpacing() const;

    void setAlternateRowSpacing(const int t_alternateRowSpacing);
    int getAlternateRowSpacing() const;

    void setAlternateColSpacing(const int t_alternateColSpacing);
    int getAlternateColSpacing() const;

    void setAlternateRowOffset(const int t_alternateRowOffset);
    int getAlternateRowOffset() const;

    void setAlternateColOffset(const int t_alternateColOffset);
    int getAlternateColOffset() const;

    void setAlternateColFlipHorizontal(const bool t_colFlipHorizontal);
    bool getAlternateColFlipHorizontal() const;

    void setAlternateColFlipVertical(const bool t_colFlipVertical);
    bool getAlternateColFlipVertical() const;

    void setAlternateRowFlipHorizontal(const bool t_rowFlipHorizontal);
    bool getAlternateRowFlipHorizontal() const;

    void setAlternateRowFlipVertical(const bool t_rowFlipVertical);
    bool getAlternateRowFlipVertical() const;

    CellShape resized(const int t_cols, const int t_rows) const;

    bool empty() const;

    //Saves the cell shape to the given file
    void saveToFile(const QString t_filename) const;

    //Loads cell shape from given file
    void loadFromFile(const QString t_filename);

private:
    cv::Mat m_cellMask;
    cv::Mat m_cellMaskFlippedH, m_cellMaskFlippedV, m_cellMaskFlippedHV;

    int m_rowSpacing, m_colSpacing;
    int m_alternateRowSpacing, m_alternateColSpacing;

    //Entire row/col offset by specified value every other row/col
    int m_alternateRowOffset, m_alternateColOffset;

    //Controls if and how cells should flip on alternate rows/columns
    bool m_alternateColFlipHorizontal, m_alternateColFlipVertical,
        m_alternateRowFlipHorizontal, m_alternateRowFlipVertical;

    //Name of cell shape
    QString m_name;
};

#endif // CELLSHAPE_H
