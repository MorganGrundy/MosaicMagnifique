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

#include "cellshape.h"

#include <opencv2/imgproc.hpp>
#include <QDebug>

#include "utilityfuncs.h"

CellShape::CellShape()
    : m_cellMask{}, m_rowSpacing{0}, m_colSpacing{0},
      m_alternateRowSpacing{0}, m_alternateColSpacing{0},
      m_alternateRowOffset{0}, m_alternateColOffset{0},
      m_colFlipHorizontal{false}, m_colFlipVertical{false},
      m_rowFlipHorizontal{false}, m_rowFlipVertical{false} {}

CellShape::CellShape(const cv::Mat &t_cellMask)
    : m_rowSpacing{t_cellMask.rows}, m_colSpacing{t_cellMask.cols},
      m_alternateRowSpacing{t_cellMask.rows}, m_alternateColSpacing{t_cellMask.cols},
      m_alternateRowOffset{0}, m_alternateColOffset{0},
      m_colFlipHorizontal{false}, m_colFlipVertical{false},
      m_rowFlipHorizontal{false}, m_rowFlipVertical{false}
{
    setCellMask(t_cellMask);
}

CellShape::CellShape(const CellShape &t_cellShape)
    : m_cellMask{t_cellShape.getCellMask(0, 0)},
      m_cellMaskFlippedH{t_cellShape.getCellMask(1, 0)},
      m_cellMaskFlippedV{t_cellShape.getCellMask(0, 1)},
      m_cellMaskFlippedHV{t_cellShape.getCellMask(1, 1)},
      m_rowSpacing{t_cellShape.getRowSpacing()},
      m_colSpacing{t_cellShape.getColSpacing()},
      m_alternateRowSpacing{t_cellShape.getAlternateRowSpacing()},
      m_alternateColSpacing{t_cellShape.getAlternateColSpacing()},
      m_alternateRowOffset{t_cellShape.getAlternateRowOffset()},
      m_alternateColOffset{t_cellShape.getAlternateColOffset()},
      m_colFlipHorizontal{t_cellShape.getColFlipHorizontal()},
      m_colFlipVertical{t_cellShape.getColFlipVertical()},
      m_rowFlipHorizontal{t_cellShape.getRowFlipHorizontal()},
      m_rowFlipVertical{t_cellShape.getRowFlipVertical()} {}

//Writes the CellShape to a QDataStream
QDataStream &operator<<(QDataStream &t_out, const CellShape &t_cellShape)
{
    t_out << t_cellShape.m_cellMask;
    t_out << t_cellShape.m_rowSpacing << t_cellShape.m_colSpacing;
    t_out << t_cellShape.m_alternateRowSpacing << t_cellShape.m_alternateColSpacing;
    t_out << t_cellShape.m_alternateRowOffset << t_cellShape.m_alternateColOffset;
    t_out << t_cellShape.m_colFlipHorizontal << t_cellShape.m_colFlipVertical;
    t_out << t_cellShape.m_rowFlipHorizontal << t_cellShape.m_rowFlipVertical;
    return t_out;
}

//Reads the CellShape from a QDataStream
QDataStream &operator>>(QDataStream &t_in, std::pair<CellShape &, const int> t_cellShape)
{
    t_in >> t_cellShape.first.m_cellMask;
    t_in >> t_cellShape.first.m_rowSpacing >> t_cellShape.first.m_colSpacing;
    if (t_cellShape.second > 5)
    {
        t_in >> t_cellShape.first.m_alternateRowSpacing >> t_cellShape.first.m_alternateColSpacing;
    }
    else
    {
        t_cellShape.first.m_alternateRowSpacing = t_cellShape.first.m_rowSpacing;
        t_cellShape.first.m_alternateColSpacing = t_cellShape.first.m_colSpacing;
    }

    t_in >> t_cellShape.first.m_alternateRowOffset >> t_cellShape.first.m_alternateColOffset;
    if (t_cellShape.second > 4)
    {
        t_in >> t_cellShape.first.m_colFlipHorizontal >> t_cellShape.first.m_colFlipVertical;
        t_in >> t_cellShape.first.m_rowFlipHorizontal >> t_cellShape.first.m_rowFlipVertical;

        //Create flipped cell
        cv::flip(t_cellShape.first.m_cellMask, t_cellShape.first.m_cellMaskFlippedH, 1);
        cv::flip(t_cellShape.first.m_cellMask, t_cellShape.first.m_cellMaskFlippedV, 0);
        cv::flip(t_cellShape.first.m_cellMask, t_cellShape.first.m_cellMaskFlippedHV, -1);
    }

    return t_in;
}

//Sets the cell mask
void CellShape::setCellMask(const cv::Mat &t_cellMask)
{
    //Mask is grayscale
    if (t_cellMask.type() == CV_8UC1)
    {
        cv::Mat result;
        //Threshold to create binary mask
        cv::threshold(t_cellMask, result, 127.0, 255.0, cv::THRESH_BINARY);
        m_cellMask = result;

        UtilityFuncs::imageToSquare(m_cellMask, UtilityFuncs::SquareMethod::PAD);

        //Create flipped cells
        cv::flip(m_cellMask, m_cellMaskFlippedH, 1);
        cv::flip(m_cellMask, m_cellMaskFlippedV, 0);
        cv::flip(m_cellMask, m_cellMaskFlippedHV, -1);
    }
    else
        qDebug() << "CellShape::setCellMask(const cv::Mat &) unsupported mask type";
}

//Returns the cell mask
cv::Mat CellShape::getCellMask(const bool t_flippedHorizontal, const bool t_flippedVertical) const
{
    if (t_flippedHorizontal)
    {
        if (t_flippedVertical)
            return m_cellMaskFlippedHV;
        else
            return m_cellMaskFlippedH;
    }
    else if (t_flippedVertical)
        return m_cellMaskFlippedV;
    else
        return m_cellMask;
}

//Sets the row spacing for tiling cell
void CellShape::setRowSpacing(const int t_rowSpacing)
{
    m_rowSpacing = t_rowSpacing;
}

//Returns the row spacing for tiling cell
int CellShape::getRowSpacing() const
{
    return m_rowSpacing;
}

//Sets the column spacing for tiling cell
void CellShape::setColSpacing(const int t_colSpacing)
{
    m_colSpacing = t_colSpacing;
}

//Returns the column spacing for tiling cell
int CellShape::getColSpacing() const
{
    return m_colSpacing;
}

//Sets the alternate row spacing for tiling cell
void CellShape::setAlternateRowSpacing(const int t_alternateRowSpacing)
{
    m_alternateRowSpacing = t_alternateRowSpacing;
}

//Returns the alternate row spacing for tiling cell
int CellShape::getAlternateRowSpacing() const
{
    return m_alternateRowSpacing;
}

//Sets the alternate column spacing for tiling cell
void CellShape::setAlternateColSpacing(const int t_alternateColSpacing)
{
    m_alternateColSpacing = t_alternateColSpacing;
}

//Returns the alternate column spacing for tiling cell
int CellShape::getAlternateColSpacing() const
{
    return m_alternateColSpacing;
}

//Sets the alternate row offset
void CellShape::setAlternateRowOffset(const int t_alternateRowOffset)
{
    m_alternateRowOffset = t_alternateRowOffset;
}

//Returns the alternate row offset
int CellShape::getAlternateRowOffset() const
{
    return m_alternateRowOffset;
}

//Sets the alternate column offset
void CellShape::setAlternateColOffset(const int t_alternateColOffset)
{
    m_alternateColOffset = t_alternateColOffset;
}

//Returns the alternate column offset
int CellShape::getAlternateColOffset() const
{
    return m_alternateColOffset;
}

//Sets if alternate columns are flipped horizontally
void CellShape::setColFlipHorizontal(const bool t_colFlipHorizontal)
{
    m_colFlipHorizontal = t_colFlipHorizontal;
}

//Returns if alternate columns are flipped horizontally
bool CellShape::getColFlipHorizontal() const
{
    return m_colFlipHorizontal;
}

//Sets if alternate columns are flipped vertically
void CellShape::setColFlipVertical(const bool t_colFlipVertical)
{
    m_colFlipVertical = t_colFlipVertical;
}

//Sets if alternate columns are flipped vertically
bool CellShape::getColFlipVertical() const
{
    return m_colFlipVertical;
}

//Sets if alternate rows are flipped horizontally
void CellShape::setRowFlipHorizontal(const bool t_rowFlipHorizontal)
{
    m_rowFlipHorizontal = t_rowFlipHorizontal;
}

//Returns if alternate rows are flipped horizontally
bool CellShape::getRowFlipHorizontal() const
{
    return m_rowFlipHorizontal;
}

//Sets if alternate rows are flipped vertically
void CellShape::setRowFlipVertical(const bool t_rowFlipVertical)
{
    m_rowFlipVertical = t_rowFlipVertical;
}

//Returns if alternate rows are flipped vertically
bool CellShape::getRowFlipVertical() const
{
    return m_rowFlipVertical;
}

//Returns the cell shape resized to the given size
CellShape CellShape::resized(const int t_cols, const int t_rows) const
{
    //If cell mask is empty then just return copy of cell shape
    if (empty())
    {
        return CellShape(*this);
    }

    const cv::Mat resizedMask = UtilityFuncs::resizeImage(m_cellMask, t_rows, t_cols,
                                                          UtilityFuncs::ResizeType::EXACT);

    CellShape result(resizedMask);
    //Resize spacing and offset
    double vRatio = static_cast<double>(resizedMask.rows) / m_cellMask.rows;
    double hRatio = static_cast<double>(resizedMask.cols) / m_cellMask.cols;
    //Spacing must never be < 1
    result.setRowSpacing(std::max(static_cast<int>(std::floor(m_rowSpacing * vRatio)), 1));
    result.setColSpacing(std::max(static_cast<int>(std::floor(m_colSpacing * hRatio)), 1));
    result.setAlternateRowSpacing(std::max(
        static_cast<int>(std::floor(m_alternateRowSpacing * vRatio)), 1));
    result.setAlternateColSpacing(std::max(
        static_cast<int>(std::floor(m_alternateColSpacing * hRatio)), 1));

    result.setAlternateRowOffset(std::floor(m_alternateRowOffset * vRatio));
    result.setAlternateColOffset(std::floor(m_alternateColOffset * hRatio));
    result.setColFlipHorizontal(m_colFlipHorizontal);
    result.setColFlipVertical(m_colFlipVertical);
    result.setRowFlipHorizontal(m_rowFlipHorizontal);
    result.setRowFlipVertical(m_rowFlipVertical);

    return result;
}

//Returns if the cell mask is empty
bool CellShape::empty() const
{
    return m_cellMask.empty();
}
