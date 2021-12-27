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

#include "CellShape.h"

#include <opencv2/imgproc.hpp>
#include <QDebug>
#include <QFile>

#include "ImageUtility.h"

CellShape::CellShape()
    : m_cellMask{}, m_rowSpacing{0}, m_colSpacing{0},
    m_alternateRowSpacing{0}, m_alternateColSpacing{0},
    m_alternateRowOffset{0}, m_alternateColOffset{0},
    m_alternateColFlipHorizontal{false}, m_alternateColFlipVertical{false},
    m_alternateRowFlipHorizontal{false}, m_alternateRowFlipVertical{false}
{}

CellShape::CellShape(const cv::Mat &t_cellMask)
    : m_rowSpacing{t_cellMask.rows}, m_colSpacing{t_cellMask.cols},
    m_alternateRowSpacing{t_cellMask.rows}, m_alternateColSpacing{t_cellMask.cols},
    m_alternateRowOffset{0}, m_alternateColOffset{0},
    m_alternateColFlipHorizontal{false}, m_alternateColFlipVertical{false},
    m_alternateRowFlipHorizontal{false}, m_alternateRowFlipVertical{false}
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
    m_alternateColFlipHorizontal{t_cellShape.getAlternateColFlipHorizontal()},
    m_alternateColFlipVertical{t_cellShape.getAlternateColFlipVertical()},
    m_alternateRowFlipHorizontal{t_cellShape.getAlternateRowFlipHorizontal()},
    m_alternateRowFlipVertical{t_cellShape.getAlternateRowFlipVertical()}
{}

//Constructor for default cell shape
CellShape::CellShape(const size_t t_size)
    : CellShape{cv::Mat(static_cast<int>(t_size), static_cast<int>(t_size),
                        CV_8UC1, cv::Scalar(255))}
{}

//Return if CellShape is equal to other
bool CellShape::operator==(const CellShape &t_other) const
{
    //Compare mask image
    if (m_cellMask.size != t_other.m_cellMask.size)
        return false;
    if (cv::sum(m_cellMask != t_other.m_cellMask)[0] != 0)
        return false;

    //Compare spacing
    if ((m_rowSpacing != t_other.m_rowSpacing) || (m_colSpacing != t_other.m_colSpacing))
        return false;

    //Compare alternate spacing
    if ((m_alternateRowSpacing != t_other.m_alternateRowSpacing)
        || (m_alternateColSpacing != t_other.m_alternateColSpacing))
        return false;

    //Compare alternate offset
    if ((m_alternateRowOffset != t_other.m_alternateRowOffset)
        || (m_alternateColOffset != t_other.m_alternateColOffset))
        return false;

    //Compare alternate flip states
    if ((m_alternateColFlipHorizontal != t_other.m_alternateColFlipHorizontal)
        || (m_alternateColFlipVertical != t_other.m_alternateColFlipVertical)
        || (m_alternateRowFlipHorizontal != t_other.m_alternateRowFlipHorizontal)
        || (m_alternateRowFlipVertical != t_other.m_alternateRowFlipVertical))
        return false;

    return true;
}

//Sets the name of the cell shape
void CellShape::setName(const QString &t_name)
{
    m_name = t_name;
}

//Returns the name of the cell shape
QString CellShape::getName() const
{
    return m_name;
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

        ImageUtility::imageToSquare(m_cellMask, ImageUtility::SquareMethod::PAD);

        //Create flipped cells
        cv::flip(m_cellMask, m_cellMaskFlippedH, 1);
        cv::flip(m_cellMask, m_cellMaskFlippedV, 0);
        cv::flip(m_cellMask, m_cellMaskFlippedHV, -1);
    }
    else
        throw std::invalid_argument(Q_FUNC_INFO " Unsupported mask type");
}

//Returns the cell mask
const cv::Mat &CellShape::getCellMask(const bool t_flippedHorizontal,
                                      const bool t_flippedVertical) const
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

//Returns cell shape size
int CellShape::getSize() const
{
    return m_cellMask.rows;
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
void CellShape::setAlternateColFlipHorizontal(const bool t_colFlipHorizontal)
{
    m_alternateColFlipHorizontal = t_colFlipHorizontal;
}

//Returns if alternate columns are flipped horizontally
bool CellShape::getAlternateColFlipHorizontal() const
{
    return m_alternateColFlipHorizontal;
}

//Sets if alternate columns are flipped vertically
void CellShape::setAlternateColFlipVertical(const bool t_colFlipVertical)
{
    m_alternateColFlipVertical = t_colFlipVertical;
}

//Sets if alternate columns are flipped vertically
bool CellShape::getAlternateColFlipVertical() const
{
    return m_alternateColFlipVertical;
}

//Sets if alternate rows are flipped horizontally
void CellShape::setAlternateRowFlipHorizontal(const bool t_rowFlipHorizontal)
{
    m_alternateRowFlipHorizontal = t_rowFlipHorizontal;
}

//Returns if alternate rows are flipped horizontally
bool CellShape::getAlternateRowFlipHorizontal() const
{
    return m_alternateRowFlipHorizontal;
}

//Sets if alternate rows are flipped vertically
void CellShape::setAlternateRowFlipVertical(const bool t_rowFlipVertical)
{
    m_alternateRowFlipVertical = t_rowFlipVertical;
}

//Returns if alternate rows are flipped vertically
bool CellShape::getAlternateRowFlipVertical() const
{
    return m_alternateRowFlipVertical;
}

//Returns the cell shape resized to the given size
CellShape CellShape::resized(const int t_size) const
{
    //If cell mask is empty or new size is equal to current size
    //then just return copy of cell shape
    if (empty() || t_size == getSize())
    {
        return CellShape(*this);
    }

    const cv::Mat resizedMask = ImageUtility::resizeImage(m_cellMask, t_size, t_size,
                                                          ImageUtility::ResizeType::EXACT);

    CellShape result(resizedMask);
    //Resize spacing and offset
    double ratio = static_cast<double>(resizedMask.rows) / m_cellMask.rows;
    //Spacing must never be < 1
    result.setRowSpacing(std::max(static_cast<int>(std::floor(m_rowSpacing * ratio)), 1));
    result.setColSpacing(std::max(static_cast<int>(std::floor(m_colSpacing * ratio)), 1));
    result.setAlternateRowSpacing(std::max(
        static_cast<int>(std::floor(m_alternateRowSpacing * ratio)), 1));
    result.setAlternateColSpacing(std::max(
        static_cast<int>(std::floor(m_alternateColSpacing * ratio)), 1));

    result.setAlternateRowOffset(std::floor(m_alternateRowOffset * ratio));
    result.setAlternateColOffset(std::floor(m_alternateColOffset * ratio));
    result.setAlternateColFlipHorizontal(m_alternateColFlipHorizontal);
    result.setAlternateColFlipVertical(m_alternateColFlipVertical);
    result.setAlternateRowFlipHorizontal(m_alternateRowFlipHorizontal);
    result.setAlternateRowFlipVertical(m_alternateRowFlipVertical);

    return result;
}

//Returns if the cell mask is empty
bool CellShape::empty() const
{
    return m_cellMask.empty();
}

//Saves the cell shape to the given file
void CellShape::saveToFile(const QString t_filename) const
{
    if (t_filename.isNull())
        throw std::invalid_argument("No filename");
    else
    {
        QFile file(t_filename);
        file.open(QIODevice::WriteOnly);
        if (!file.isWritable())
            throw std::invalid_argument("File is not writable: " + t_filename.toStdString());
        else
        {
            QDataStream out(&file);
            //Write header with "magic number" and version
            out << MCS_MAGIC;
            out << MCS_VERSION;

            out.setVersion(QDataStream::Qt_5_0);

            //Write cell shape
            out << m_name;
            out << m_cellMask;

            //Write cell spacing
            out << static_cast<qint32>(m_rowSpacing) << static_cast<qint32>(m_colSpacing);
            out << static_cast<qint32>(m_alternateRowSpacing)
                << static_cast<qint32>(m_alternateColSpacing);
            out << static_cast<qint32>(m_alternateRowOffset)
                << static_cast<qint32>(m_alternateColOffset);

            //Write flip states
            out << m_alternateColFlipHorizontal << m_alternateColFlipVertical;
            out << m_alternateRowFlipHorizontal << m_alternateRowFlipVertical;

            file.close();
        }
    }
}

//Loads cell shape from given file
void CellShape::loadFromFile(const QString t_filename)
{
    if (t_filename.isNull())
        throw std::invalid_argument("No filename");
    {
        //Check for valid file
        QFile file(t_filename);
        file.open(QIODevice::ReadOnly);
        if (!file.isReadable())
            throw std::invalid_argument("File is not readable: " + t_filename.toStdString());
        else
        {
            QDataStream in(&file);

            //Read and check magic number
            quint32 magic;
            in >> magic;
            if (magic != MCS_MAGIC)
                throw std::invalid_argument("File is not a valid .mcs: "
                                            + t_filename.toStdString());

            //Read the version
            quint32 version;
            in >> version;
            if (version <= MCS_VERSION && version >= 7)
                in.setVersion(QDataStream::Qt_5_0);
            else
            {
                if (version < MCS_VERSION)
                    throw std::invalid_argument(".mcs uses an outdated file version: "
                                                + t_filename.toStdString());
                else
                    throw std::invalid_argument(".mcs uses a newer file version: "
                                                + t_filename.toStdString());
            }

            //Read cell shape
            in >> m_name;
            in >> m_cellMask;
            //Create flipped masks
            cv::flip(m_cellMask, m_cellMaskFlippedH, 1);
            cv::flip(m_cellMask, m_cellMaskFlippedV, 0);
            cv::flip(m_cellMask, m_cellMaskFlippedHV, -1);

            //Read cell spacing
            qint32 tmpRow, tmpCol;
            in >> tmpRow >> tmpCol;
            m_rowSpacing = static_cast<int>(tmpRow);
            m_colSpacing = static_cast<int>(tmpCol);

            in >> tmpRow >> tmpCol;
            m_alternateRowSpacing = static_cast<int>(tmpRow);
            m_alternateColSpacing = static_cast<int>(tmpCol);

            in >> tmpRow >> tmpCol;
            m_alternateRowOffset = static_cast<int>(tmpRow);
            m_alternateColOffset = static_cast<int>(tmpCol);

            //Read flip states
            in >> m_alternateColFlipHorizontal >> m_alternateColFlipVertical;
            in >> m_alternateRowFlipHorizontal >> m_alternateRowFlipVertical;

            file.close();
        }
    }
}
