#include "cellshape.h"

#include <opencv2/imgproc.hpp>
#include <QDebug>

#include "utilityfuncs.h"

CellShape::CellShape()
    : m_cellMask{}, m_rowSpacing{0}, m_colSpacing{0},
      m_alternateRowOffset{0}, m_alternateColOffset{0} {}

CellShape::CellShape(const cv::Mat &t_cellMask)
    : m_rowSpacing{t_cellMask.rows}, m_colSpacing{t_cellMask.cols},
      m_alternateRowOffset{0}, m_alternateColOffset{0}
{
    setCellMask(t_cellMask);
}

CellShape::CellShape(const CellShape &t_cellShape)
    : m_cellMask{t_cellShape.getCellMask()}, m_rowSpacing{t_cellShape.getRowSpacing()},
      m_colSpacing{t_cellShape.getColSpacing()},
      m_alternateRowOffset{t_cellShape.getAlternateRowOffset()},
      m_alternateColOffset{t_cellShape.getAlternateColOffset()} {}

//Writes the CellShape to a QDataStream
QDataStream &operator<<(QDataStream &t_out, const CellShape &t_cellShape)
{
    t_out << t_cellShape.m_cellMask;
    t_out << t_cellShape.m_rowSpacing << t_cellShape.m_colSpacing;
    t_out << t_cellShape.m_alternateRowOffset << t_cellShape.m_alternateColOffset;
    return t_out;
}

//Reads the CellShape from a QDataStream
QDataStream &operator>>(QDataStream &t_in, CellShape &t_cellShape)
{
    t_in >> t_cellShape.m_cellMask;
    t_in >> t_cellShape.m_rowSpacing >> t_cellShape.m_colSpacing;
    t_in >> t_cellShape.m_alternateRowOffset >> t_cellShape.m_alternateColOffset;
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
    }
    else
        qDebug() << "CellShape::setCellMask(const cv::Mat &) unsupported mask type";

    UtilityFuncs::imageToSquare(m_cellMask, UtilityFuncs::SquareMethod::PAD);
}

//Returns the cell mask
cv::Mat CellShape::getCellMask() const
{
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

//Returns the cell shape resized to the given size
CellShape CellShape::resized(const int t_cols, const int t_rows) const
{
    //If cell mask is empty then just return copy of cell shape
    if (empty())
    {
        return CellShape(*this);
    }

    const cv::Mat resizedMask = UtilityFuncs::resizeImage(m_cellMask, t_rows, t_cols,
                                                          UtilityFuncs::ResizeType::INCLUSIVE);

    CellShape result(resizedMask);
    double vRatio = static_cast<double>(resizedMask.rows) / m_cellMask.rows;
    double hRatio = static_cast<double>(resizedMask.cols) / m_cellMask.cols;
    result.setRowSpacing(static_cast<int>(m_rowSpacing * vRatio));
    result.setColSpacing(static_cast<int>(m_colSpacing * hRatio));
    result.setAlternateRowOffset(static_cast<int>(m_alternateRowOffset * vRatio));
    result.setAlternateColOffset(static_cast<int>(m_alternateColOffset * hRatio));

    return result;
}

//Returns if the cell mask is empty
bool CellShape::empty() const
{
    return m_cellMask.empty();
}

//Returns rect of cell shape at the given grid position
cv::Rect CellShape::getRectAt(const int x, const int y) const
{
    cv::Rect result;
    result.x = x * m_colSpacing + ((abs(y % 2) == 1) ? m_alternateRowOffset : 0);
    result.y = y * m_rowSpacing + ((abs(x % 2) == 1) ? m_alternateColOffset : 0);
    result.width = m_cellMask.cols;
    result.height = m_cellMask.rows;
    return result;
}
