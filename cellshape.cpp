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
        cv::Mat tmp = t_cellMask;
        //Threshold to create binary mask
        cv::threshold(tmp, tmp, 127.0, 255.0, cv::THRESH_BINARY);
        //Convert to RGBA
        cv::cvtColor(tmp, tmp, cv::COLOR_GRAY2RGBA);

        //Replace black with transparent
        int channels = tmp.channels();
        int nRows = tmp.rows;
        int nCols = tmp.cols * channels;
        if (tmp.isContinuous())
        {
            nCols *= nRows;
            nRows = 1;
        }

        uchar *p;
        for (int i = 0; i < nRows; ++i)
        {
            p = tmp.ptr<uchar>(i);
            for (int j = 0; j < nCols; j += channels)
            {
                if (p[j] == 0)
                    p[j+3] = 0;
            }
        }
        m_cellMask = tmp;
    }
    //Mask is RGBA
    else if (t_cellMask.type() == CV_8UC4)
        m_cellMask = t_cellMask.clone();
    else
        qDebug() << "CellShape::setCellMask(const cv::Mat &) unsupported mask type";
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