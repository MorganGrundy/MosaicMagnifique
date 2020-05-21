#ifndef CELLSHAPE_H
#define CELLSHAPE_H

#include <opencv2/core/mat.hpp>
#include <QDataStream>

class CellShape
{
public:
    //Constructors
    CellShape();
    CellShape(const cv::Mat &t_cellMask);
    CellShape(const CellShape &t_cellShape);

    //Operators
    friend QDataStream &operator<<(QDataStream &t_out, const CellShape &t_cellShape);
    friend QDataStream &operator>>(QDataStream &t_in, std::pair<CellShape &, const int> t_cellShape);

    void setCellMask(const cv::Mat &t_cellMask);
    cv::Mat getCellMask() const;

    void setRowSpacing(const int t_rowSpacing);
    int getRowSpacing() const;

    void setColSpacing(const int t_colSpacing);
    int getColSpacing() const;

    void setAlternateRowOffset(const int t_alternateRowOffset);
    int getAlternateRowOffset() const;

    void setAlternateColOffset(const int t_alternateColOffset);
    int getAlternateColOffset() const;

    void setHorizontalFlipping(const bool t_horizontalFlipping);
    bool getHorizontalFlipping() const;

    void setVerticalFlipping(const bool t_verticalFlipping);
    bool getVerticalFlipping() const;

    CellShape resized(const int t_cols, const int t_rows) const;

    bool empty() const;

    cv::Rect getRectAt(const int x, const int y) const;

private:
    cv::Mat m_cellMask;

    int m_rowSpacing, m_colSpacing;
    //Entire row/col offset by specified value every other row/col
    int m_alternateRowOffset, m_alternateColOffset;

    bool m_horizontalFlipping, m_verticalFlipping;
};

#endif // CELLSHAPE_H
