#ifndef CELLGRID_H
#define CELLGRID_H

#include "cellshape.h"

class CellGrid
{
public:
    static cv::Point calculateGridSize(const CellShape &t_cellShape,
                                       const int t_imageWidth, const int t_imageHeight,
                                       const int t_pad);

    static cv::Rect getRectAt(const CellShape &t_cellShape, const int t_x, const int t_y);

    static std::pair<bool, bool> getFlipStateAt(const CellShape &t_cellShape,
                                                const int t_x, const int t_y, const int t_pad);

private:
    CellGrid() {}
};

#endif // CELLGRID_H
