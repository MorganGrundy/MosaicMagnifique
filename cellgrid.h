#ifndef CELLGRID_H
#define CELLGRID_H

#include <opencv2/core.hpp>
#include <optional>

#include "cellshape.h"

class CellGrid
{
public:
    typedef std::pair<std::optional<size_t>, bool> cellBestFit;
    typedef std::vector<std::vector<cellBestFit>> stepBestFit;
    typedef std::vector<stepBestFit> mosaicBestFit;

    static const int PAD_GRID = 2;

    static cv::Point calculateGridSize(const CellShape &t_cellShape,
                                       const int t_imageWidth, const int t_imageHeight,
                                       const int t_pad);

    static cv::Rect getRectAt(const CellShape &t_cellShape, const int t_x, const int t_y);

    static std::pair<bool, bool> getFlipStateAt(const CellShape &t_cellShape,
                                                const int t_x, const int t_y, const int t_pad);

    //Maximum possible entropy value
    static double MAX_ENTROPY() {return 8.0;};
    static double calculateEntropy(const cv::Mat &t_mask, const cv::Mat &t_image);

private:
    CellGrid() {}
};

#endif // CELLGRID_H
