#ifndef GRIDGENERATOR_H
#define GRIDGENERATOR_H

#include <opencv2/core/mat.hpp>
#include <QImage>

#include "cellshape.h"
#include "gridutility.h"
#include "gridbounds.h"
#include "cellgroup.h"

class GridGenerator
{
public:
    static GridUtility::mosaicBestFit getGridState(const CellGroup &t_cells,
                                                const cv::Mat &t_mainImage,
                                                const int height, const int width);

private:
    GridGenerator();

    static std::pair<GridUtility::cellBestFit, bool>
    findCellState(const CellGroup &t_cells, const cv::Mat &t_mainImage, const int x, const int y,
                  const GridBounds &t_bounds, const size_t t_step = 0);
};

#endif // GRIDGENERATOR_H
