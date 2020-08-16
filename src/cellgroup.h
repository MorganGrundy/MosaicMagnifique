#ifndef CELLGROUP_H
#define CELLGROUP_H

#include "cellshape.h"

#include <opencv2/core.hpp>

class CellGroup
{
public:
    CellGroup();

    //Sets cell shape
    void setCellShape(const CellShape &t_cellShape);

    //Sets detail level
    void setDetail(const int t_detail = 100, const bool t_reset = false);
    //Returns the detail level
    double getDetail() const;

    //Sets the number of size steps, for each step cell is halved in size
    void setSizeSteps(const size_t t_steps, const bool t_reset = false);
    //Returns the size steps
    size_t getSizeSteps() const;

    //Returns reference to normal and detail cell shapes
    CellShape &getCell(size_t t_sizeStep, bool t_detail = false);
    //Return const reference to normal and detail cell shapes
    const CellShape &getCell(size_t t_sizeStep, bool t_detail = false) const;
    //Returns reference to an edge cell mask
    cv::Mat &getEdgeCell(size_t t_sizeStep, bool t_flipHorizontal, bool t_flipVertical);
    //Returns const reference to an edge cell mask
    const cv::Mat &getEdgeCell(size_t t_sizeStep, bool t_flipHorizontal, bool t_flipVertical) const;

private:
    std::vector<CellShape> cells;
    std::vector<CellShape> detailCells;
    std::vector<std::vector<cv::Mat>> edgeCells;

    double detail;
    size_t sizeSteps;
};

#endif // CELLGROUP_H
