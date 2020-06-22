#include "gridbounds.h"

GridBounds::GridBounds() {}

//Adds bound
void GridBounds::addBound(const cv::Rect &t_bound)
{
    bounds.push_back(t_bound);
}

//Creates and adds bound
void GridBounds::addBound(const int t_height, const int t_width)
{
    addBound(cv::Rect(0, 0, t_width, t_height));
}

std::vector<cv::Rect>::const_iterator GridBounds::cbegin() const
{
    return bounds.cbegin();
}

std::vector<cv::Rect>::const_iterator GridBounds::cend() const
{
    return bounds.cend();
}
