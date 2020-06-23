#ifndef GRIDBOUNDS_H
#define GRIDBOUNDS_H

#include <opencv2/core.hpp>

class GridBounds
{
public:
    GridBounds();

    void addBound(const cv::Rect &t_bound);
    void addBound(const int t_height, const int t_width);
    void clear();

    std::vector<cv::Rect>::const_iterator cbegin() const;
    std::vector<cv::Rect>::const_iterator cend() const;

    bool empty();

private:
    std::vector<cv::Rect> bounds;
};

#endif // GRIDBOUNDS_H
