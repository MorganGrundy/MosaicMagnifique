#ifndef QUADTREE_H
#define QUADTREE_H

#include <vector>
#include <memory>
#include <opencv2/core.hpp>

//Quadtree implementation for use with cell grid
class Quadtree
{
public:
    //Each element has bounding box and cell x,y
    typedef std::pair<cv::Rect, cv::Point> elementType;

    //Constructor
    Quadtree(const cv::Rect &t_bounds);

    //Try to insert an element into the quadtree
    bool insert(const elementType &t_element);

    //Split node into four and sort elements into them
    void subdivide();

    //Returns all elements at given point
    std::vector<elementType> query(const cv::Point t_point) const;
    //Returns all elements that intersect rect
    std::vector<elementType> query(const cv::Rect t_rect) const;

private:
    //Push_back t_element to t_vector if not already in t_vector
    void push_back_unique(std::vector<elementType> &t_vector, const elementType &t_element) const;

    //Number of elements a node can hold before splitting
    const size_t NODE_CAPACITY = 10;

    //Quadtree node bounding box
    cv::Rect m_bounds;

    //Stores elements in node
    std::vector<elementType> m_elements;

    //Pointers to sub quadrants
    std::shared_ptr<Quadtree> m_quadrantNW;
    std::shared_ptr<Quadtree> m_quadrantNE;
    std::shared_ptr<Quadtree> m_quadrantSW;
    std::shared_ptr<Quadtree> m_quadrantSE;
};

#endif // QUADTREE_H
