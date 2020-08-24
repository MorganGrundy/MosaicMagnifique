#include "quadtree.h"

Quadtree::Quadtree(const cv::Rect &t_bounds)
    : m_bounds{t_bounds}
{}

//Try to insert an element into the quadtree
bool Quadtree::insert(const elementType &t_element)
{
    //If bounds do not intersect then do not insert
    const cv::Rect boundIntersect = m_bounds & t_element.first;
    if (boundIntersect.area() == 0)
        return false;

    //Check for space in node and no sub quadrants
    if (m_elements.size() < NODE_CAPACITY && !m_quadrantNW)
    {
        //Add element
        m_elements.push_back(t_element);
        return true;
    }

    //Create sub quadrants if not already created
    if (!m_quadrantNW)
        subdivide();

    //Try to insert into sub quadrants
    bool wasInserted = false;
    wasInserted = m_quadrantNW->insert(t_element) || wasInserted;
    wasInserted = m_quadrantNE->insert(t_element) || wasInserted;;
    wasInserted = m_quadrantSW->insert(t_element) || wasInserted;;
    wasInserted = m_quadrantSE->insert(t_element) || wasInserted;;

    return wasInserted;
}

//Split node into four and sort elements into them
void Quadtree::subdivide()
{
    //Get center point of node
    const cv::Point nodeCenter(m_bounds.x + m_bounds.width / 2, m_bounds.y + m_bounds.height / 2);

    //Create sub quadrants
    //North-West quadrant
    m_quadrantNW = std::make_shared<Quadtree>(
        cv::Rect(m_bounds.x, m_bounds.y,
                 m_bounds.width / 2, m_bounds.height / 2));
    //North-East quadrant
    m_quadrantNE = std::make_shared<Quadtree>(
        cv::Rect(nodeCenter.x, m_bounds.y,
                 m_bounds.width - m_bounds.width / 2, m_bounds.height / 2));
    //South-West quadrant
    m_quadrantSW = std::make_shared<Quadtree>(
        cv::Rect(m_bounds.x, nodeCenter.y,
                 m_bounds.width / 2, m_bounds.height - m_bounds.height / 2));
    //South-East quadrant
    m_quadrantSE = std::make_shared<Quadtree>(
        cv::Rect(nodeCenter.x, nodeCenter.y,
                 m_bounds.width - m_bounds.width / 2, m_bounds.height - m_bounds.height / 2));

    //Insert elements into sub quadrants
    for (const auto element: m_elements)
    {
        m_quadrantNW->insert(element);
        m_quadrantNE->insert(element);
        m_quadrantSW->insert(element);
        m_quadrantSE->insert(element);
    }

    //Remove elements
    m_elements.clear();
}

//Returns all elements at given point
std::vector<Quadtree::elementType> Quadtree::query(const cv::Point t_point) const
{
    //Store all elements at point
    std::vector<elementType> elementsAtPoint;

    //If point not in bounds then return
    if (!m_bounds.contains(t_point))
        return elementsAtPoint;

    //Check if point in element bounds
    for (const auto element: m_elements)
    {
        if (element.first.contains(t_point))
            elementsAtPoint.push_back(element);
    }

    //No sub quadrants, return
    if (!m_quadrantNW)
        return elementsAtPoint;

    //Store elements at point for sub quadrant
    std::vector<elementType> quadrantElementsAtPoint;
    //Get North-West elements at point
    quadrantElementsAtPoint = m_quadrantNW->query(t_point);
    elementsAtPoint.insert(elementsAtPoint.end(), quadrantElementsAtPoint.begin(),
                           quadrantElementsAtPoint.end());
    //Get North-East elements at point
    quadrantElementsAtPoint = m_quadrantNE->query(t_point);
    elementsAtPoint.insert(elementsAtPoint.end(), quadrantElementsAtPoint.begin(),
                           quadrantElementsAtPoint.end());
    //Get South-West elements at point
    quadrantElementsAtPoint = m_quadrantSW->query(t_point);
    elementsAtPoint.insert(elementsAtPoint.end(), quadrantElementsAtPoint.begin(),
                           quadrantElementsAtPoint.end());
    //Get South-East elements at point
    quadrantElementsAtPoint = m_quadrantSE->query(t_point);
    elementsAtPoint.insert(elementsAtPoint.end(), quadrantElementsAtPoint.begin(),
                           quadrantElementsAtPoint.end());

    return elementsAtPoint;
}
