/*
	Copyright © 2018-2020, Morgan Grundy

	This file is part of Mosaic Magnifique.

    Mosaic Magnifique is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    Mosaic Magnifique is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with Mosaic Magnifique.  If not, see <https://www.gnu.org/licenses/>.
*/

#include "GridBounds.h"

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

//Attempts to merge bounds together
//Bounds must be touching and have either matching y and height, or x and width
//Or be a sub-section of another
void GridBounds::mergeBounds()
{
    bool boundWasMerged = true;

    //Repeat until no bounds are merged
    while (boundWasMerged)
    {
        boundWasMerged = false;
        //Iterate over all pairs of bounds
        for (auto firstIt = bounds.begin(); firstIt + 1 != bounds.end();)
        {
            for (auto secondIt = firstIt + 1; secondIt != bounds.end()
                 && firstIt + 1 != bounds.end();)
            {
                bool merge = false;
                //Check that x and width match
                if (firstIt->x == secondIt->x
                        && firstIt->width == secondIt->width)
                {
                    const int yDiff = secondIt->y - firstIt->y;
                    //Check that bounds touch
                    if (yDiff == 0)
                        merge = true;
                    else if (yDiff > 0 && yDiff <= firstIt->height)
                        merge = true;
                    else if (yDiff < 0 && -yDiff <= secondIt->height)
                        merge = true;

                }
                //Check that y and height match
                else if (firstIt->y == secondIt->y
                         && firstIt->height == secondIt->height)
                {
                    const int xDiff = secondIt->x - firstIt->x;
                    //Check that bounds touch
                    if (xDiff == 0)
                        merge = true;
                    else if (xDiff > 0 && xDiff <= firstIt->width)
                        merge = true;
                    else if (xDiff < 0 && -xDiff <= secondIt->width)
                        merge = true;
                }
                else
                {
                    //Check if either bound is a sub-section of the other
                    const cv::Rect boundUnion = *firstIt | *secondIt;
                    if (boundUnion == *firstIt || boundUnion == *secondIt)
                        merge = true;
                }

                if (merge)
                {
                    //Remove bounds and add merged bound
                    const cv::Rect mergedBound = *firstIt | *secondIt;
                    *firstIt = mergedBound;
                    secondIt = bounds.erase(secondIt);
                    boundWasMerged = true;
                }
                else
                    ++secondIt;
            }
            if (firstIt + 1 != bounds.end())
                ++firstIt;
        }
    }
}

//Removes all bounds
void GridBounds::clear()
{
    bounds.clear();
}

//Returns const iterator to start of bounds
std::vector<cv::Rect>::const_iterator GridBounds::cbegin() const
{
    return bounds.cbegin();
}

//Returns const iterator to end of bounds
std::vector<cv::Rect>::const_iterator GridBounds::cend() const
{
    return bounds.cend();
}

//Returns if there are no bounds
bool GridBounds::empty() const
{
    return bounds.empty();
}
