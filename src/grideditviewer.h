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

#ifndef GRIDEDITVIEWER_H
#define GRIDEDITVIEWER_H

#include "gridviewer.h"

#include <QMouseEvent>

class GridEditViewer : public GridViewer
{
    Q_OBJECT
public:
    explicit GridEditViewer(QWidget *parent = nullptr);

    //Sets current size step for editor
    void setSizeStep(const size_t t_sizeStep);

protected:
    //Inverts state of clicked cell
    void mousePressEvent(QMouseEvent *event) override;

private:
    size_t m_sizeStep;
};

#endif // GRIDEDITVIEWER_H
