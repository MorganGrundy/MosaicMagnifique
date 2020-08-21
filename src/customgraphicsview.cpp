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

#include "customgraphicsview.h"
#include <QDebug>

CustomGraphicsView::CustomGraphicsView(QWidget *t_parent)
    : QGraphicsView{t_parent}, MIN_ZOOM{0.8}, MAX_ZOOM{10}
{
    setCacheMode(CacheBackground);
    setViewportUpdateMode(BoundingRectViewportUpdate);
    setRenderHint(QPainter::Antialiasing);

    setDragMode(QGraphicsView::ScrollHandDrag);
    setTransformationAnchor(AnchorUnderMouse);

    //Centres image
    Qt::Alignment alignment;
    alignment.setFlag(Qt::AlignmentFlag::AlignHCenter);
    alignment.setFlag(Qt::AlignmentFlag::AlignVCenter);
    setAlignment(alignment);
}

//Resizes image to fit in view
void CustomGraphicsView::fitToView()
{
    fitInView(sceneRect(), Qt::KeepAspectRatio);
}

//Handles scrollwheel event
void CustomGraphicsView::wheelEvent(QWheelEvent *event)
{
    const double zoomScale = 1.1;
    //Ctrl-scrollwheel modifies image zoom
    if (event->modifiers() == Qt::ControlModifier)
    {
        zoom((event->angleDelta().y() > 0) ? zoomScale : 1.0 / zoomScale);
    }
    //Shift-scrollwheel scrolls horizontally
    else if (event->modifiers() == Qt::ShiftModifier)
    {
        //Swap vertical and horizontal delta
        QPoint pixelDelta(event->pixelDelta().y(), event->pixelDelta().x());
        QPoint angleDelta(event->angleDelta().y(), event->angleDelta().x());
        QWheelEvent horizontalScrollEvent(event->position(), event->globalPosition(),
                                          pixelDelta, angleDelta, event->buttons(),
                                          Qt::NoModifier, event->phase(), event->inverted());
        QGraphicsView::wheelEvent(&horizontalScrollEvent);
    }
    //Vertical scrolling
    else if (event->modifiers() == Qt::NoModifier)
        QGraphicsView::wheelEvent(event);
}

//Modifies zoom by factor
void CustomGraphicsView::zoom(const double factor)
{
    if (sceneRect().isEmpty())
        return;

    const QRectF expectedRect = transform().scale(factor, factor).mapRect(sceneRect());
    double expRectLength;
    int viewportLength;

    //Compares ratio between height and width of viewport(widget) and scene(image, etc...)
    if (viewport()->rect().width() / sceneRect().width() >
        viewport()->rect().height() / sceneRect().height())
    {
        //Height has lowest ratio
        expRectLength = expectedRect.height();
        viewportLength = viewport()->rect().height();
    }
    else
    {
        //Width has lowest ratio
        expRectLength = expectedRect.width();
        viewportLength = viewport()->rect().width();
    }

    //Minimum zoom
    if (expRectLength < viewportLength * MIN_ZOOM)
    {
        if (factor < 1)
            return;
    }

    //Maximum zoom
    else if (expRectLength > viewportLength * MAX_ZOOM)
    {
        if (factor > 1)
            return;
    }

    scale(factor, factor);
}
