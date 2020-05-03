#include "customgraphicsview.h"
#include <QDebug>

CustomGraphicsView::CustomGraphicsView(QWidget *t_parent)
    : QGraphicsView{t_parent}
{
    setCacheMode(CacheBackground);
    setViewportUpdateMode(BoundingRectViewportUpdate);
    setRenderHint(QPainter::HighQualityAntialiasing);

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
        zoom((event->delta() > 0) ? zoomScale : 1.0 / zoomScale);
    }
    //Shift-scrollwheel scrolls horizontally
    else if (event->modifiers() == Qt::ShiftModifier)
    {
        QWheelEvent horizontalScrollEvent(event->pos(), event->delta(), event->buttons(),
                                          Qt::NoModifier, Qt::Horizontal);
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
    int imgLength;

    if (sceneRect().width() > sceneRect().height())
    {
        expRectLength = expectedRect.width();
        viewportLength = viewport()->rect().width();
        imgLength = sceneRect().width();
    }
    else
    {
        expRectLength = expectedRect.height();
        viewportLength = viewport()->rect().height();
        imgLength = sceneRect().height();
    }

    //Minimum zoom
    if (expRectLength < viewportLength / 1.5)
    {
        if (factor < 1)
            return;
    }
    //Maximum zoom
    else if (expRectLength > imgLength * 10)
    {
        if (factor > 1)
            return;
    }

    scale(factor, factor);
}
