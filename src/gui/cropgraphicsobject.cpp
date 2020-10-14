#include "cropgraphicsobject.h"

#include <QPainter>

CropGraphicsObject::CropGraphicsObject(const int t_width, const int t_height,
                                       QGraphicsItem *t_parent)
    : QGraphicsObject(t_parent), m_bounds{0, 0, t_width, t_height}
{
    setAcceptDrops(true);

    //Initialise crop at center
    const int cropSize = std::min(t_width, t_height);
    m_cropBounds = QRect((t_width - cropSize) / 2, (t_height - cropSize) / 2, cropSize, cropSize);
}

//Returns crop
const QRect CropGraphicsObject::getCrop() const
{
    return m_cropBounds;
}

QRectF CropGraphicsObject::boundingRect() const
{
    return m_bounds;
}

//Draws partially transparent rect in areas not in crop
void CropGraphicsObject::paint(QPainter *painter,
                               [[maybe_unused]] const QStyleOptionGraphicsItem *option,
                               [[maybe_unused]] QWidget *widget)
{
    painter->setBrush(QBrush(QColor(0, 0, 0, 170)));
    painter->setPen(Qt::PenStyle::NoPen);

    //Create rect of area before crop selection
    QRect beforeRect(0, 0, m_cropBounds.x(), m_cropBounds.y());
    if (m_bounds.width() > m_bounds.height())
        beforeRect.setHeight(m_bounds.height());
    else
        beforeRect.setWidth(m_bounds.width());
    painter->drawRect(beforeRect);

    //Create rect of area after crop selection
    QRect afterRect(0, 0, m_bounds.x(), m_bounds.y());
    if (m_bounds.width() > m_bounds.height())
    {
        afterRect.setHeight(m_bounds.height());
        afterRect.setX(m_cropBounds.right());
        afterRect.setWidth(m_bounds.width() - afterRect.x());
    }
    else
    {
        afterRect.setWidth(m_bounds.width());
        afterRect.setY(m_cropBounds.bottom());
        afterRect.setHeight(m_bounds.height() - afterRect.y());
    }
    painter->drawRect(afterRect);
}

//Gets intial click position
void CropGraphicsObject::mousePressEvent(QGraphicsSceneMouseEvent *event)
{
    //If left mouse button pressed
    if (event->buttons() & Qt::MouseButton::LeftButton)
        m_clickStart = mapToScene(event->pos()).toPoint();
}

//Moves crop
void CropGraphicsObject::mouseMoveEvent(QGraphicsSceneMouseEvent *event)
{
    //If left mouse button pressed
    if (event->buttons() & Qt::MouseButton::LeftButton)
    {
        //Get offset for crop
        const QPoint newPosition = mapToScene(event->pos()).toPoint();
        QPoint offset = newPosition - m_clickStart;

        //Prevent crop from exceeding bounds
        if (m_bounds.width() > m_bounds.height())
        {
            offset.setY(0);
            if (m_cropBounds.left() + offset.x() < 0)
                offset.setX(-m_cropBounds.left());
            else if (m_cropBounds.right() + offset.x() > m_bounds.width())
                offset.setX(m_bounds.width() - m_cropBounds.right() - 1);
        }
        else
        {
            offset.setX(0);
            if (m_cropBounds.top() + offset.y() < 0)
                offset.setY(-m_cropBounds.top());
            else if (m_cropBounds.bottom() + offset.y() > m_bounds.height())
                offset.setY(m_bounds.height() - m_cropBounds.bottom() - 1);
        }

        //Move crop and update
        if (offset.x() != 0 || offset.y() != 0)
        {
            m_clickStart = newPosition;
            m_cropBounds.translate(offset);
            update();
        }
    }
}
