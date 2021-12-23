#pragma once

#include <QGraphicsObject>
#include <QGraphicsSceneMouseEvent>

class CropGraphicsObject : public QGraphicsObject
{
public:
    CropGraphicsObject(const int t_width, const int t_height, QGraphicsItem *t_parent = nullptr);

    //Returns crop
    const QRect getCrop() const;
    QRectF boundingRect() const override;

    void paint(QPainter *painter, const QStyleOptionGraphicsItem *option, QWidget *widget) override;

protected:
    //Gets intial click position
    void mousePressEvent(QGraphicsSceneMouseEvent *event) override;

    //Moves crop
    void mouseMoveEvent(QGraphicsSceneMouseEvent *event) override;

private:
    //Space that crop is in
    QRect m_bounds;

    //Position and size of crop
    QRect m_cropBounds;

    //Position of click start
    QPoint m_clickStart;
};