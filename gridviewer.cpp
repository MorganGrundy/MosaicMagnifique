#include "gridviewer.h"

#include <QPainter>

GridViewer::GridViewer(QWidget *parent) : QWidget(parent)
{

}

void GridViewer::paintEvent(QPaintEvent *event)
{
    QPainter painter(this);
}
