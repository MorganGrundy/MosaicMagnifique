#include "switch.h"

#include <QPainter>

Switch::Switch(QWidget *t_parent)
    : QAbstractButton{t_parent}
{
    setCheckable(true);
    setChecked(false);
}

QSize Switch::sizeHint() const
{
    return QSize(40, 20);
}

//Returns switch state
Switch::SwitchState Switch::getState() const
{
    return isChecked() ? SwitchState::RIGHT : SwitchState::LEFT;
}

void Switch::paintEvent([[maybe_unused]] QPaintEvent *t_event)
{
    QPainter painter(this);
    painter.setRenderHint(QPainter::Antialiasing);

    //Border
    QPen pen(QColor(105, 105, 105));
    pen.setWidth(2);
    painter.setPen(pen);

    const qreal width = geometry().width();
    const qreal height = geometry().height();
    const qreal margin = height * 0.2;

    //Draw switch path/background
    painter.setBrush(QBrush(QColor(60, 60, 60)));
    const qreal radius = (height - margin) / 2;
    painter.drawRoundedRect(QRectF(margin, margin, width - margin * 2, height - margin * 2),
                            radius, radius);

    //Draw switch state
    painter.setBrush(Qt::white);
    painter.drawEllipse(isChecked() ? height : 0, 0, height, height);
}
