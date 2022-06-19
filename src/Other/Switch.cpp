#include "Switch.h"

#include <QPainter>
#include <stdexcept>

Switch::Switch(QWidget *t_parent)
    : QAbstractButton{t_parent}, m_fontMetric{QFont()}, m_leftColour{Qt::white}, m_rightColour{Qt::white}
{
    setCheckable(true);
    setChecked(false);

    //Calculate text size
    m_leftTextWidth = m_fontMetric.horizontalAdvance(m_leftText);
    m_rightTextWidth = m_fontMetric.horizontalAdvance(m_rightText);
    m_textHeight = m_fontMetric.height();

    //Calculate switch size
    m_height = m_textHeight * 10/6.0;
    m_width = m_height * 2;
    m_margin = m_height * 0.2;
}

QSize Switch::sizeHint() const
{
    return QSize(m_leftTextWidth + m_width + m_margin * 2 + m_rightTextWidth, m_height);
}

//Returns switch state
Switch::SwitchState Switch::getState() const
{
    return isChecked() ? SwitchState::RIGHT : SwitchState::LEFT;
}

//Sets text
void Switch::setText(const QString &t_text, const SwitchState t_state)
{
    if (t_state == SwitchState::LEFT)
    {
        m_leftText = t_text;
        m_leftTextWidth = m_fontMetric.horizontalAdvance(m_leftText);
    }
    else if (t_state == SwitchState::RIGHT)
    {
        m_rightText = t_text;
        m_rightTextWidth = m_fontMetric.horizontalAdvance(m_rightText);
    }
    else
        throw std::invalid_argument(Q_FUNC_INFO" unknown state in t_state");
}

//Sets colour
void Switch::setColour(const QColor &t_colour, const SwitchState t_state)
{
    if (t_state == SwitchState::LEFT)
        m_leftColour = t_colour;
    else if (t_state == SwitchState::RIGHT)
        m_rightColour = t_colour;
    else
        throw std::invalid_argument(Q_FUNC_INFO" unknown state in t_state");
}

void Switch::paintEvent([[maybe_unused]] QPaintEvent *t_event)
{
    QPainter painter(this);
    painter.setRenderHint(QPainter::Antialiasing);

    //Draw background
    painter.setBrush(QBrush(QColor(60, 60, 60, 220)));
    painter.setPen(Qt::NoPen);
    painter.drawRect(0, 0, m_leftTextWidth + m_width + m_margin * 2 + m_rightTextWidth, m_height);

    //Draw text
    QPen pen(Qt::white);
    pen.setWidth(2);
    painter.setPen(pen);
    painter.drawText(0, m_textHeight + (m_height - m_textHeight) / 3, m_leftText);
    painter.drawText(m_leftTextWidth + m_width + m_margin * 2,
                     m_textHeight + (m_height - m_textHeight) / 3, m_rightText);

    //Border
    pen.setColor(QColor(105, 105, 105));
    painter.setPen(pen);

    //Draw switch path/background
    painter.setBrush(QBrush(QColor(60, 60, 60)));
    const qreal radius = m_height / 2 - m_margin;
    painter.drawRoundedRect(QRectF(m_leftTextWidth + m_margin * 2, m_margin,
                                   m_width - m_margin * 2, m_height - m_margin * 2),
                            radius, radius);

    //Draw switch head
    painter.setBrush(isChecked() ? m_rightColour : m_leftColour);
    painter.drawEllipse(m_leftTextWidth + m_margin + (isChecked() ? m_height : 0), 0,
                        m_height, m_height);
}
