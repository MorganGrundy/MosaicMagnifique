#ifndef SWITCH_H
#define SWITCH_H

#include <QAbstractButton>
#include <QPaintEvent>
#include <QMouseEvent>

class Switch : public QAbstractButton
{
    Q_OBJECT
public:
    enum class SwitchState {LEFT, RIGHT};

    Switch(QWidget *t_parent = nullptr);

    QSize sizeHint() const override;

    //Returns switch state
    SwitchState getState() const;

    //Sets text
    void setText(const QString &t_text, const SwitchState t_state);

    //Sets colour
    void setColour(const QColor &t_colour, const SwitchState t_state);

public slots:
    void paintEvent(QPaintEvent *t_event) override;

private:
    QFontMetrics m_fontMetric;
    QString m_leftText, m_rightText;
    //Text size
    int m_leftTextWidth, m_rightTextWidth, m_textHeight;
    //Switch size
    qreal m_height, m_width, m_margin;

    //Colour of switch head in each position
    QColor m_leftColour, m_rightColour;
};

#endif // SWITCH_H
