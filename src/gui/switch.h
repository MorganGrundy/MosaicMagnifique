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
    void setText(const QString &t_text, const bool t_checked);

public slots:
    void paintEvent(QPaintEvent *t_event) override;

private:
};

#endif // SWITCH_H
