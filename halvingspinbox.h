#ifndef EXPONENTIATIONSPINBOX_H
#define EXPONENTIATIONSPINBOX_H

#include <QSpinBox>

class HalvingSpinBox : public QSpinBox
{
    Q_OBJECT
public:
    HalvingSpinBox(QWidget *t_parent = nullptr);

    void stepBy(int steps) override;

private:
    int halveSteps = 0;
};

#endif // EXPONENTIATIONSPINBOX_H
