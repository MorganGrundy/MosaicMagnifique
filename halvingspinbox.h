#ifndef EXPONENTIATIONSPINBOX_H
#define EXPONENTIATIONSPINBOX_H

#include <QSpinBox>

class HalvingSpinBox : public QSpinBox
{
    Q_OBJECT
public:
    HalvingSpinBox(QWidget *t_parent = nullptr);

    void stepBy(int steps) override;

    size_t getHalveSteps() const;

private:
    size_t halveSteps = 0;
};

#endif // EXPONENTIATIONSPINBOX_H
