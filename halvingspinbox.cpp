#include "halvingspinbox.h"

#include <QLineEdit>

HalvingSpinBox::HalvingSpinBox(QWidget *t_parent)
    : QSpinBox(t_parent)
{
    setButtonSymbols(ButtonSymbols::PlusMinus);
    lineEdit()->setReadOnly(true);
    setMinimum(1);
    lineEdit()->setText(QString::number(maximum()));
}

void HalvingSpinBox::stepBy(int steps)
{
    int newValue = maximum();

    //Prevent negative steps
    int step = std::max(0, halveSteps - steps);
    //For every step halve value, stop once minimum reached
    while (step > 0 && newValue > minimum())
    {
        newValue /= 2;
        --step;
    }

    //Store new step
    halveSteps -= steps + step;
    //Display new value
    lineEdit()->setText(QString::number(newValue));
}
