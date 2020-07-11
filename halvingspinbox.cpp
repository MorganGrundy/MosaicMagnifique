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
    size_t step = 0;
    //For every step halve value, stop once minimum reached
    while (step < static_cast<size_t>(std::max(0, static_cast<int>(halveSteps) - steps)) && newValue > minimum())
    {
        newValue /= 2;
        ++step;
    }

    //Store new step
    halveSteps = step;
    //Display new value
    lineEdit()->setText(QString::number(newValue));
}

size_t HalvingSpinBox::getHalveSteps() const
{
    return halveSteps;
}
