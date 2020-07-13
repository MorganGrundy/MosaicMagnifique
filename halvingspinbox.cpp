/*
	Copyright © 2018-2020, Morgan Grundy

	This file is part of Mosaic Magnifique.

    Mosaic Magnifique is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    Mosaic Magnifique is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with Foobar.  If not, see <https://www.gnu.org/licenses/>.
*/

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
    while (step < static_cast<size_t>(std::max(0, static_cast<int>(halveSteps) - steps))
           && newValue > minimum())
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
