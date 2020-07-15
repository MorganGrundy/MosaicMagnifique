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
    along with Mosaic Magnifique.  If not, see <https://www.gnu.org/licenses/>.
*/

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
