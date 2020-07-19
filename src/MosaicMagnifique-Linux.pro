#	Copyright © 2018-2020, Morgan Grundy
#
#	This file is part of Mosaic Magnifique.
#
#    Mosaic Magnifique is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    Mosaic Magnifique is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with Mosaic Magnifique.  If not, see <https://www.gnu.org/licenses/>.

QT += core gui
greaterThan(QT_MAJOR_VERSION, 4): QT += widgets

CONFIG += c++17
DEFINES += _USE_MATH_DEFINES
DESTDIR = ..

TARGET = MosaicMagnifique

# The following define makes your compiler emit warnings if you use
# any Qt feature that has been marked deprecated (the exact warnings
# depend on your compiler). Please consult the documentation of the
# deprecated API in order to know how to port your code away from it.
DEFINES += QT_DEPRECATED_WARNINGS

SOURCES += \
	cellgrid.cpp \
	cellshape.cpp \
	colourvisualisation.cpp \
	customgraphicsview.cpp \
	gridbounds.cpp \
    	gridviewer.cpp \
	halvingspinbox.cpp \
	imageviewer.cpp \
    	main.cpp \
    	mainwindow.cpp \
    	photomosaicgenerator.cpp \
    	utilityfuncs.cpp

HEADERS += \
	cellgrid.h \
	cellshape.h \
	colourvisualisation.h \
	customgraphicsview.h \
	gridbounds.h \
    	gridviewer.h \
	halvingspinbox.h \
	imageviewer.h \
    	mainwindow.h \
    	photomosaicgenerator.h \
    	utilityfuncs.h

FORMS += \
    	colourvisualisation.ui \
    	imageviewer.ui \
    	mainwindow.ui

RESOURCES += \
	ui.qrc

# OpenCV libraries
CONFIG += link_pkgconfig
PKGCONFIG += opencv4
