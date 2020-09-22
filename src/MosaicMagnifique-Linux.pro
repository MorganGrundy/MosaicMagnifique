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
QMAKE_CXXFLAGS += -std=c++17

DEFINES += _USE_MATH_DEFINES

#Application version
VERSION_MAJOR = 3
VERSION_MINOR = 1
VERSION_BUILD = 2

DEFINES += "VERSION_MAJOR=$$VERSION_MAJOR" \
	"VERSION_MINOR=$$VERSION_MINOR" \
	"VERSION_BUILD=$$VERSION_BUILD"

TARGET = MosaicMagnifique

# The following define makes your compiler emit warnings if you use
# any Qt feature that has been marked deprecated (the exact warnings
# depend on your compiler). Please consult the documentation of the
# deprecated API in order to know how to port your code away from it.
DEFINES += QT_DEPRECATED_WARNINGS

SOURCES += \
	cellgroup.cpp \
	cellshape.cpp \
	cellshapeeditor.cpp \
	colourdifference.cpp \
	colourvisualisation.cpp \
	cpuphotomosaicgenerator.cpp \
	cropgraphicsobject.cpp \
	customgraphicsview.cpp \
	gridbounds.cpp \
	grideditor.cpp \
	grideditviewer.cpp \
	gridgenerator.cpp \
	gridutility.cpp \
	gridviewer.cpp \
	halvingspinbox.cpp \
	imagehistogramcompare.cpp \
	imagelibraryeditor.cpp \
	imagesquarer.cpp \
	imageutility.cpp \
	main.cpp \
	mainwindow.cpp \
	photomosaicgeneratorbase.cpp \
	photomosaicviewer.cpp \
	quadtree.cpp

HEADERS += \
	cellgroup.h \
	cellshape.h \
	cellshapeeditor.h \
	colourdifference.h \
	colourvisualisation.h \
	cpuphotomosaicgenerator.h \
	cropgraphicsobject.h \
	customgraphicsview.h \
	gridbounds.h \
	grideditor.h \
	grideditviewer.h \
	gridgenerator.h \
	gridutility.h \
	gridviewer.h \
	halvingspinbox.h \
	imagehistogramcompare.h \
	imagelibraryeditor.h \
	imagesquarer.h \
	imageutility.h \
	mainwindow.h \
	photomosaicgeneratorbase.h \
	photomosaicviewer.h \
	quadtree.h

FORMS += \
	cellshapeeditor.ui \
	colourvisualisation.ui \
	grideditor.ui \
	imagelibraryeditor.ui \
	imagesquarer.ui \
	mainwindow.ui \
	photomosaicviewer.ui

RESOURCES += \
	ui.qrc

# OpenCV libraries
CONFIG += link_pkgconfig
PKGCONFIG += opencv4
