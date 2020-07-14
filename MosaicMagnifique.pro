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
#    along with Foobar.  If not, see <https://www.gnu.org/licenses/>.

QT += core gui

greaterThan(QT_MAJOR_VERSION, 4): QT += widgets

CONFIG += c++17 CUDA OPENCV_W_CUDA
DEFINES += _USE_MATH_DEFINES

TARGET = "Mosaic Magnifique"

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

# OpenCV libraries
INCLUDEPATH += $$(OPENCV_SDK_DIR)/include
CONFIG( debug, debug|release ) {
	# debug
	LIBS += -L$$(OPENCV_DIR)/lib \
	-lopencv_core411d \
	-lopencv_imgcodecs411d \
	-lopencv_imgproc411d
} else {
	# release
	LIBS += -L$$(OPENCV_DIR)/lib \
	-lopencv_core411 \
	-lopencv_imgcodecs411 \
	-lopencv_imgproc411
}

CUDA {
DEFINES += CUDA
# Define output directories
CUDA_OBJECTS_DIR = OBJECTS_DIR/../cuda

CUDA_SOURCES += \
	photomosaicgenerator.cu \
	reduction.cu

CUDA_HEADERS += \
	reduction.cuh

SOURCES += \
	cudaphotomosaicdata.cpp

HEADERS += \
	cudaphotomosaicdata.h

DISTFILES += \
	reduction.cuh

# MSVCRT link option (static or dynamic, it must be the same with your Qt SDK link option)
MSVCRT_LINK_FLAG_DEBUG   = "/MDd"
MSVCRT_LINK_FLAG_RELEASE = "/MD"

# CUDA settings
CUDA_DIR = $$(CUDA_PATH)            # Path to cuda toolkit install
SYSTEM_NAME = x64                   # Depending on your system either 'Win32', 'x64', or 'Win64'
SYSTEM_TYPE = 64                    # '32' or '64', depending on your system
CUDA_ARCH = sm_61                   # Type of CUDA architecture
NVCC_OPTIONS = --use_fast_math

# CUDA libraries
INCLUDEPATH += $$CUDA_DIR/include \
			   $$CUDA_DIR/common/inc \
			   $$CUDA_DIR/../shared/inc

QMAKE_LIBDIR += $$CUDA_DIR/lib/$$SYSTEM_NAME \
				$$CUDA_DIR/common/lib/$$SYSTEM_NAME \
				$$CUDA_DIR/../shared/lib/$$SYSTEM_NAME

# The following makes sure all path names (which often include spaces) are put between quotation marks
CUDA_INC = $$join(INCLUDEPATH,'" -I"','-I"','"')

# Add the necessary libraries
LIBS += -lcuda -lcudart

# Configuration of the Cuda compiler
CONFIG(debug, debug|release) {
	# Debug mode
	cuda_d.input = CUDA_SOURCES
	cuda_d.output = $$CUDA_OBJECTS_DIR/${QMAKE_FILE_BASE}_cuda.obj
	cuda_d.commands = $$CUDA_DIR/bin/nvcc.exe -D_DEBUG $$NVCC_OPTIONS $$CUDA_INC $$LIBS \
					  --machine $$SYSTEM_TYPE -arch=$$CUDA_ARCH \
					  --compile -cudart static -g -DWIN32 -D_MBCS \
					  -Xcompiler "/wd4819,/EHsc,/W3,/nologo,/Od,/Zi,/RTC1,/FS" \
					  -Xcompiler $$MSVCRT_LINK_FLAG_DEBUG\
					  -c -o ${QMAKE_FILE_OUT} ${QMAKE_FILE_NAME}
	cuda_d.dependency_type = TYPE_C
	QMAKE_EXTRA_COMPILERS += cuda_d
}
else {
	# Release mode
	cuda.input = CUDA_SOURCES
	cuda.output = $$CUDA_OBJECTS_DIR/${QMAKE_FILE_BASE}_cuda.obj
	cuda.commands = $$CUDA_DIR/bin/nvcc.exe $$NVCC_OPTIONS $$CUDA_INC $$LIBS \
					--machine $$SYSTEM_TYPE -arch=$$CUDA_ARCH \
					--compile -cudart static -DWIN32 -D_MBCS \
					-Xcompiler "/wd4819,/EHsc,/W3,/nologo,/O2,/Zi,/FS" \
					-Xcompiler $$MSVCRT_LINK_FLAG_RELEASE \
					-c -o ${QMAKE_FILE_OUT} ${QMAKE_FILE_NAME}
	cuda.dependency_type = TYPE_C
	QMAKE_EXTRA_COMPILERS += cuda
}
}

OPENCV_W_CUDA {
DEFINES += OPENCV_W_CUDA
CONFIG( debug, debug|release ) {
	# debug
	LIBS += -lopencv_cudaarithm411d \
	-lopencv_cudawarping411d \
	-lopencv_cudaimgproc411d
} else {
	# release
	LIBS += -lopencv_cudaarithm411 \
	-lopencv_cudawarping411 \
	-lopencv_cudaimgproc411
}
}

# Default rules for deployment.
qnx: target.path = /tmp/$${TARGET}/bin
else: unix:!android: target.path = /opt/$${TARGET}/bin
!isEmpty(target.path): INSTALLS += target

RESOURCES += \
	ui.qrc