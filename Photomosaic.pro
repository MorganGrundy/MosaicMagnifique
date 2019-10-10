QT += core gui

greaterThan(QT_MAJOR_VERSION, 4): QT += widgets

CONFIG += c++17

# The following define makes your compiler emit warnings if you use
# any Qt feature that has been marked deprecated (the exact warnings
# depend on your compiler). Please consult the documentation of the
# deprecated API in order to know how to port your code away from it.
DEFINES += QT_DEPRECATED_WARNINGS

# You can also make your code fail to compile if it uses deprecated APIs.
# In order to do so, uncomment the following line.
# You can also select to disable deprecated APIs only up to a certain version of Qt.
#DEFINES += QT_DISABLE_DEPRECATED_BEFORE=0x060000    # disables all the APIs deprecated before Qt 6.0.0

SOURCES += \
    gridviewer.cpp \
    main.cpp \
    mainwindow.cpp \
    photomosaicgenerator.cpp \
    utilityfuncs.cpp

HEADERS += \
    gridviewer.h \
    mainwindow.h \
    photomosaicgenerator.h \
    utilityfuncs.h

FORMS += \
    mainwindow.ui

# Boost libraries
INCLUDEPATH += $$(BOOST_ROOT)
LIBS += -L$$(BOOST_ROOT)/stage/lib

# OpenCV libraries
INCLUDEPATH += $$(OPENCV_SDK_DIR)/include
CONFIG( debug, debug|release ) {
	# debug
	LIBS += -L$$(OPENCV_DIR)/lib \
	-lopencv_core411d \
	-lopencv_highgui411d \
	-lopencv_imgcodecs411d \
	-lopencv_imgproc411d \
	-lopencv_features2d411d \
	-lopencv_calib3d411d \
	-lopencv_cudaarithm411d \
	-lopencv_cudawarping411d \
	-lopencv_cudaimgproc411d
} else {
	# release
	LIBS += -L$$(OPENCV_DIR)/lib \
	-lopencv_core411 \
	-lopencv_highgui411 \
	-lopencv_imgcodecs411 \
	-lopencv_imgproc411 \
	-lopencv_features2d411 \
	-lopencv_calib3d411 \
	-lopencv_cudaarithm411 \
	-lopencv_cudawarping411 \
	-lopencv_cudaimgproc411
}

# Default rules for deployment.
qnx: target.path = /tmp/$${TARGET}/bin
else: unix:!android: target.path = /opt/$${TARGET}/bin
!isEmpty(target.path): INSTALLS += target

DISTFILES +=

RESOURCES += \
	ui.qrc
