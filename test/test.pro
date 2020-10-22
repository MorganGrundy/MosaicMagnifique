include(gtest_dependency.pri)

TEMPLATE = app
CONFIG += console c++17
CONFIG -= app_bundle
CONFIG += thread

include(../common.pri)
include(../src/logic/logic.pri)

HEADERS += \
	tst_colourdifference.h \
	tst_cudakernel.h

SOURCES += \
        main.cpp
