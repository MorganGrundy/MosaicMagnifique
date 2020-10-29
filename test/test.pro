include(gtest_dependency.pri)

TEMPLATE = app
CONFIG += console c++17
CONFIG -= app_bundle
CONFIG += thread

include(../common.pri)
include(../src/logic/logic.pri)

HEADERS += \
	testutility.h \
	tst_colourdifference.h \
	tst_cudakernel.h \
	tst_generator.h \
	tst_imagelibrary.h

SOURCES += \
        main.cpp \
        testutility.cpp
