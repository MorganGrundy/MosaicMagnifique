HEADERS += \
	$$PWD/cellgroup.h \
	$$PWD/cellshape.h \
	$$PWD/colourdifference.h \
	$$PWD/cpuphotomosaicgenerator.h \
	$$PWD/gridbounds.h \
	$$PWD/gridgenerator.h \
	$$PWD/gridutility.h \
	$$PWD/imagehistogramcompare.h \
	$$PWD/imageutility.h \
	$$PWD/photomosaicgeneratorbase.h \
	$$PWD/quadtree.h

SOURCES += \
	$$PWD/cellgroup.cpp \
	$$PWD/cellshape.cpp \
	$$PWD/colourdifference.cpp \
	$$PWD/cpuphotomosaicgenerator.cpp \
	$$PWD/gridbounds.cpp \
	$$PWD/gridgenerator.cpp \
	$$PWD/gridutility.cpp \
	$$PWD/imagehistogramcompare.cpp \
	$$PWD/imageutility.cpp \
	$$PWD/photomosaicgeneratorbase.cpp \
	$$PWD/quadtree.cpp

CUDA {
	CUDA_SOURCES += \
		$$PWD/photomosaicgenerator.cu \
		$$PWD/reduction.cu

	CUDA_HEADERS += \
		$$PWD/reduction.cuh

	SOURCES += \
		$$PWD/cudaphotomosaicdata.cpp \
		$$PWD/cudaphotomosaicgenerator.cpp

	HEADERS += \
		$$PWD/cudaphotomosaicdata.h \
		$$PWD/cudaphotomosaicgenerator.h

	DISTFILES += \
		$$PWD/reduction.cuh
}

INCLUDEPATH += $$PWD
