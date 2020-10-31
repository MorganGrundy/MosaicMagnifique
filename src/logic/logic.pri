HEADERS += \
	$$PWD/cellgroup.h \
	$$PWD/cellshape.h \
	$$PWD/colourdifference.h \
	$$PWD/cpuphotomosaicgenerator.h \
	$$PWD/gridbounds.h \
	$$PWD/gridgenerator.h \
	$$PWD/gridutility.h \
	$$PWD/imagehistogramcompare.h \
	$$PWD/imagelibrary.h \
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
	$$PWD/imagelibrary.cpp \
	$$PWD/imageutility.cpp \
	$$PWD/photomosaicgeneratorbase.cpp \
	$$PWD/quadtree.cpp

CUDA {
	CUDA_SOURCES += \
		$$PWD/photomosaicgenerator.cu \
		$$PWD/reduction.cu

	SOURCES += \
		$$PWD/cudaphotomosaicdata.cpp \
		$$PWD/cudaphotomosaicgenerator.cpp

	HEADERS += \
		$$PWD/cudaphotomosaicdata.h \
		$$PWD/cudaphotomosaicgenerator.h \
		$$PWD/photomosaicgenerator.cuh

	DISTFILES += \
		$$PWD/reduction.cuh \
		$$PWD/photomosaicgenerator.cuh
}

INCLUDEPATH += $$PWD
