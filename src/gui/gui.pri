win32:RC_ICONS += $$PWD/MosaicMagnifique.ico

#Application version
VERSION_MAJOR = 3
VERSION_MINOR = 2
VERSION_BUILD = 14

DEFINES += "VERSION_MAJOR=$$VERSION_MAJOR" \
	"VERSION_MINOR=$$VERSION_MINOR" \
	"VERSION_BUILD=$$VERSION_BUILD"

#Target version
VERSION = $${VERSION_MAJOR}.$${VERSION_MINOR}.$${VERSION_BUILD}

FORMS += \
	$$PWD/cellshapeeditor.ui \
	$$PWD/colourvisualisation.ui \
	$$PWD/grideditor.ui \
	$$PWD/imagelibraryeditor.ui \
	$$PWD/imagesquarer.ui \
	$$PWD/mainwindow.ui \
	$$PWD/photomosaicviewer.ui

HEADERS += \
	$$PWD/cellshapeeditor.h \
	$$PWD/colourvisualisation.h \
	$$PWD/cropgraphicsobject.h \
	$$PWD/customgraphicsview.h \
	$$PWD/grideditor.h \
	$$PWD/grideditviewer.h \
	$$PWD/gridviewer.h \
	$$PWD/imagelibraryeditor.h \
	$$PWD/imagesquarer.h \
	$$PWD/mainwindow.h \
	$$PWD/photomosaicviewer.h \
	$$PWD/switch.h

SOURCES += \
	$$PWD/cellshapeeditor.cpp \
	$$PWD/colourvisualisation.cpp \
	$$PWD/cropgraphicsobject.cpp \
	$$PWD/customgraphicsview.cpp \
	$$PWD/grideditor.cpp \
	$$PWD/grideditviewer.cpp \
	$$PWD/gridviewer.cpp \
	$$PWD/imagelibraryeditor.cpp \
	$$PWD/imagesquarer.cpp \
	$$PWD/main.cpp \
	$$PWD/mainwindow.cpp \
	$$PWD/photomosaicviewer.cpp \
	$$PWD/switch.cpp

RESOURCES += \
	$$PWD/ui.qrc

INCLUDEPATH += $$PWD
