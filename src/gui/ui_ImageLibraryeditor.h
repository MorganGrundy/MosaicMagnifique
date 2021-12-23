/********************************************************************************
** Form generated from reading UI file 'ImageLibraryeditor.ui'
**
** Created by: Qt User Interface Compiler version 5.15.0
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef UI_IMAGELIBRARYEDITOR_H
#define UI_IMAGELIBRARYEDITOR_H

#include <QtCore/QVariant>
#include <QtWidgets/QApplication>
#include <QtWidgets/QComboBox>
#include <QtWidgets/QFrame>
#include <QtWidgets/QHBoxLayout>
#include <QtWidgets/QLabel>
#include <QtWidgets/QListWidget>
#include <QtWidgets/QPushButton>
#include <QtWidgets/QSpacerItem>
#include <QtWidgets/QSpinBox>
#include <QtWidgets/QVBoxLayout>
#include <QtWidgets/QWidget>

QT_BEGIN_NAMESPACE

class Ui_ImageLibraryEditor
{
public:
    QVBoxLayout *verticalLayout_3;
    QHBoxLayout *horizontalLayout;
    QPushButton *buttonSave;
    QPushButton *buttonLoad;
    QSpacerItem *horizontalSpacer_2;
    QComboBox *comboCropMode;
    QPushButton *buttonAdd;
    QPushButton *buttonDelete;
    QPushButton *buttonClear;
    QSpacerItem *horizontalSpacer;
    QLabel *label;
    QSpinBox *spinLibCellSize;
    QPushButton *buttonLibCellSize;
    QFrame *line;
    QListWidget *listPhoto;

    void setupUi(QWidget *ImageLibraryEditor)
    {
        if (ImageLibraryEditor->objectName().isEmpty())
            ImageLibraryEditor->setObjectName(QString::fromUtf8("ImageLibraryEditor"));
        ImageLibraryEditor->resize(1080, 720);
        QSizePolicy sizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
        sizePolicy.setHorizontalStretch(0);
        sizePolicy.setVerticalStretch(0);
        sizePolicy.setHeightForWidth(ImageLibraryEditor->sizePolicy().hasHeightForWidth());
        ImageLibraryEditor->setSizePolicy(sizePolicy);
        ImageLibraryEditor->setStyleSheet(QString::fromUtf8("QWidget {\n"
"background-color: rgb(60, 60, 60);\n"
"color: rgb(255, 255, 255);\n"
"border-color: rgb(0, 0, 0);\n"
"}"));
        verticalLayout_3 = new QVBoxLayout(ImageLibraryEditor);
        verticalLayout_3->setObjectName(QString::fromUtf8("verticalLayout_3"));
        horizontalLayout = new QHBoxLayout();
        horizontalLayout->setObjectName(QString::fromUtf8("horizontalLayout"));
        buttonSave = new QPushButton(ImageLibraryEditor);
        buttonSave->setObjectName(QString::fromUtf8("buttonSave"));
        buttonSave->setMinimumSize(QSize(0, 24));
        buttonSave->setToolTipDuration(-1);

        horizontalLayout->addWidget(buttonSave);

        buttonLoad = new QPushButton(ImageLibraryEditor);
        buttonLoad->setObjectName(QString::fromUtf8("buttonLoad"));
        buttonLoad->setMinimumSize(QSize(0, 24));

        horizontalLayout->addWidget(buttonLoad);

        horizontalSpacer_2 = new QSpacerItem(40, 20, QSizePolicy::Expanding, QSizePolicy::Minimum);

        horizontalLayout->addItem(horizontalSpacer_2);

        comboCropMode = new QComboBox(ImageLibraryEditor);
        comboCropMode->addItem(QString());
        comboCropMode->addItem(QString());
        comboCropMode->addItem(QString());
        comboCropMode->addItem(QString());
        comboCropMode->addItem(QString());
        comboCropMode->setObjectName(QString::fromUtf8("comboCropMode"));
        QSizePolicy sizePolicy1(QSizePolicy::Minimum, QSizePolicy::Fixed);
        sizePolicy1.setHorizontalStretch(0);
        sizePolicy1.setVerticalStretch(0);
        sizePolicy1.setHeightForWidth(comboCropMode->sizePolicy().hasHeightForWidth());
        comboCropMode->setSizePolicy(sizePolicy1);
        comboCropMode->setMinimumSize(QSize(0, 24));
        comboCropMode->setStyleSheet(QString::fromUtf8(""));

        horizontalLayout->addWidget(comboCropMode, 0, Qt::AlignHCenter|Qt::AlignVCenter);

        buttonAdd = new QPushButton(ImageLibraryEditor);
        buttonAdd->setObjectName(QString::fromUtf8("buttonAdd"));
        sizePolicy1.setHeightForWidth(buttonAdd->sizePolicy().hasHeightForWidth());
        buttonAdd->setSizePolicy(sizePolicy1);
        buttonAdd->setMinimumSize(QSize(0, 24));

        horizontalLayout->addWidget(buttonAdd);

        buttonDelete = new QPushButton(ImageLibraryEditor);
        buttonDelete->setObjectName(QString::fromUtf8("buttonDelete"));
        buttonDelete->setMinimumSize(QSize(0, 24));

        horizontalLayout->addWidget(buttonDelete);

        buttonClear = new QPushButton(ImageLibraryEditor);
        buttonClear->setObjectName(QString::fromUtf8("buttonClear"));

        horizontalLayout->addWidget(buttonClear);

        horizontalSpacer = new QSpacerItem(40, 20, QSizePolicy::Expanding, QSizePolicy::Minimum);

        horizontalLayout->addItem(horizontalSpacer);

        label = new QLabel(ImageLibraryEditor);
        label->setObjectName(QString::fromUtf8("label"));

        horizontalLayout->addWidget(label, 0, Qt::AlignVCenter);

        spinLibCellSize = new QSpinBox(ImageLibraryEditor);
        spinLibCellSize->setObjectName(QString::fromUtf8("spinLibCellSize"));
        spinLibCellSize->setMinimumSize(QSize(0, 24));
        spinLibCellSize->setStyleSheet(QString::fromUtf8("QSpinBox {\n"
"border: 1px solid dimgray;\n"
"}"));
        spinLibCellSize->setButtonSymbols(QAbstractSpinBox::PlusMinus);
        spinLibCellSize->setMinimum(10);
        spinLibCellSize->setMaximum(10000);
        spinLibCellSize->setValue(128);

        horizontalLayout->addWidget(spinLibCellSize);

        buttonLibCellSize = new QPushButton(ImageLibraryEditor);
        buttonLibCellSize->setObjectName(QString::fromUtf8("buttonLibCellSize"));
        sizePolicy1.setHeightForWidth(buttonLibCellSize->sizePolicy().hasHeightForWidth());
        buttonLibCellSize->setSizePolicy(sizePolicy1);
        buttonLibCellSize->setMinimumSize(QSize(0, 24));
        buttonLibCellSize->setIconSize(QSize(16, 16));

        horizontalLayout->addWidget(buttonLibCellSize);


        verticalLayout_3->addLayout(horizontalLayout);

        line = new QFrame(ImageLibraryEditor);
        line->setObjectName(QString::fromUtf8("line"));
        line->setFrameShape(QFrame::HLine);
        line->setFrameShadow(QFrame::Sunken);

        verticalLayout_3->addWidget(line);

        listPhoto = new QListWidget(ImageLibraryEditor);
        listPhoto->setObjectName(QString::fromUtf8("listPhoto"));
        listPhoto->setStyleSheet(QString::fromUtf8("QListWidget::item {\n"
"border: 1px solid transparent;\n"
"padding: 1px;\n"
"margin: 5px;\n"
"}\n"
"\n"
"QListWidget::item:selected {\n"
"border: 1px solid blue;\n"
"}\n"
"\n"
"QListWidget {\n"
"border: 1px solid dimgray;\n"
"}"));
        listPhoto->setFrameShadow(QFrame::Sunken);
        listPhoto->setLineWidth(1);
        listPhoto->setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
        listPhoto->setSizeAdjustPolicy(QAbstractScrollArea::AdjustIgnored);
        listPhoto->setAlternatingRowColors(false);
        listPhoto->setSelectionMode(QAbstractItemView::ExtendedSelection);
        listPhoto->setIconSize(QSize(100, 100));
        listPhoto->setTextElideMode(Qt::ElideNone);
        listPhoto->setVerticalScrollMode(QAbstractItemView::ScrollPerPixel);
        listPhoto->setMovement(QListView::Static);
        listPhoto->setFlow(QListView::LeftToRight);
        listPhoto->setProperty("isWrapping", QVariant(true));
        listPhoto->setResizeMode(QListView::Fixed);
        listPhoto->setLayoutMode(QListView::SinglePass);
        listPhoto->setSpacing(0);
        listPhoto->setGridSize(QSize(100, 150));
        listPhoto->setViewMode(QListView::IconMode);
        listPhoto->setModelColumn(0);
        listPhoto->setUniformItemSizes(false);
        listPhoto->setBatchSize(5);
        listPhoto->setWordWrap(true);
        listPhoto->setSelectionRectVisible(true);

        verticalLayout_3->addWidget(listPhoto);


        retranslateUi(ImageLibraryEditor);

        comboCropMode->setCurrentIndex(1);
        listPhoto->setCurrentRow(-1);


        QMetaObject::connectSlotsByName(ImageLibraryEditor);
    } // setupUi

    void retranslateUi(QWidget *ImageLibraryEditor)
    {
#if QT_CONFIG(tooltip)
        buttonSave->setToolTip(QString());
#endif // QT_CONFIG(tooltip)
#if QT_CONFIG(statustip)
        buttonSave->setStatusTip(QCoreApplication::translate("ImageLibraryEditor", "Save Image Library to .mil", nullptr));
#endif // QT_CONFIG(statustip)
#if QT_CONFIG(whatsthis)
        buttonSave->setWhatsThis(QCoreApplication::translate("ImageLibraryEditor", "<html><head/><body><p>Opens a file dialog to allow user to save the current image library as a .mil file.</p></body></html>", nullptr));
#endif // QT_CONFIG(whatsthis)
        buttonSave->setText(QCoreApplication::translate("ImageLibraryEditor", "Save Library", nullptr));
#if QT_CONFIG(statustip)
        buttonLoad->setStatusTip(QCoreApplication::translate("ImageLibraryEditor", "Load an Image Library from .mil", nullptr));
#endif // QT_CONFIG(statustip)
#if QT_CONFIG(whatsthis)
        buttonLoad->setWhatsThis(QCoreApplication::translate("ImageLibraryEditor", "<html><head/><body><p>Opens file dialog to select a .mil file to load.</p></body></html>", nullptr));
#endif // QT_CONFIG(whatsthis)
        buttonLoad->setText(QCoreApplication::translate("ImageLibraryEditor", "Load Library", nullptr));
        comboCropMode->setItemText(0, QCoreApplication::translate("ImageLibraryEditor", "Manual", nullptr));
        comboCropMode->setItemText(1, QCoreApplication::translate("ImageLibraryEditor", "Center", nullptr));
        comboCropMode->setItemText(2, QCoreApplication::translate("ImageLibraryEditor", "Features", nullptr));
        comboCropMode->setItemText(3, QCoreApplication::translate("ImageLibraryEditor", "Entropy", nullptr));
        comboCropMode->setItemText(4, QCoreApplication::translate("ImageLibraryEditor", "Cascade Classifier", nullptr));

#if QT_CONFIG(statustip)
        comboCropMode->setStatusTip(QCoreApplication::translate("ImageLibraryEditor", "Controls how new images are cropped", nullptr));
#endif // QT_CONFIG(statustip)
#if QT_CONFIG(whatsthis)
        comboCropMode->setWhatsThis(QCoreApplication::translate("ImageLibraryEditor", "<html><head/><body><p>When new Library Images are added they are cropped to be square. This controls how the image is cropped.</p><p>Modes: </p><ul style=\"margin-top: 0px; margin-bottom: 0px; margin-left: 0px; margin-right: 0px; -qt-list-indent: 1;\"><li style=\" margin-top:12px; margin-bottom:12px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">Manual; Manually set how the image is cropped.</li></ul><ul style=\"margin-top: 0px; margin-bottom: 0px; margin-left: 0px; margin-right: 0px; -qt-list-indent: 1;\"><li style=\" margin-top:12px; margin-bottom:12px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">Center; Crops around the center of the image.</li><li style=\" margin-top:12px; margin-bottom:12px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">Features; Detects corners in image and crops such that maximum number of corners visible.</li><li style=\" margin-top:12px; margin-bottom:12px; margin-left:0px; margin-right:0px; -qt-bloc"
                        "k-indent:0; text-indent:0px;\">Entropy; Crops image such that entropy is maximised (entropy is higher when more colours are used).</li><li style=\" margin-top:12px; margin-bottom:12px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">Cascade Classifier; Allows user to load a cascade classifier .xml file for object detection. Crops image such that maximum number of objects visible and closest to crop center.</li></ul></body></html>", nullptr));
#endif // QT_CONFIG(whatsthis)
#if QT_CONFIG(statustip)
        buttonAdd->setStatusTip(QCoreApplication::translate("ImageLibraryEditor", "Adds images to Image Library", nullptr));
#endif // QT_CONFIG(statustip)
#if QT_CONFIG(whatsthis)
        buttonAdd->setWhatsThis(QCoreApplication::translate("ImageLibraryEditor", "<html><head/><body><p>Opens a file dialog allowing user to select one or multiple images to add to current image library.</p></body></html>", nullptr));
#endif // QT_CONFIG(whatsthis)
        buttonAdd->setText(QCoreApplication::translate("ImageLibraryEditor", "Add Images", nullptr));
#if QT_CONFIG(statustip)
        buttonDelete->setStatusTip(QCoreApplication::translate("ImageLibraryEditor", "Deletes selected images from library", nullptr));
#endif // QT_CONFIG(statustip)
#if QT_CONFIG(whatsthis)
        buttonDelete->setWhatsThis(QCoreApplication::translate("ImageLibraryEditor", "<html><head/><body><p>Deletes all of the selected images from the image library.</p></body></html>", nullptr));
#endif // QT_CONFIG(whatsthis)
        buttonDelete->setText(QCoreApplication::translate("ImageLibraryEditor", "Delete Images", nullptr));
        buttonClear->setText(QCoreApplication::translate("ImageLibraryEditor", "Clear", nullptr));
#if QT_CONFIG(statustip)
        label->setStatusTip(QCoreApplication::translate("ImageLibraryEditor", "Width and height of each image in pixels", nullptr));
#endif // QT_CONFIG(statustip)
        label->setText(QCoreApplication::translate("ImageLibraryEditor", "Cell size:", nullptr));
#if QT_CONFIG(statustip)
        spinLibCellSize->setStatusTip(QCoreApplication::translate("ImageLibraryEditor", "Width and height of each image in pixels", nullptr));
#endif // QT_CONFIG(statustip)
#if QT_CONFIG(whatsthis)
        spinLibCellSize->setWhatsThis(QCoreApplication::translate("ImageLibraryEditor", "<html><head/><body><p>Controls width and height of library images in pixels.</p><p>All library images are cropped around their centre to ensure they are square.</p><p>An ideal value for library image size would be the maximum cell size to be used in Photomosaic generation.</p></body></html>", nullptr));
#endif // QT_CONFIG(whatsthis)
        spinLibCellSize->setSpecialValueText(QString());
        spinLibCellSize->setSuffix(QCoreApplication::translate("ImageLibraryEditor", "px", nullptr));
#if QT_CONFIG(statustip)
        buttonLibCellSize->setStatusTip(QCoreApplication::translate("ImageLibraryEditor", "Applies cell size to Library Images", nullptr));
#endif // QT_CONFIG(statustip)
        buttonLibCellSize->setText(QCoreApplication::translate("ImageLibraryEditor", "Apply", nullptr));
#if QT_CONFIG(statustip)
        listPhoto->setStatusTip(QCoreApplication::translate("ImageLibraryEditor", "Displays images in library", nullptr));
#endif // QT_CONFIG(statustip)
#if QT_CONFIG(whatsthis)
        listPhoto->setWhatsThis(QCoreApplication::translate("ImageLibraryEditor", "<html><head/><body><p>Displays a grid of all the images in the library.</p><p>Ctrl clicking adds to selection. Shift clicking adds a range to selection.</p></body></html>", nullptr));
#endif // QT_CONFIG(whatsthis)
        (void)ImageLibraryEditor;
    } // retranslateUi

};

namespace Ui {
    class ImageLibraryEditor: public Ui_ImageLibraryEditor {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_IMAGELIBRARYEDITOR_H
