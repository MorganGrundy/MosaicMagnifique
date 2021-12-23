/********************************************************************************
** Form generated from reading UI file 'PhotomosaicViewer.ui'
**
** Created by: Qt User Interface Compiler version 5.15.0
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef UI_PHOTOMOSAICVIEWER_H
#define UI_PHOTOMOSAICVIEWER_H

#include <QtCore/QVariant>
#include <QtWidgets/QApplication>
#include <QtWidgets/QGridLayout>
#include <QtWidgets/QGroupBox>
#include <QtWidgets/QHBoxLayout>
#include <QtWidgets/QLabel>
#include <QtWidgets/QMainWindow>
#include <QtWidgets/QMenuBar>
#include <QtWidgets/QPushButton>
#include <QtWidgets/QSpacerItem>
#include <QtWidgets/QStatusBar>
#include <QtWidgets/QVBoxLayout>
#include <QtWidgets/QWidget>
#include <customgraphicsview.h>

QT_BEGIN_NAMESPACE

class Ui_PhotomosaicViewer
{
public:
    QWidget *centralwidget;
    QGridLayout *gridLayout;
    QVBoxLayout *verticalLayoutOptions;
    QGroupBox *groupBoxBackground;
    QVBoxLayout *verticalLayout;
    QHBoxLayout *horizontalLayoutBackgroundColour;
    QLabel *labelBackgroundColour;
    QPushButton *pushBackgroundColour;
    QSpacerItem *verticalSpacer;
    CustomGraphicsView *graphicsView;
    QHBoxLayout *horizontalLayoutTop;
    QPushButton *saveButton;
    QSpacerItem *horizontalSpacer;
    QPushButton *fitButton;
    QMenuBar *menubar;
    QStatusBar *statusbar;

    void setupUi(QMainWindow *PhotomosaicViewer)
    {
        if (PhotomosaicViewer->objectName().isEmpty())
            PhotomosaicViewer->setObjectName(QString::fromUtf8("PhotomosaicViewer"));
        PhotomosaicViewer->resize(800, 600);
        PhotomosaicViewer->setStyleSheet(QString::fromUtf8("QWidget {\n"
"background-color: rgb(60, 60, 60);\n"
"color: rgb(255, 255, 255);\n"
"border-color: rgb(0, 0, 0);\n"
"}\n"
"\n"
"QGroupBox {\n"
"border: 1px solid black;\n"
"border-radius: 10px;\n"
"margin-top: 0.5em;\n"
"}\n"
"\n"
"QGroupBox::title {\n"
"subcontrol-origin: margin;\n"
"left: 10px;\n"
"padding: 0 3px 0 3px;\n"
"}"));
        centralwidget = new QWidget(PhotomosaicViewer);
        centralwidget->setObjectName(QString::fromUtf8("centralwidget"));
        gridLayout = new QGridLayout(centralwidget);
        gridLayout->setObjectName(QString::fromUtf8("gridLayout"));
        verticalLayoutOptions = new QVBoxLayout();
        verticalLayoutOptions->setObjectName(QString::fromUtf8("verticalLayoutOptions"));
        verticalLayoutOptions->setContentsMargins(0, 0, -1, -1);
        groupBoxBackground = new QGroupBox(centralwidget);
        groupBoxBackground->setObjectName(QString::fromUtf8("groupBoxBackground"));
        verticalLayout = new QVBoxLayout(groupBoxBackground);
        verticalLayout->setObjectName(QString::fromUtf8("verticalLayout"));
        horizontalLayoutBackgroundColour = new QHBoxLayout();
        horizontalLayoutBackgroundColour->setObjectName(QString::fromUtf8("horizontalLayoutBackgroundColour"));
        horizontalLayoutBackgroundColour->setContentsMargins(-1, 0, -1, -1);
        labelBackgroundColour = new QLabel(groupBoxBackground);
        labelBackgroundColour->setObjectName(QString::fromUtf8("labelBackgroundColour"));

        horizontalLayoutBackgroundColour->addWidget(labelBackgroundColour);

        pushBackgroundColour = new QPushButton(groupBoxBackground);
        pushBackgroundColour->setObjectName(QString::fromUtf8("pushBackgroundColour"));
        pushBackgroundColour->setStyleSheet(QString::fromUtf8(""));

        horizontalLayoutBackgroundColour->addWidget(pushBackgroundColour);


        verticalLayout->addLayout(horizontalLayoutBackgroundColour);


        verticalLayoutOptions->addWidget(groupBoxBackground);

        verticalSpacer = new QSpacerItem(20, 40, QSizePolicy::Minimum, QSizePolicy::Expanding);

        verticalLayoutOptions->addItem(verticalSpacer);


        gridLayout->addLayout(verticalLayoutOptions, 1, 0, 1, 1);

        graphicsView = new CustomGraphicsView(centralwidget);
        graphicsView->setObjectName(QString::fromUtf8("graphicsView"));
        graphicsView->setSceneRect(QRectF(0, 0, 0, 0));
        graphicsView->setAlignment(Qt::AlignCenter);
        graphicsView->setRenderHints(QPainter::HighQualityAntialiasing);
        graphicsView->setDragMode(QGraphicsView::ScrollHandDrag);
        graphicsView->setCacheMode(QGraphicsView::CacheBackground);
        graphicsView->setTransformationAnchor(QGraphicsView::AnchorUnderMouse);
        graphicsView->setViewportUpdateMode(QGraphicsView::BoundingRectViewportUpdate);

        gridLayout->addWidget(graphicsView, 1, 1, 1, 1);

        horizontalLayoutTop = new QHBoxLayout();
        horizontalLayoutTop->setObjectName(QString::fromUtf8("horizontalLayoutTop"));
        horizontalLayoutTop->setContentsMargins(-1, 0, -1, 0);
        saveButton = new QPushButton(centralwidget);
        saveButton->setObjectName(QString::fromUtf8("saveButton"));

        horizontalLayoutTop->addWidget(saveButton);

        horizontalSpacer = new QSpacerItem(40, 20, QSizePolicy::Expanding, QSizePolicy::Minimum);

        horizontalLayoutTop->addItem(horizontalSpacer);

        fitButton = new QPushButton(centralwidget);
        fitButton->setObjectName(QString::fromUtf8("fitButton"));

        horizontalLayoutTop->addWidget(fitButton);


        gridLayout->addLayout(horizontalLayoutTop, 0, 0, 1, 2);

        PhotomosaicViewer->setCentralWidget(centralwidget);
        menubar = new QMenuBar(PhotomosaicViewer);
        menubar->setObjectName(QString::fromUtf8("menubar"));
        menubar->setGeometry(QRect(0, 0, 800, 20));
        PhotomosaicViewer->setMenuBar(menubar);
        statusbar = new QStatusBar(PhotomosaicViewer);
        statusbar->setObjectName(QString::fromUtf8("statusbar"));
        PhotomosaicViewer->setStatusBar(statusbar);

        retranslateUi(PhotomosaicViewer);

        QMetaObject::connectSlotsByName(PhotomosaicViewer);
    } // setupUi

    void retranslateUi(QMainWindow *PhotomosaicViewer)
    {
        PhotomosaicViewer->setWindowTitle(QCoreApplication::translate("PhotomosaicViewer", "Photomosaic Viewer", nullptr));
        groupBoxBackground->setTitle(QCoreApplication::translate("PhotomosaicViewer", "Background:", nullptr));
        labelBackgroundColour->setText(QCoreApplication::translate("PhotomosaicViewer", "Colour:", nullptr));
        pushBackgroundColour->setText(QString());
#if QT_CONFIG(statustip)
        graphicsView->setStatusTip(QCoreApplication::translate("PhotomosaicViewer", "Displays Photomosaic", nullptr));
#endif // QT_CONFIG(statustip)
#if QT_CONFIG(statustip)
        saveButton->setStatusTip(QCoreApplication::translate("PhotomosaicViewer", "Save Photomosaic to an image file", nullptr));
#endif // QT_CONFIG(statustip)
#if QT_CONFIG(whatsthis)
        saveButton->setWhatsThis(QCoreApplication::translate("PhotomosaicViewer", "<html><head/><body><p>Opens a file dialog allowing user to save the Photomosaic as an image file.</p></body></html>", nullptr));
#endif // QT_CONFIG(whatsthis)
        saveButton->setText(QCoreApplication::translate("PhotomosaicViewer", "Save", nullptr));
#if QT_CONFIG(statustip)
        fitButton->setStatusTip(QCoreApplication::translate("PhotomosaicViewer", "Resizes view so that Photomosaic is fully visible", nullptr));
#endif // QT_CONFIG(statustip)
        fitButton->setText(QCoreApplication::translate("PhotomosaicViewer", "Fit To View", nullptr));
    } // retranslateUi

};

namespace Ui {
    class PhotomosaicViewer: public Ui_PhotomosaicViewer {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_PHOTOMOSAICVIEWER_H
