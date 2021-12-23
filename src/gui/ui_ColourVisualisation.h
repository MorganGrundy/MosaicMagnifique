/********************************************************************************
** Form generated from reading UI file 'ColourVisualisation.ui'
**
** Created by: Qt User Interface Compiler version 5.15.0
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef UI_COLOURVISUALISATION_H
#define UI_COLOURVISUALISATION_H

#include <QtCore/QVariant>
#include <QtWidgets/QApplication>
#include <QtWidgets/QLabel>
#include <QtWidgets/QListWidget>
#include <QtWidgets/QMainWindow>
#include <QtWidgets/QMenuBar>
#include <QtWidgets/QStatusBar>
#include <QtWidgets/QVBoxLayout>
#include <QtWidgets/QWidget>

QT_BEGIN_NAMESPACE

class Ui_ColourVisualisation
{
public:
    QWidget *centralwidget;
    QVBoxLayout *verticalLayout;
    QLabel *label;
    QListWidget *listWidget;
    QMenuBar *menubar;
    QStatusBar *statusbar;

    void setupUi(QMainWindow *ColourVisualisation)
    {
        if (ColourVisualisation->objectName().isEmpty())
            ColourVisualisation->setObjectName(QString::fromUtf8("ColourVisualisation"));
        ColourVisualisation->resize(800, 600);
        ColourVisualisation->setStyleSheet(QString::fromUtf8("QWidget {\n"
"background-color: rgb(60, 60, 60);\n"
"color: rgb(255, 255, 255);\n"
"border-color: rgb(0, 0, 0);\n"
"}"));
        centralwidget = new QWidget(ColourVisualisation);
        centralwidget->setObjectName(QString::fromUtf8("centralwidget"));
        verticalLayout = new QVBoxLayout(centralwidget);
        verticalLayout->setObjectName(QString::fromUtf8("verticalLayout"));
        label = new QLabel(centralwidget);
        label->setObjectName(QString::fromUtf8("label"));
        label->setAlignment(Qt::AlignCenter);

        verticalLayout->addWidget(label);

        listWidget = new QListWidget(centralwidget);
        listWidget->setObjectName(QString::fromUtf8("listWidget"));
        listWidget->setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
        listWidget->setEditTriggers(QAbstractItemView::NoEditTriggers);
        listWidget->setProperty("showDropIndicator", QVariant(false));
        listWidget->setDragDropMode(QAbstractItemView::DragDrop);
        listWidget->setDefaultDropAction(Qt::IgnoreAction);
        listWidget->setAlternatingRowColors(false);
        listWidget->setHorizontalScrollMode(QAbstractItemView::ScrollPerItem);
        listWidget->setFlow(QListView::LeftToRight);
        listWidget->setProperty("isWrapping", QVariant(true));
        listWidget->setResizeMode(QListView::Adjust);
        listWidget->setLayoutMode(QListView::Batched);
        listWidget->setViewMode(QListView::IconMode);
        listWidget->setUniformItemSizes(true);
        listWidget->setBatchSize(5);
        listWidget->setSelectionRectVisible(false);

        verticalLayout->addWidget(listWidget);

        ColourVisualisation->setCentralWidget(centralwidget);
        menubar = new QMenuBar(ColourVisualisation);
        menubar->setObjectName(QString::fromUtf8("menubar"));
        menubar->setGeometry(QRect(0, 0, 800, 20));
        ColourVisualisation->setMenuBar(menubar);
        statusbar = new QStatusBar(ColourVisualisation);
        statusbar->setObjectName(QString::fromUtf8("statusbar"));
        ColourVisualisation->setStatusBar(statusbar);

        retranslateUi(ColourVisualisation);

        QMetaObject::connectSlotsByName(ColourVisualisation);
    } // setupUi

    void retranslateUi(QMainWindow *ColourVisualisation)
    {
        ColourVisualisation->setWindowTitle(QCoreApplication::translate("ColourVisualisation", "Colour Visualisation", nullptr));
        label->setText(QCoreApplication::translate("ColourVisualisation", "Add images to library with colours similar to the following to improve Photomosaic (ordered by descending priority):", nullptr));
    } // retranslateUi

};

namespace Ui {
    class ColourVisualisation: public Ui_ColourVisualisation {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_COLOURVISUALISATION_H
