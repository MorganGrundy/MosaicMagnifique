/********************************************************************************
** Form generated from reading UI file 'GridEditor.ui'
**
** Created by: Qt User Interface Compiler version 5.15.0
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef UI_GRIDEDITOR_H
#define UI_GRIDEDITOR_H

#include <QtCore/QVariant>
#include <QtWidgets/QApplication>
#include <QtWidgets/QButtonGroup>
#include <QtWidgets/QHBoxLayout>
#include <QtWidgets/QLabel>
#include <QtWidgets/QMainWindow>
#include <QtWidgets/QMenuBar>
#include <QtWidgets/QSpacerItem>
#include <QtWidgets/QSpinBox>
#include <QtWidgets/QStatusBar>
#include <QtWidgets/QToolButton>
#include <QtWidgets/QVBoxLayout>
#include <QtWidgets/QWidget>
#include "grideditviewer.h"

QT_BEGIN_NAMESPACE

class Ui_GridEditor
{
public:
    QWidget *widget;
    QVBoxLayout *verticalLayout;
    QHBoxLayout *horizontalLayoutTools;
    QHBoxLayout *horizontalLayout;
    QToolButton *toolSingle;
    QToolButton *toolSelection;
    QSpacerItem *horizontalSpacer;
    QHBoxLayout *horizontalLayoutSizeStep;
    QLabel *labelSizeStep;
    QSpinBox *spinSizeStep;
    QSpacerItem *horizontalSpacer_2;
    GridEditViewer *gridEditViewer;
    QMenuBar *menubar;
    QStatusBar *statusbar;
    QButtonGroup *buttonGroupTools;

    void setupUi(QMainWindow *GridEditor)
    {
        if (GridEditor->objectName().isEmpty())
            GridEditor->setObjectName(QString::fromUtf8("GridEditor"));
        GridEditor->resize(1080, 720);
        GridEditor->setStyleSheet(QString::fromUtf8("QWidget {\n"
"background-color: rgb(60, 60, 60);\n"
"color: rgb(255, 255, 255);\n"
"border-color: rgb(0, 0, 0);\n"
"}\n"
"\n"
"QToolButton::checked {\n"
"	background-color: rgb(30, 30, 30);\n"
"}"));
        widget = new QWidget(GridEditor);
        widget->setObjectName(QString::fromUtf8("widget"));
        verticalLayout = new QVBoxLayout(widget);
        verticalLayout->setObjectName(QString::fromUtf8("verticalLayout"));
        horizontalLayoutTools = new QHBoxLayout();
        horizontalLayoutTools->setObjectName(QString::fromUtf8("horizontalLayoutTools"));
        horizontalLayoutTools->setContentsMargins(-1, 0, -1, -1);
        horizontalLayout = new QHBoxLayout();
        horizontalLayout->setObjectName(QString::fromUtf8("horizontalLayout"));
        horizontalLayout->setContentsMargins(0, -1, -1, -1);
        toolSingle = new QToolButton(widget);
        buttonGroupTools = new QButtonGroup(GridEditor);
        buttonGroupTools->setObjectName(QString::fromUtf8("buttonGroupTools"));
        buttonGroupTools->addButton(toolSingle);
        toolSingle->setObjectName(QString::fromUtf8("toolSingle"));
        QSizePolicy sizePolicy(QSizePolicy::Minimum, QSizePolicy::Minimum);
        sizePolicy.setHorizontalStretch(0);
        sizePolicy.setVerticalStretch(0);
        sizePolicy.setHeightForWidth(toolSingle->sizePolicy().hasHeightForWidth());
        toolSingle->setSizePolicy(sizePolicy);
        toolSingle->setMinimumSize(QSize(60, 40));
        toolSingle->setCursor(QCursor(Qt::CrossCursor));
        toolSingle->setStyleSheet(QString::fromUtf8(""));
        toolSingle->setCheckable(true);
        toolSingle->setChecked(true);
        toolSingle->setToolButtonStyle(Qt::ToolButtonTextUnderIcon);

        horizontalLayout->addWidget(toolSingle);

        toolSelection = new QToolButton(widget);
        buttonGroupTools->addButton(toolSelection);
        toolSelection->setObjectName(QString::fromUtf8("toolSelection"));
        sizePolicy.setHeightForWidth(toolSelection->sizePolicy().hasHeightForWidth());
        toolSelection->setSizePolicy(sizePolicy);
        toolSelection->setMinimumSize(QSize(60, 40));
        toolSelection->setCursor(QCursor(Qt::SizeBDiagCursor));
        toolSelection->setCheckable(true);
        toolSelection->setChecked(false);
        toolSelection->setToolButtonStyle(Qt::ToolButtonTextUnderIcon);

        horizontalLayout->addWidget(toolSelection);


        horizontalLayoutTools->addLayout(horizontalLayout);

        horizontalSpacer = new QSpacerItem(40, 20, QSizePolicy::Expanding, QSizePolicy::Minimum);

        horizontalLayoutTools->addItem(horizontalSpacer);

        horizontalLayoutSizeStep = new QHBoxLayout();
        horizontalLayoutSizeStep->setObjectName(QString::fromUtf8("horizontalLayoutSizeStep"));
        horizontalLayoutSizeStep->setContentsMargins(0, -1, -1, -1);
        labelSizeStep = new QLabel(widget);
        labelSizeStep->setObjectName(QString::fromUtf8("labelSizeStep"));

        horizontalLayoutSizeStep->addWidget(labelSizeStep);

        spinSizeStep = new QSpinBox(widget);
        spinSizeStep->setObjectName(QString::fromUtf8("spinSizeStep"));
        spinSizeStep->setStyleSheet(QString::fromUtf8("QSpinBox {\n"
"border: 1px solid dimgray;\n"
"}"));
        spinSizeStep->setButtonSymbols(QAbstractSpinBox::PlusMinus);

        horizontalLayoutSizeStep->addWidget(spinSizeStep);


        horizontalLayoutTools->addLayout(horizontalLayoutSizeStep);

        horizontalSpacer_2 = new QSpacerItem(40, 20, QSizePolicy::Expanding, QSizePolicy::Minimum);

        horizontalLayoutTools->addItem(horizontalSpacer_2);


        verticalLayout->addLayout(horizontalLayoutTools);

        gridEditViewer = new GridEditViewer(widget);
        gridEditViewer->setObjectName(QString::fromUtf8("gridEditViewer"));
        QSizePolicy sizePolicy1(QSizePolicy::Expanding, QSizePolicy::Expanding);
        sizePolicy1.setHorizontalStretch(0);
        sizePolicy1.setVerticalStretch(0);
        sizePolicy1.setHeightForWidth(gridEditViewer->sizePolicy().hasHeightForWidth());
        gridEditViewer->setSizePolicy(sizePolicy1);

        verticalLayout->addWidget(gridEditViewer);

        GridEditor->setCentralWidget(widget);
        menubar = new QMenuBar(GridEditor);
        menubar->setObjectName(QString::fromUtf8("menubar"));
        menubar->setGeometry(QRect(0, 0, 1080, 20));
        GridEditor->setMenuBar(menubar);
        statusbar = new QStatusBar(GridEditor);
        statusbar->setObjectName(QString::fromUtf8("statusbar"));
        GridEditor->setStatusBar(statusbar);

        retranslateUi(GridEditor);

        QMetaObject::connectSlotsByName(GridEditor);
    } // setupUi

    void retranslateUi(QMainWindow *GridEditor)
    {
        GridEditor->setWindowTitle(QCoreApplication::translate("GridEditor", "Grid Editor", nullptr));
#if QT_CONFIG(statustip)
        toolSingle->setStatusTip(QCoreApplication::translate("GridEditor", "Edit a single cell at a time", nullptr));
#endif // QT_CONFIG(statustip)
#if QT_CONFIG(whatsthis)
        toolSingle->setWhatsThis(QCoreApplication::translate("GridEditor", "<html><head/><body><p>Tool: Single</p><p>Used to toggle the state of a single cell at a time.</p></body></html>", nullptr));
#endif // QT_CONFIG(whatsthis)
        toolSingle->setText(QCoreApplication::translate("GridEditor", "Single", nullptr));
#if QT_CONFIG(statustip)
        toolSelection->setStatusTip(QCoreApplication::translate("GridEditor", "Edit all cells in selection", nullptr));
#endif // QT_CONFIG(statustip)
#if QT_CONFIG(whatsthis)
        toolSelection->setWhatsThis(QCoreApplication::translate("GridEditor", "<html><head/><body><p>Tool: Selection</p><p>Used to toggle the state of all cells in a selection area.</p></body></html>", nullptr));
#endif // QT_CONFIG(whatsthis)
        toolSelection->setText(QCoreApplication::translate("GridEditor", "Selection", nullptr));
#if QT_CONFIG(statustip)
        labelSizeStep->setStatusTip(QCoreApplication::translate("GridEditor", "Step size of grid to edit", nullptr));
#endif // QT_CONFIG(statustip)
        labelSizeStep->setText(QCoreApplication::translate("GridEditor", "Size step:", nullptr));
#if QT_CONFIG(statustip)
        spinSizeStep->setStatusTip(QCoreApplication::translate("GridEditor", "Step size of grid to edit", nullptr));
#endif // QT_CONFIG(statustip)
#if QT_CONFIG(whatsthis)
        spinSizeStep->setWhatsThis(QCoreApplication::translate("GridEditor", "<html><head/><body><p>Can only interact with a single step size of the grid at a time.</p><p>Use this to control which step size is interactable.</p></body></html>", nullptr));
#endif // QT_CONFIG(whatsthis)
        spinSizeStep->setSuffix(QString());
        spinSizeStep->setPrefix(QString());
#if QT_CONFIG(statustip)
        gridEditViewer->setStatusTip(QCoreApplication::translate("GridEditor", "View and edit grid state", nullptr));
#endif // QT_CONFIG(statustip)
    } // retranslateUi

};

namespace Ui {
    class GridEditor: public Ui_GridEditor {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_GRIDEDITOR_H
