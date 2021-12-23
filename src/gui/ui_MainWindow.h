/********************************************************************************
** Form generated from reading UI file 'MainWindow.ui'
**
** Created by: Qt User Interface Compiler version 5.15.0
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef UI_MAINWINDOW_H
#define UI_MAINWINDOW_H

#include <QtCore/QVariant>
#include <QtGui/QIcon>
#include <QtWidgets/QAction>
#include <QtWidgets/QApplication>
#include <QtWidgets/QCheckBox>
#include <QtWidgets/QComboBox>
#include <QtWidgets/QFrame>
#include <QtWidgets/QGridLayout>
#include <QtWidgets/QGroupBox>
#include <QtWidgets/QHBoxLayout>
#include <QtWidgets/QLabel>
#include <QtWidgets/QLineEdit>
#include <QtWidgets/QMainWindow>
#include <QtWidgets/QMenu>
#include <QtWidgets/QMenuBar>
#include <QtWidgets/QPushButton>
#include <QtWidgets/QSpacerItem>
#include <QtWidgets/QSpinBox>
#include <QtWidgets/QStatusBar>
#include <QtWidgets/QTabWidget>
#include <QtWidgets/QVBoxLayout>
#include <QtWidgets/QWidget>
#include "cellshapeeditor.h"
#include "gridviewer.h"
#include "imagelibraryeditor.h"

QT_BEGIN_NAMESPACE

class Ui_MainWindow
{
public:
    QAction *actionWebsite;
    QAction *actionGithub;
    QAction *actionAbout;
    QWidget *centralwidget;
    QVBoxLayout *verticalLayout;
    QTabWidget *tabWidget;
    CellShapeEditor *cellShapeEditor;
    ImageLibraryEditor *imageLibraryEditor;
    QWidget *GeneratorSettings;
    QHBoxLayout *horizontalLayout_7;
    QVBoxLayout *verticalLayoutOptions;
    QGroupBox *groupGeneral;
    QVBoxLayout *verticalLayout_4;
    QFrame *frameMainImage;
    QHBoxLayout *horizontalLayout_2;
    QLabel *labelMainImage;
    QLineEdit *lineMainImage;
    QPushButton *buttonMainImage;
    QPushButton *buttonCompareColours;
    QFrame *line_2;
    QFrame *frameSizeDetail;
    QGridLayout *gridLayout;
    QLabel *labelWidth;
    QLabel *labelHeight;
    QSpinBox *spinPhotomosaicWidth;
    QSpacerItem *horizontalSpacer_3;
    QPushButton *buttonPhotomosaicSize;
    QSpinBox *spinPhotomosaicHeight;
    QLabel *labelSize;
    QSpinBox *spinDetail;
    QPushButton *buttonPhotomosaicSizeLink;
    QLabel *labelDetail;
    QSpacerItem *verticalSpacer;
    QGroupBox *groupColour;
    QVBoxLayout *verticalLayout_7;
    QFrame *frameMode_2;
    QGridLayout *gridLayout_5;
    QLabel *labelColourDifference;
    QSpacerItem *horizontalSpacer_8;
    QComboBox *comboColourDifference;
    QLabel *labelColourScheme;
    QComboBox *comboColourScheme;
    QSpacerItem *verticalSpacer_4;
    QGroupBox *groupCells;
    QVBoxLayout *verticalLayout_5;
    QFrame *frameCellSize;
    QHBoxLayout *horizontalLayout_4;
    QLabel *labelCellSize;
    QSpinBox *spinCellSize;
    QSpacerItem *horizontalSpacer_6;
    QFrame *frameCellShape;
    QHBoxLayout *horizontalLayout_3;
    QCheckBox *checkCellShape;
    QLineEdit *lineCellShape;
    QFrame *line_5;
    QHBoxLayout *horizontalLayoutMinCellSize;
    QLabel *labelSizeSteps;
    QSpinBox *spinSizeSteps;
    QSpacerItem *horizontalSpacer_10;
    QLabel *labelCellSizesList;
    QFrame *line_6;
    QPushButton *buttonEditGrid;
    QSpacerItem *verticalSpacer_2;
    QGroupBox *groupRepeats;
    QVBoxLayout *verticalLayout_6;
    QFrame *frame_5;
    QGridLayout *gridLayout_3;
    QSpacerItem *horizontalSpacer_4;
    QLabel *labelRepeatAddition;
    QLabel *labelRepeatRange;
    QSpinBox *spinRepeatRange;
    QSpinBox *spinRepeatAddition;
    QSpacerItem *verticalSpacer_3;
    QSpacerItem *verticalSpacer_5;
    QHBoxLayout *groupGenerate;
    QCheckBox *checkCUDA;
    QComboBox *comboCUDA;
    QSpacerItem *horizontalSpacer_9;
    QPushButton *buttonGenerate;
    GridViewer *widgetGridPreview;
    QMenuBar *menubar;
    QMenu *menuAbout;
    QStatusBar *statusbar;

    void setupUi(QMainWindow *MainWindow)
    {
        if (MainWindow->objectName().isEmpty())
            MainWindow->setObjectName(QString::fromUtf8("MainWindow"));
        MainWindow->resize(1080, 720);
        MainWindow->setStyleSheet(QString::fromUtf8("QWidget {\n"
"background-color: rgb(60, 60, 60);\n"
"color: rgb(255, 255, 255);\n"
"border-color: rgb(0, 0, 0);\n"
"}"));
        actionWebsite = new QAction(MainWindow);
        actionWebsite->setObjectName(QString::fromUtf8("actionWebsite"));
        actionGithub = new QAction(MainWindow);
        actionGithub->setObjectName(QString::fromUtf8("actionGithub"));
        actionAbout = new QAction(MainWindow);
        actionAbout->setObjectName(QString::fromUtf8("actionAbout"));
        centralwidget = new QWidget(MainWindow);
        centralwidget->setObjectName(QString::fromUtf8("centralwidget"));
        QSizePolicy sizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
        sizePolicy.setHorizontalStretch(0);
        sizePolicy.setVerticalStretch(0);
        sizePolicy.setHeightForWidth(centralwidget->sizePolicy().hasHeightForWidth());
        centralwidget->setSizePolicy(sizePolicy);
        verticalLayout = new QVBoxLayout(centralwidget);
        verticalLayout->setSpacing(0);
        verticalLayout->setObjectName(QString::fromUtf8("verticalLayout"));
        verticalLayout->setContentsMargins(0, 0, 0, 0);
        tabWidget = new QTabWidget(centralwidget);
        tabWidget->setObjectName(QString::fromUtf8("tabWidget"));
        tabWidget->setStyleSheet(QString::fromUtf8("QTabBar::tab {\n"
"	background-color: rgb(30, 30, 30);\n"
"}\n"
"\n"
"QTabBar::tab:selected {\n"
"	background-color: rgb(60, 60, 60);\n"
"}\n"
"\n"
"QTabBar::tab:hover {\n"
"	background-color: rgb(45, 45, 45);\n"
"}\n"
"\n"
"QTabWidget::pane {\n"
"border: 0px;\n"
"}"));
        tabWidget->setTabPosition(QTabWidget::North);
        tabWidget->setTabShape(QTabWidget::Rounded);
        tabWidget->setElideMode(Qt::ElideNone);
        cellShapeEditor = new CellShapeEditor();
        cellShapeEditor->setObjectName(QString::fromUtf8("cellShapeEditor"));
        tabWidget->addTab(cellShapeEditor, QString());
        imageLibraryEditor = new ImageLibraryEditor();
        imageLibraryEditor->setObjectName(QString::fromUtf8("imageLibraryEditor"));
        sizePolicy.setHeightForWidth(imageLibraryEditor->sizePolicy().hasHeightForWidth());
        imageLibraryEditor->setSizePolicy(sizePolicy);
        imageLibraryEditor->setStyleSheet(QString::fromUtf8(""));
        tabWidget->addTab(imageLibraryEditor, QString());
        GeneratorSettings = new QWidget();
        GeneratorSettings->setObjectName(QString::fromUtf8("GeneratorSettings"));
        sizePolicy.setHeightForWidth(GeneratorSettings->sizePolicy().hasHeightForWidth());
        GeneratorSettings->setSizePolicy(sizePolicy);
        GeneratorSettings->setStyleSheet(QString::fromUtf8("QGroupBox {\n"
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
        horizontalLayout_7 = new QHBoxLayout(GeneratorSettings);
        horizontalLayout_7->setObjectName(QString::fromUtf8("horizontalLayout_7"));
        verticalLayoutOptions = new QVBoxLayout();
        verticalLayoutOptions->setObjectName(QString::fromUtf8("verticalLayoutOptions"));
        verticalLayoutOptions->setContentsMargins(9, 9, 9, 9);
        groupGeneral = new QGroupBox(GeneratorSettings);
        groupGeneral->setObjectName(QString::fromUtf8("groupGeneral"));
        QSizePolicy sizePolicy1(QSizePolicy::Minimum, QSizePolicy::MinimumExpanding);
        sizePolicy1.setHorizontalStretch(0);
        sizePolicy1.setVerticalStretch(0);
        sizePolicy1.setHeightForWidth(groupGeneral->sizePolicy().hasHeightForWidth());
        groupGeneral->setSizePolicy(sizePolicy1);
        groupGeneral->setAlignment(Qt::AlignLeading|Qt::AlignLeft|Qt::AlignVCenter);
        groupGeneral->setFlat(false);
        verticalLayout_4 = new QVBoxLayout(groupGeneral);
        verticalLayout_4->setObjectName(QString::fromUtf8("verticalLayout_4"));
        verticalLayout_4->setContentsMargins(-1, 14, -1, -1);
        frameMainImage = new QFrame(groupGeneral);
        frameMainImage->setObjectName(QString::fromUtf8("frameMainImage"));
        QSizePolicy sizePolicy2(QSizePolicy::Minimum, QSizePolicy::Minimum);
        sizePolicy2.setHorizontalStretch(0);
        sizePolicy2.setVerticalStretch(0);
        sizePolicy2.setHeightForWidth(frameMainImage->sizePolicy().hasHeightForWidth());
        frameMainImage->setSizePolicy(sizePolicy2);
        frameMainImage->setStyleSheet(QString::fromUtf8(""));
        frameMainImage->setFrameShape(QFrame::StyledPanel);
        frameMainImage->setFrameShadow(QFrame::Raised);
        horizontalLayout_2 = new QHBoxLayout(frameMainImage);
        horizontalLayout_2->setObjectName(QString::fromUtf8("horizontalLayout_2"));
        horizontalLayout_2->setContentsMargins(0, 0, 0, 0);
        labelMainImage = new QLabel(frameMainImage);
        labelMainImage->setObjectName(QString::fromUtf8("labelMainImage"));
        QSizePolicy sizePolicy3(QSizePolicy::Fixed, QSizePolicy::Fixed);
        sizePolicy3.setHorizontalStretch(0);
        sizePolicy3.setVerticalStretch(0);
        sizePolicy3.setHeightForWidth(labelMainImage->sizePolicy().hasHeightForWidth());
        labelMainImage->setSizePolicy(sizePolicy3);

        horizontalLayout_2->addWidget(labelMainImage);

        lineMainImage = new QLineEdit(frameMainImage);
        lineMainImage->setObjectName(QString::fromUtf8("lineMainImage"));
        lineMainImage->setStyleSheet(QString::fromUtf8("QLineEdit {\n"
"border: 1px solid dimgray;\n"
"}"));
        lineMainImage->setReadOnly(true);

        horizontalLayout_2->addWidget(lineMainImage);

        buttonMainImage = new QPushButton(frameMainImage);
        buttonMainImage->setObjectName(QString::fromUtf8("buttonMainImage"));
        QSizePolicy sizePolicy4(QSizePolicy::Fixed, QSizePolicy::Minimum);
        sizePolicy4.setHorizontalStretch(0);
        sizePolicy4.setVerticalStretch(0);
        sizePolicy4.setHeightForWidth(buttonMainImage->sizePolicy().hasHeightForWidth());
        buttonMainImage->setSizePolicy(sizePolicy4);

        horizontalLayout_2->addWidget(buttonMainImage);


        verticalLayout_4->addWidget(frameMainImage);

        buttonCompareColours = new QPushButton(groupGeneral);
        buttonCompareColours->setObjectName(QString::fromUtf8("buttonCompareColours"));
        buttonCompareColours->setEnabled(true);
        buttonCompareColours->setStyleSheet(QString::fromUtf8("QPushButton:disabled {\n"
"	background-color: rgb(30, 30, 30);\n"
"	color: rgb(60, 60, 60);\n"
"	border-color: rgb(0, 0, 0);\n"
"}"));

        verticalLayout_4->addWidget(buttonCompareColours);

        line_2 = new QFrame(groupGeneral);
        line_2->setObjectName(QString::fromUtf8("line_2"));
        line_2->setFrameShape(QFrame::HLine);
        line_2->setFrameShadow(QFrame::Sunken);

        verticalLayout_4->addWidget(line_2);

        frameSizeDetail = new QFrame(groupGeneral);
        frameSizeDetail->setObjectName(QString::fromUtf8("frameSizeDetail"));
        sizePolicy2.setHeightForWidth(frameSizeDetail->sizePolicy().hasHeightForWidth());
        frameSizeDetail->setSizePolicy(sizePolicy2);
        frameSizeDetail->setFrameShape(QFrame::StyledPanel);
        frameSizeDetail->setFrameShadow(QFrame::Raised);
        gridLayout = new QGridLayout(frameSizeDetail);
        gridLayout->setSpacing(6);
        gridLayout->setObjectName(QString::fromUtf8("gridLayout"));
        gridLayout->setSizeConstraint(QLayout::SetMinimumSize);
        gridLayout->setContentsMargins(0, 0, 0, 0);
        labelWidth = new QLabel(frameSizeDetail);
        labelWidth->setObjectName(QString::fromUtf8("labelWidth"));
        QSizePolicy sizePolicy5(QSizePolicy::Minimum, QSizePolicy::Fixed);
        sizePolicy5.setHorizontalStretch(0);
        sizePolicy5.setVerticalStretch(0);
        sizePolicy5.setHeightForWidth(labelWidth->sizePolicy().hasHeightForWidth());
        labelWidth->setSizePolicy(sizePolicy5);
        labelWidth->setAlignment(Qt::AlignCenter);

        gridLayout->addWidget(labelWidth, 0, 1, 1, 1);

        labelHeight = new QLabel(frameSizeDetail);
        labelHeight->setObjectName(QString::fromUtf8("labelHeight"));
        sizePolicy5.setHeightForWidth(labelHeight->sizePolicy().hasHeightForWidth());
        labelHeight->setSizePolicy(sizePolicy5);
        labelHeight->setAlignment(Qt::AlignCenter);

        gridLayout->addWidget(labelHeight, 0, 2, 1, 1);

        spinPhotomosaicWidth = new QSpinBox(frameSizeDetail);
        spinPhotomosaicWidth->setObjectName(QString::fromUtf8("spinPhotomosaicWidth"));
        QSizePolicy sizePolicy6(QSizePolicy::Preferred, QSizePolicy::Minimum);
        sizePolicy6.setHorizontalStretch(0);
        sizePolicy6.setVerticalStretch(0);
        sizePolicy6.setHeightForWidth(spinPhotomosaicWidth->sizePolicy().hasHeightForWidth());
        spinPhotomosaicWidth->setSizePolicy(sizePolicy6);
        spinPhotomosaicWidth->setMinimumSize(QSize(0, 20));
        spinPhotomosaicWidth->setStyleSheet(QString::fromUtf8("QSpinBox {\n"
"border: 1px solid dimgray;\n"
"}"));
        spinPhotomosaicWidth->setButtonSymbols(QAbstractSpinBox::PlusMinus);
        spinPhotomosaicWidth->setKeyboardTracking(false);
        spinPhotomosaicWidth->setProperty("showGroupSeparator", QVariant(false));
        spinPhotomosaicWidth->setMinimum(10);
        spinPhotomosaicWidth->setMaximum(100000);

        gridLayout->addWidget(spinPhotomosaicWidth, 1, 1, 1, 1);

        horizontalSpacer_3 = new QSpacerItem(40, 20, QSizePolicy::Expanding, QSizePolicy::Minimum);

        gridLayout->addItem(horizontalSpacer_3, 1, 5, 1, 1);

        buttonPhotomosaicSize = new QPushButton(frameSizeDetail);
        buttonPhotomosaicSize->setObjectName(QString::fromUtf8("buttonPhotomosaicSize"));

        gridLayout->addWidget(buttonPhotomosaicSize, 1, 4, 1, 1);

        spinPhotomosaicHeight = new QSpinBox(frameSizeDetail);
        spinPhotomosaicHeight->setObjectName(QString::fromUtf8("spinPhotomosaicHeight"));
        sizePolicy6.setHeightForWidth(spinPhotomosaicHeight->sizePolicy().hasHeightForWidth());
        spinPhotomosaicHeight->setSizePolicy(sizePolicy6);
        spinPhotomosaicHeight->setStyleSheet(QString::fromUtf8("QSpinBox {\n"
"border: 1px solid dimgray;\n"
"}"));
        spinPhotomosaicHeight->setButtonSymbols(QAbstractSpinBox::PlusMinus);
        spinPhotomosaicHeight->setKeyboardTracking(false);
        spinPhotomosaicHeight->setMinimum(10);
        spinPhotomosaicHeight->setMaximum(100000);

        gridLayout->addWidget(spinPhotomosaicHeight, 1, 2, 1, 1);

        labelSize = new QLabel(frameSizeDetail);
        labelSize->setObjectName(QString::fromUtf8("labelSize"));
        sizePolicy3.setHeightForWidth(labelSize->sizePolicy().hasHeightForWidth());
        labelSize->setSizePolicy(sizePolicy3);

        gridLayout->addWidget(labelSize, 1, 0, 1, 1);

        spinDetail = new QSpinBox(frameSizeDetail);
        spinDetail->setObjectName(QString::fromUtf8("spinDetail"));
        sizePolicy2.setHeightForWidth(spinDetail->sizePolicy().hasHeightForWidth());
        spinDetail->setSizePolicy(sizePolicy2);
        spinDetail->setStyleSheet(QString::fromUtf8("QSpinBox {\n"
"border: 1px solid dimgray;\n"
"}"));
        spinDetail->setButtonSymbols(QAbstractSpinBox::PlusMinus);
        spinDetail->setKeyboardTracking(false);
        spinDetail->setMinimum(1);
        spinDetail->setMaximum(100);
        spinDetail->setValue(50);

        gridLayout->addWidget(spinDetail, 2, 1, 1, 1);

        buttonPhotomosaicSizeLink = new QPushButton(frameSizeDetail);
        buttonPhotomosaicSizeLink->setObjectName(QString::fromUtf8("buttonPhotomosaicSizeLink"));
        QIcon icon;
        icon.addFile(QString::fromUtf8(":/img/LinkIcon.png"), QSize(), QIcon::Normal, QIcon::Off);
        buttonPhotomosaicSizeLink->setIcon(icon);
        buttonPhotomosaicSizeLink->setCheckable(true);
        buttonPhotomosaicSizeLink->setChecked(true);

        gridLayout->addWidget(buttonPhotomosaicSizeLink, 1, 3, 1, 1);

        labelDetail = new QLabel(frameSizeDetail);
        labelDetail->setObjectName(QString::fromUtf8("labelDetail"));

        gridLayout->addWidget(labelDetail, 2, 0, 1, 1);


        verticalLayout_4->addWidget(frameSizeDetail);

        verticalSpacer = new QSpacerItem(1, 0, QSizePolicy::Minimum, QSizePolicy::MinimumExpanding);

        verticalLayout_4->addItem(verticalSpacer);


        verticalLayoutOptions->addWidget(groupGeneral);

        groupColour = new QGroupBox(GeneratorSettings);
        groupColour->setObjectName(QString::fromUtf8("groupColour"));
        sizePolicy1.setHeightForWidth(groupColour->sizePolicy().hasHeightForWidth());
        groupColour->setSizePolicy(sizePolicy1);
        groupColour->setAlignment(Qt::AlignLeading|Qt::AlignLeft|Qt::AlignVCenter);
        groupColour->setFlat(false);
        verticalLayout_7 = new QVBoxLayout(groupColour);
        verticalLayout_7->setObjectName(QString::fromUtf8("verticalLayout_7"));
        verticalLayout_7->setContentsMargins(-1, 14, -1, -1);
        frameMode_2 = new QFrame(groupColour);
        frameMode_2->setObjectName(QString::fromUtf8("frameMode_2"));
        frameMode_2->setFrameShape(QFrame::StyledPanel);
        frameMode_2->setFrameShadow(QFrame::Raised);
        gridLayout_5 = new QGridLayout(frameMode_2);
        gridLayout_5->setObjectName(QString::fromUtf8("gridLayout_5"));
        gridLayout_5->setContentsMargins(0, 0, 0, 0);
        labelColourDifference = new QLabel(frameMode_2);
        labelColourDifference->setObjectName(QString::fromUtf8("labelColourDifference"));

        gridLayout_5->addWidget(labelColourDifference, 0, 0, 1, 1);

        horizontalSpacer_8 = new QSpacerItem(40, 20, QSizePolicy::Expanding, QSizePolicy::Minimum);

        gridLayout_5->addItem(horizontalSpacer_8, 0, 2, 1, 1);

        comboColourDifference = new QComboBox(frameMode_2);
        comboColourDifference->setObjectName(QString::fromUtf8("comboColourDifference"));
        sizePolicy5.setHeightForWidth(comboColourDifference->sizePolicy().hasHeightForWidth());
        comboColourDifference->setSizePolicy(sizePolicy5);

        gridLayout_5->addWidget(comboColourDifference, 0, 1, 1, 1);

        labelColourScheme = new QLabel(frameMode_2);
        labelColourScheme->setObjectName(QString::fromUtf8("labelColourScheme"));

        gridLayout_5->addWidget(labelColourScheme, 1, 0, 1, 1);

        comboColourScheme = new QComboBox(frameMode_2);
        comboColourScheme->setObjectName(QString::fromUtf8("comboColourScheme"));

        gridLayout_5->addWidget(comboColourScheme, 1, 1, 1, 1);


        verticalLayout_7->addWidget(frameMode_2);

        verticalSpacer_4 = new QSpacerItem(1, 0, QSizePolicy::Minimum, QSizePolicy::MinimumExpanding);

        verticalLayout_7->addItem(verticalSpacer_4);


        verticalLayoutOptions->addWidget(groupColour);

        groupCells = new QGroupBox(GeneratorSettings);
        groupCells->setObjectName(QString::fromUtf8("groupCells"));
        sizePolicy1.setHeightForWidth(groupCells->sizePolicy().hasHeightForWidth());
        groupCells->setSizePolicy(sizePolicy1);
        groupCells->setStyleSheet(QString::fromUtf8(""));
        verticalLayout_5 = new QVBoxLayout(groupCells);
        verticalLayout_5->setObjectName(QString::fromUtf8("verticalLayout_5"));
        frameCellSize = new QFrame(groupCells);
        frameCellSize->setObjectName(QString::fromUtf8("frameCellSize"));
        frameCellSize->setFrameShape(QFrame::StyledPanel);
        frameCellSize->setFrameShadow(QFrame::Raised);
        horizontalLayout_4 = new QHBoxLayout(frameCellSize);
        horizontalLayout_4->setObjectName(QString::fromUtf8("horizontalLayout_4"));
        horizontalLayout_4->setContentsMargins(0, 0, 0, 0);
        labelCellSize = new QLabel(frameCellSize);
        labelCellSize->setObjectName(QString::fromUtf8("labelCellSize"));

        horizontalLayout_4->addWidget(labelCellSize);

        spinCellSize = new QSpinBox(frameCellSize);
        spinCellSize->setObjectName(QString::fromUtf8("spinCellSize"));
        spinCellSize->setMinimumSize(QSize(0, 0));
        spinCellSize->setStyleSheet(QString::fromUtf8("QSpinBox {\n"
"border: 1px solid dimgray;\n"
"}"));
        spinCellSize->setButtonSymbols(QAbstractSpinBox::PlusMinus);
        spinCellSize->setKeyboardTracking(false);
        spinCellSize->setMinimum(10);
        spinCellSize->setMaximum(10000);

        horizontalLayout_4->addWidget(spinCellSize);

        horizontalSpacer_6 = new QSpacerItem(40, 20, QSizePolicy::Expanding, QSizePolicy::Minimum);

        horizontalLayout_4->addItem(horizontalSpacer_6);


        verticalLayout_5->addWidget(frameCellSize);

        frameCellShape = new QFrame(groupCells);
        frameCellShape->setObjectName(QString::fromUtf8("frameCellShape"));
        sizePolicy2.setHeightForWidth(frameCellShape->sizePolicy().hasHeightForWidth());
        frameCellShape->setSizePolicy(sizePolicy2);
        frameCellShape->setStyleSheet(QString::fromUtf8(""));
        frameCellShape->setFrameShape(QFrame::StyledPanel);
        frameCellShape->setFrameShadow(QFrame::Raised);
        horizontalLayout_3 = new QHBoxLayout(frameCellShape);
        horizontalLayout_3->setObjectName(QString::fromUtf8("horizontalLayout_3"));
        horizontalLayout_3->setContentsMargins(0, 0, 0, 0);
        checkCellShape = new QCheckBox(frameCellShape);
        checkCellShape->setObjectName(QString::fromUtf8("checkCellShape"));
        checkCellShape->setStyleSheet(QString::fromUtf8(""));

        horizontalLayout_3->addWidget(checkCellShape);

        lineCellShape = new QLineEdit(frameCellShape);
        lineCellShape->setObjectName(QString::fromUtf8("lineCellShape"));
        lineCellShape->setEnabled(false);
        lineCellShape->setStyleSheet(QString::fromUtf8("QLineEdit {\n"
"border: 1px solid dimgray;\n"
"}\n"
"\n"
"QLineEdit:disabled {\n"
"	background-color: rgb(30, 30, 30);\n"
"	color: rgb(60, 60, 60);\n"
"	border-color: rgb(0, 0, 0);\n"
"}"));
        lineCellShape->setReadOnly(true);

        horizontalLayout_3->addWidget(lineCellShape);


        verticalLayout_5->addWidget(frameCellShape);

        line_5 = new QFrame(groupCells);
        line_5->setObjectName(QString::fromUtf8("line_5"));
        line_5->setFrameShape(QFrame::HLine);
        line_5->setFrameShadow(QFrame::Sunken);

        verticalLayout_5->addWidget(line_5);

        horizontalLayoutMinCellSize = new QHBoxLayout();
        horizontalLayoutMinCellSize->setObjectName(QString::fromUtf8("horizontalLayoutMinCellSize"));
        horizontalLayoutMinCellSize->setContentsMargins(-1, 0, -1, -1);
        labelSizeSteps = new QLabel(groupCells);
        labelSizeSteps->setObjectName(QString::fromUtf8("labelSizeSteps"));

        horizontalLayoutMinCellSize->addWidget(labelSizeSteps);

        spinSizeSteps = new QSpinBox(groupCells);
        spinSizeSteps->setObjectName(QString::fromUtf8("spinSizeSteps"));
        spinSizeSteps->setStyleSheet(QString::fromUtf8("QSpinBox {\n"
"border: 1px solid dimgray;\n"
"}"));
        spinSizeSteps->setButtonSymbols(QAbstractSpinBox::PlusMinus);

        horizontalLayoutMinCellSize->addWidget(spinSizeSteps);

        horizontalSpacer_10 = new QSpacerItem(40, 20, QSizePolicy::Expanding, QSizePolicy::Minimum);

        horizontalLayoutMinCellSize->addItem(horizontalSpacer_10);


        verticalLayout_5->addLayout(horizontalLayoutMinCellSize);

        labelCellSizesList = new QLabel(groupCells);
        labelCellSizesList->setObjectName(QString::fromUtf8("labelCellSizesList"));

        verticalLayout_5->addWidget(labelCellSizesList);

        line_6 = new QFrame(groupCells);
        line_6->setObjectName(QString::fromUtf8("line_6"));
        line_6->setFrameShape(QFrame::HLine);
        line_6->setFrameShadow(QFrame::Sunken);

        verticalLayout_5->addWidget(line_6);

        buttonEditGrid = new QPushButton(groupCells);
        buttonEditGrid->setObjectName(QString::fromUtf8("buttonEditGrid"));
        buttonEditGrid->setEnabled(true);

        verticalLayout_5->addWidget(buttonEditGrid);

        verticalSpacer_2 = new QSpacerItem(1, 0, QSizePolicy::Minimum, QSizePolicy::MinimumExpanding);

        verticalLayout_5->addItem(verticalSpacer_2);


        verticalLayoutOptions->addWidget(groupCells);

        groupRepeats = new QGroupBox(GeneratorSettings);
        groupRepeats->setObjectName(QString::fromUtf8("groupRepeats"));
        sizePolicy1.setHeightForWidth(groupRepeats->sizePolicy().hasHeightForWidth());
        groupRepeats->setSizePolicy(sizePolicy1);
        verticalLayout_6 = new QVBoxLayout(groupRepeats);
        verticalLayout_6->setObjectName(QString::fromUtf8("verticalLayout_6"));
        frame_5 = new QFrame(groupRepeats);
        frame_5->setObjectName(QString::fromUtf8("frame_5"));
        frame_5->setFrameShape(QFrame::StyledPanel);
        frame_5->setFrameShadow(QFrame::Raised);
        gridLayout_3 = new QGridLayout(frame_5);
        gridLayout_3->setObjectName(QString::fromUtf8("gridLayout_3"));
        gridLayout_3->setContentsMargins(0, 0, 0, 0);
        horizontalSpacer_4 = new QSpacerItem(40, 20, QSizePolicy::Expanding, QSizePolicy::Minimum);

        gridLayout_3->addItem(horizontalSpacer_4, 0, 2, 1, 1);

        labelRepeatAddition = new QLabel(frame_5);
        labelRepeatAddition->setObjectName(QString::fromUtf8("labelRepeatAddition"));

        gridLayout_3->addWidget(labelRepeatAddition, 1, 0, 1, 1);

        labelRepeatRange = new QLabel(frame_5);
        labelRepeatRange->setObjectName(QString::fromUtf8("labelRepeatRange"));
        QSizePolicy sizePolicy7(QSizePolicy::Preferred, QSizePolicy::Preferred);
        sizePolicy7.setHorizontalStretch(0);
        sizePolicy7.setVerticalStretch(0);
        sizePolicy7.setHeightForWidth(labelRepeatRange->sizePolicy().hasHeightForWidth());
        labelRepeatRange->setSizePolicy(sizePolicy7);

        gridLayout_3->addWidget(labelRepeatRange, 0, 0, 1, 1);

        spinRepeatRange = new QSpinBox(frame_5);
        spinRepeatRange->setObjectName(QString::fromUtf8("spinRepeatRange"));
        QSizePolicy sizePolicy8(QSizePolicy::Preferred, QSizePolicy::Fixed);
        sizePolicy8.setHorizontalStretch(0);
        sizePolicy8.setVerticalStretch(0);
        sizePolicy8.setHeightForWidth(spinRepeatRange->sizePolicy().hasHeightForWidth());
        spinRepeatRange->setSizePolicy(sizePolicy8);
        spinRepeatRange->setMinimumSize(QSize(0, 20));
        spinRepeatRange->setStyleSheet(QString::fromUtf8("QSpinBox {\n"
"border: 1px solid dimgray;\n"
"}"));
        spinRepeatRange->setButtonSymbols(QAbstractSpinBox::PlusMinus);
        spinRepeatRange->setKeyboardTracking(false);
        spinRepeatRange->setProperty("showGroupSeparator", QVariant(false));
        spinRepeatRange->setMinimum(0);
        spinRepeatRange->setMaximum(100000);
        spinRepeatRange->setValue(0);

        gridLayout_3->addWidget(spinRepeatRange, 0, 1, 1, 1);

        spinRepeatAddition = new QSpinBox(frame_5);
        spinRepeatAddition->setObjectName(QString::fromUtf8("spinRepeatAddition"));
        sizePolicy8.setHeightForWidth(spinRepeatAddition->sizePolicy().hasHeightForWidth());
        spinRepeatAddition->setSizePolicy(sizePolicy8);
        spinRepeatAddition->setMinimumSize(QSize(0, 20));
        spinRepeatAddition->setStyleSheet(QString::fromUtf8("QSpinBox {\n"
"border: 1px solid dimgray;\n"
"}"));
        spinRepeatAddition->setButtonSymbols(QAbstractSpinBox::PlusMinus);
        spinRepeatAddition->setKeyboardTracking(false);
        spinRepeatAddition->setProperty("showGroupSeparator", QVariant(false));
        spinRepeatAddition->setMinimum(0);
        spinRepeatAddition->setMaximum(100000);
        spinRepeatAddition->setValue(0);

        gridLayout_3->addWidget(spinRepeatAddition, 1, 1, 1, 1);


        verticalLayout_6->addWidget(frame_5);

        verticalSpacer_3 = new QSpacerItem(1, 0, QSizePolicy::Minimum, QSizePolicy::MinimumExpanding);

        verticalLayout_6->addItem(verticalSpacer_3);


        verticalLayoutOptions->addWidget(groupRepeats);

        verticalSpacer_5 = new QSpacerItem(1, 0, QSizePolicy::Minimum, QSizePolicy::MinimumExpanding);

        verticalLayoutOptions->addItem(verticalSpacer_5);

        groupGenerate = new QHBoxLayout();
        groupGenerate->setObjectName(QString::fromUtf8("groupGenerate"));
        checkCUDA = new QCheckBox(GeneratorSettings);
        checkCUDA->setObjectName(QString::fromUtf8("checkCUDA"));
        checkCUDA->setLayoutDirection(Qt::RightToLeft);
        checkCUDA->setStyleSheet(QString::fromUtf8(""));
        checkCUDA->setChecked(true);

        groupGenerate->addWidget(checkCUDA, 0, Qt::AlignRight);

        comboCUDA = new QComboBox(GeneratorSettings);
        comboCUDA->setObjectName(QString::fromUtf8("comboCUDA"));

        groupGenerate->addWidget(comboCUDA);

        horizontalSpacer_9 = new QSpacerItem(40, 20, QSizePolicy::Expanding, QSizePolicy::Minimum);

        groupGenerate->addItem(horizontalSpacer_9);

        buttonGenerate = new QPushButton(GeneratorSettings);
        buttonGenerate->setObjectName(QString::fromUtf8("buttonGenerate"));
        sizePolicy3.setHeightForWidth(buttonGenerate->sizePolicy().hasHeightForWidth());
        buttonGenerate->setSizePolicy(sizePolicy3);
        buttonGenerate->setLayoutDirection(Qt::LeftToRight);

        groupGenerate->addWidget(buttonGenerate, 0, Qt::AlignRight);


        verticalLayoutOptions->addLayout(groupGenerate);


        horizontalLayout_7->addLayout(verticalLayoutOptions);

        widgetGridPreview = new GridViewer(GeneratorSettings);
        widgetGridPreview->setObjectName(QString::fromUtf8("widgetGridPreview"));
        QSizePolicy sizePolicy9(QSizePolicy::Expanding, QSizePolicy::Preferred);
        sizePolicy9.setHorizontalStretch(2);
        sizePolicy9.setVerticalStretch(0);
        sizePolicy9.setHeightForWidth(widgetGridPreview->sizePolicy().hasHeightForWidth());
        widgetGridPreview->setSizePolicy(sizePolicy9);

        horizontalLayout_7->addWidget(widgetGridPreview);

        tabWidget->addTab(GeneratorSettings, QString());

        verticalLayout->addWidget(tabWidget);

        MainWindow->setCentralWidget(centralwidget);
        menubar = new QMenuBar(MainWindow);
        menubar->setObjectName(QString::fromUtf8("menubar"));
        menubar->setGeometry(QRect(0, 0, 1080, 21));
        menuAbout = new QMenu(menubar);
        menuAbout->setObjectName(QString::fromUtf8("menuAbout"));
        MainWindow->setMenuBar(menubar);
        statusbar = new QStatusBar(MainWindow);
        statusbar->setObjectName(QString::fromUtf8("statusbar"));
        MainWindow->setStatusBar(statusbar);

        menubar->addAction(menuAbout->menuAction());
        menuAbout->addAction(actionWebsite);
        menuAbout->addAction(actionGithub);
        menuAbout->addSeparator();
        menuAbout->addAction(actionAbout);

        retranslateUi(MainWindow);

        tabWidget->setCurrentIndex(2);


        QMetaObject::connectSlotsByName(MainWindow);
    } // setupUi

    void retranslateUi(QMainWindow *MainWindow)
    {
        MainWindow->setWindowTitle(QCoreApplication::translate("MainWindow", "Mosaic Magnifique", nullptr));
        actionWebsite->setText(QCoreApplication::translate("MainWindow", "Website", nullptr));
        actionGithub->setText(QCoreApplication::translate("MainWindow", "Github", nullptr));
        actionAbout->setText(QCoreApplication::translate("MainWindow", "About Mosaic Magnifique", nullptr));
#if QT_CONFIG(statustip)
        tabWidget->setStatusTip(QString());
#endif // QT_CONFIG(statustip)
#if QT_CONFIG(whatsthis)
        tabWidget->setWhatsThis(QCoreApplication::translate("MainWindow", "<html><head/><body><p>Cell Shape: Create, load, and save custom cell shapes.</p><p>Image Library (N): Create, load, and save image libraries. N = Number of images in current library.</p><p>Generator Settings: Generate Photomosaic and modify settings used in generation.</p></body></html>", nullptr));
#endif // QT_CONFIG(whatsthis)
        tabWidget->setTabText(tabWidget->indexOf(cellShapeEditor), QCoreApplication::translate("MainWindow", "Cell Shape", nullptr));
        tabWidget->setTabText(tabWidget->indexOf(imageLibraryEditor), QCoreApplication::translate("MainWindow", "Image Library (0)", nullptr));
        groupGeneral->setTitle(QCoreApplication::translate("MainWindow", "General", nullptr));
#if QT_CONFIG(statustip)
        labelMainImage->setStatusTip(QCoreApplication::translate("MainWindow", "The main image to base the Photomosaic on", nullptr));
#endif // QT_CONFIG(statustip)
        labelMainImage->setText(QCoreApplication::translate("MainWindow", "Main Image:", nullptr));
#if QT_CONFIG(statustip)
        lineMainImage->setStatusTip(QCoreApplication::translate("MainWindow", "Displays path and filename of main image", nullptr));
#endif // QT_CONFIG(statustip)
#if QT_CONFIG(statustip)
        buttonMainImage->setStatusTip(QCoreApplication::translate("MainWindow", "Opens a file dialog to allow an image file to be chosen", nullptr));
#endif // QT_CONFIG(statustip)
        buttonMainImage->setText(QCoreApplication::translate("MainWindow", "Browse", nullptr));
#if QT_CONFIG(statustip)
        buttonCompareColours->setStatusTip(QCoreApplication::translate("MainWindow", "Compare colours in main image and library images", nullptr));
#endif // QT_CONFIG(statustip)
#if QT_CONFIG(whatsthis)
        buttonCompareColours->setWhatsThis(QCoreApplication::translate("MainWindow", "<html><head/><body><p>Provides a list of colours that could improve Photomosaic if images containing similar colours are added to library.</p><p><br/></p><p>Colours listed in order of descending priority.</p></body></html>", nullptr));
#endif // QT_CONFIG(whatsthis)
        buttonCompareColours->setText(QCoreApplication::translate("MainWindow", "Compare Colours", nullptr));
        labelWidth->setText(QCoreApplication::translate("MainWindow", "Width", nullptr));
        labelHeight->setText(QCoreApplication::translate("MainWindow", "Height", nullptr));
#if QT_CONFIG(statustip)
        spinPhotomosaicWidth->setStatusTip(QCoreApplication::translate("MainWindow", "Controls width of Photomosaic in pixels", nullptr));
#endif // QT_CONFIG(statustip)
        spinPhotomosaicWidth->setSpecialValueText(QString());
        spinPhotomosaicWidth->setSuffix(QCoreApplication::translate("MainWindow", "px", nullptr));
        spinPhotomosaicWidth->setPrefix(QString());
#if QT_CONFIG(statustip)
        buttonPhotomosaicSize->setStatusTip(QCoreApplication::translate("MainWindow", "Sets size to size of main image", nullptr));
#endif // QT_CONFIG(statustip)
        buttonPhotomosaicSize->setText(QCoreApplication::translate("MainWindow", "From Main Image", nullptr));
#if QT_CONFIG(statustip)
        spinPhotomosaicHeight->setStatusTip(QCoreApplication::translate("MainWindow", "Controls height of Photomosaic in pixels", nullptr));
#endif // QT_CONFIG(statustip)
        spinPhotomosaicHeight->setSuffix(QCoreApplication::translate("MainWindow", "px", nullptr));
        spinPhotomosaicHeight->setPrefix(QString());
#if QT_CONFIG(statustip)
        labelSize->setStatusTip(QCoreApplication::translate("MainWindow", "Controls size of main image", nullptr));
#endif // QT_CONFIG(statustip)
        labelSize->setText(QCoreApplication::translate("MainWindow", "Size:", nullptr));
#if QT_CONFIG(statustip)
        spinDetail->setStatusTip(QCoreApplication::translate("MainWindow", "Controls detail level when comparing images", nullptr));
#endif // QT_CONFIG(statustip)
#if QT_CONFIG(whatsthis)
        spinDetail->setWhatsThis(QCoreApplication::translate("MainWindow", "<html><head/><body><p>Controls the detail level when comparing images.</p><p>Lowering detail allows Photomosaics to be generated faster, but results in less detail being identifiable from main image.</p><p>If detail level ever results in a cell being 0px (empty) the detail is increased automatically.</p></body></html>", nullptr));
#endif // QT_CONFIG(whatsthis)
        spinDetail->setSuffix(QCoreApplication::translate("MainWindow", "%", nullptr));
#if QT_CONFIG(statustip)
        buttonPhotomosaicSizeLink->setStatusTip(QCoreApplication::translate("MainWindow", "Locks aspect ratio of width/height", nullptr));
#endif // QT_CONFIG(statustip)
        buttonPhotomosaicSizeLink->setText(QString());
#if QT_CONFIG(statustip)
        labelDetail->setStatusTip(QCoreApplication::translate("MainWindow", "Controls detail level when comparing images", nullptr));
#endif // QT_CONFIG(statustip)
        labelDetail->setText(QCoreApplication::translate("MainWindow", "Detail:", nullptr));
        groupColour->setTitle(QCoreApplication::translate("MainWindow", "Colour", nullptr));
#if QT_CONFIG(statustip)
        labelColourDifference->setStatusTip(QCoreApplication::translate("MainWindow", "Controls how a Photomosaic is generated", nullptr));
#endif // QT_CONFIG(statustip)
        labelColourDifference->setText(QCoreApplication::translate("MainWindow", "Difference:", nullptr));
#if QT_CONFIG(statustip)
        comboColourDifference->setStatusTip(QCoreApplication::translate("MainWindow", "Controls how a Photomosaic is generated", nullptr));
#endif // QT_CONFIG(statustip)
#if QT_CONFIG(whatsthis)
        comboColourDifference->setWhatsThis(QCoreApplication::translate("MainWindow", "<html><head/><body><p>A photomosaic is generated by comparing cells against the library images and finding which is the best fit.</p><p>The mode controls which formula is used for comparing images.</p><p>The following is descriptions of each mode (in order of increasing time complexity and result accuracy):</p>\n"
"<ul>\n"
"<li>RGB Euclidean: Compares images in the RGB colour space using a euclidean difference formula</li>\n"
"<li>CIE76: Compares images in the CIELAB colour space using a euclidean difference formula</li>\n"
"<li>CIEDE2000: Compares images in the CIELAB colour space using CIEDE2000 difference formula</li>\n"
"</ul>\n"
"</body></html>", nullptr));
#endif // QT_CONFIG(whatsthis)
        labelColourScheme->setText(QCoreApplication::translate("MainWindow", "Scheme:", nullptr));
        groupCells->setTitle(QCoreApplication::translate("MainWindow", "Cells", nullptr));
#if QT_CONFIG(statustip)
        labelCellSize->setStatusTip(QCoreApplication::translate("MainWindow", "Width and height of each cell in pixels", nullptr));
#endif // QT_CONFIG(statustip)
        labelCellSize->setText(QCoreApplication::translate("MainWindow", "Cell Size:", nullptr));
#if QT_CONFIG(statustip)
        spinCellSize->setStatusTip(QCoreApplication::translate("MainWindow", "Width and height of each cell in pixels", nullptr));
#endif // QT_CONFIG(statustip)
#if QT_CONFIG(whatsthis)
        spinCellSize->setWhatsThis(QCoreApplication::translate("MainWindow", "<html><head/><body><p>When generating a Photomosaic, the main image is split into cells. Each cell is replaced by a library image.</p><p>The cell size sets the width and height of each cell in pixels.</p><p>A smaller cell size results in more detail from the main image being identifiable, but less detail in the library images.</p></body></html>", nullptr));
#endif // QT_CONFIG(whatsthis)
        spinCellSize->setSpecialValueText(QString());
        spinCellSize->setSuffix(QCoreApplication::translate("MainWindow", "px", nullptr));
#if QT_CONFIG(statustip)
        checkCellShape->setStatusTip(QCoreApplication::translate("MainWindow", "Controls use of custom cell shapes", nullptr));
#endif // QT_CONFIG(statustip)
#if QT_CONFIG(whatsthis)
        checkCellShape->setWhatsThis(QCoreApplication::translate("MainWindow", "<html><head/><body><p>When enabled will use the currently loaded cell shape for generating Photomosaics.</p><p>When disabled, or no cell shape is loaded, the default cell shape (square) is used.</p></body></html>", nullptr));
#endif // QT_CONFIG(whatsthis)
        checkCellShape->setText(QCoreApplication::translate("MainWindow", "Cell Shape:", nullptr));
#if QT_CONFIG(statustip)
        lineCellShape->setStatusTip(QCoreApplication::translate("MainWindow", "Name of currently loaded cell shape", nullptr));
#endif // QT_CONFIG(statustip)
#if QT_CONFIG(whatsthis)
        lineCellShape->setWhatsThis(QCoreApplication::translate("MainWindow", "<html><head/><body><p>Displays the name of the currently loaded cell shape. When disabled the box is greyed out.</p><p>If no cell shape is loaded the default (square cell) is used.</p></body></html>", nullptr));
#endif // QT_CONFIG(whatsthis)
#if QT_CONFIG(statustip)
        labelSizeSteps->setStatusTip(QCoreApplication::translate("MainWindow", "Maximum number of times cell can be split", nullptr));
#endif // QT_CONFIG(statustip)
        labelSizeSteps->setText(QCoreApplication::translate("MainWindow", "Size Steps:", nullptr));
#if QT_CONFIG(statustip)
        spinSizeSteps->setStatusTip(QCoreApplication::translate("MainWindow", "Maximum number of times cell can be split", nullptr));
#endif // QT_CONFIG(statustip)
#if QT_CONFIG(whatsthis)
        spinSizeSteps->setWhatsThis(QCoreApplication::translate("MainWindow", "<html><head/><body><p>Photomosaics can be generated using variable cell sizes. Areas of higher detail will use a smaller cell size.</p><p>For each step down a cell is split into four smaller cells of half the size (and rounded down). For best cell alignment the cell size should be a power of 2.</p></body></html>", nullptr));
#endif // QT_CONFIG(whatsthis)
        spinSizeSteps->setSuffix(QString());
#if QT_CONFIG(statustip)
        labelCellSizesList->setStatusTip(QCoreApplication::translate("MainWindow", "List of cell sizes", nullptr));
#endif // QT_CONFIG(statustip)
        labelCellSizesList->setText(QCoreApplication::translate("MainWindow", "Cell Sizes: 10px", nullptr));
#if QT_CONFIG(statustip)
        buttonEditGrid->setStatusTip(QCoreApplication::translate("MainWindow", "Manually edit the grid state", nullptr));
#endif // QT_CONFIG(statustip)
#if QT_CONFIG(whatsthis)
        buttonEditGrid->setWhatsThis(QCoreApplication::translate("MainWindow", "<html><head/><body><p>Opens grid editor.</p><p>Allows you to edit the state of each cell in the grid.</p></body></html>", nullptr));
#endif // QT_CONFIG(whatsthis)
        buttonEditGrid->setText(QCoreApplication::translate("MainWindow", "Edit Grid", nullptr));
        groupRepeats->setTitle(QCoreApplication::translate("MainWindow", "Repeats", nullptr));
#if QT_CONFIG(statustip)
        labelRepeatAddition->setStatusTip(QCoreApplication::translate("MainWindow", "Value to add to each cell based on number of repeats", nullptr));
#endif // QT_CONFIG(statustip)
#if QT_CONFIG(whatsthis)
        labelRepeatAddition->setWhatsThis(QString());
#endif // QT_CONFIG(whatsthis)
        labelRepeatAddition->setText(QCoreApplication::translate("MainWindow", "Repeat Addition:", nullptr));
#if QT_CONFIG(statustip)
        labelRepeatRange->setStatusTip(QCoreApplication::translate("MainWindow", "Range around each cell to look for repeats", nullptr));
#endif // QT_CONFIG(statustip)
#if QT_CONFIG(whatsthis)
        labelRepeatRange->setWhatsThis(QString());
#endif // QT_CONFIG(whatsthis)
        labelRepeatRange->setText(QCoreApplication::translate("MainWindow", "Repeat Range:", nullptr));
#if QT_CONFIG(statustip)
        spinRepeatRange->setStatusTip(QCoreApplication::translate("MainWindow", "Range around each cell to look for repeats", nullptr));
#endif // QT_CONFIG(statustip)
#if QT_CONFIG(whatsthis)
        spinRepeatRange->setWhatsThis(QCoreApplication::translate("MainWindow", "<html><head/><body><p>Defines the range in cells to look around each cell for repeating images.</p></body></html>", nullptr));
#endif // QT_CONFIG(whatsthis)
        spinRepeatRange->setSpecialValueText(QString());
        spinRepeatRange->setSuffix(QString());
        spinRepeatRange->setPrefix(QString());
#if QT_CONFIG(statustip)
        spinRepeatAddition->setStatusTip(QCoreApplication::translate("MainWindow", "Value to add to each cell based on number of repeats", nullptr));
#endif // QT_CONFIG(statustip)
#if QT_CONFIG(whatsthis)
        spinRepeatAddition->setWhatsThis(QCoreApplication::translate("MainWindow", "<html><head/><body><p>To generate the Photomosaic the cell images are compared against the library images, this gives a non-negative variant. The closer the variant is to 0 to better of a fit the image is.</p><p><br/></p><p>The repeat value is added to this variant based on the number of repeats in range, making an image a less desirable fit with more repeats.</p><p><br/></p><p>Max variants to help with choosing repeat addition:</p><p>RGB =~ 442 * (Cell Size * Detail)^2</p><p>CIE76 =~ 375 * (Cell Size * Detail)^2</p><p>CIEDE2000 =~ <span style=\" font-style:italic;\">154?</span> * (Cell Size * Detail)^2</p></body></html>", nullptr));
#endif // QT_CONFIG(whatsthis)
        spinRepeatAddition->setSpecialValueText(QString());
        spinRepeatAddition->setSuffix(QString());
        spinRepeatAddition->setPrefix(QString());
#if QT_CONFIG(statustip)
        checkCUDA->setStatusTip(QCoreApplication::translate("MainWindow", "Controls CUDA usage", nullptr));
#endif // QT_CONFIG(statustip)
#if QT_CONFIG(whatsthis)
        checkCUDA->setWhatsThis(QCoreApplication::translate("MainWindow", "<html><head/><body><p>Enabling CUDA allows a GPU to be used to generate the Photomosaic.</p><p>This will generally result in much faster results, depending on your system.</p></body></html>", nullptr));
#endif // QT_CONFIG(whatsthis)
        checkCUDA->setText(QCoreApplication::translate("MainWindow", "CUDA:", nullptr));
#if QT_CONFIG(statustip)
        buttonGenerate->setStatusTip(QCoreApplication::translate("MainWindow", "Generate a Photomosaic using current settings", nullptr));
#endif // QT_CONFIG(statustip)
        buttonGenerate->setText(QCoreApplication::translate("MainWindow", "Generate Photomosaic", nullptr));
#if QT_CONFIG(statustip)
        widgetGridPreview->setStatusTip(QCoreApplication::translate("MainWindow", "Displays a preview of the grid on main image", nullptr));
#endif // QT_CONFIG(statustip)
#if QT_CONFIG(whatsthis)
        widgetGridPreview->setWhatsThis(QCoreApplication::translate("MainWindow", "<html><head/><body><p>Displays a preview of the cell grid overlaid on the main image to give some idea of how the Photomosaic will look.</p><p>When the edge detect check box is enabled an edge-detected version of the cell shape is used to form the grid preview.</p></body></html>", nullptr));
#endif // QT_CONFIG(whatsthis)
        tabWidget->setTabText(tabWidget->indexOf(GeneratorSettings), QCoreApplication::translate("MainWindow", "Generator Settings", nullptr));
        menuAbout->setTitle(QCoreApplication::translate("MainWindow", "Help", nullptr));
    } // retranslateUi

};

namespace Ui {
    class MainWindow: public Ui_MainWindow {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_MAINWINDOW_H
