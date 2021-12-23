/********************************************************************************
** Form generated from reading UI file 'CellShapeEditor.ui'
**
** Created by: Qt User Interface Compiler version 5.15.0
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef UI_CELLSHAPEEDITOR_H
#define UI_CELLSHAPEEDITOR_H

#include <QtCore/QVariant>
#include <QtWidgets/QApplication>
#include <QtWidgets/QCheckBox>
#include <QtWidgets/QFrame>
#include <QtWidgets/QGridLayout>
#include <QtWidgets/QHBoxLayout>
#include <QtWidgets/QLabel>
#include <QtWidgets/QLineEdit>
#include <QtWidgets/QPushButton>
#include <QtWidgets/QSpacerItem>
#include <QtWidgets/QSpinBox>
#include <QtWidgets/QVBoxLayout>
#include <QtWidgets/QWidget>
#include "gridviewer.h"

QT_BEGIN_NAMESPACE

class Ui_CellShapeEditor
{
public:
    QVBoxLayout *verticalLayout_7;
    QFrame *frameSaveLoad;
    QHBoxLayout *horizontalLayout_5;
    QPushButton *buttonSaveCell;
    QPushButton *buttonLoadCell;
    QSpacerItem *horizontalSpacer_1;
    QFrame *horizontalLine_1;
    QFrame *frameCellName;
    QHBoxLayout *horizontalLayout_6;
    QLabel *labelCellName;
    QLineEdit *lineCellName;
    QFrame *frameCellMask;
    QHBoxLayout *horizontalLayout_8;
    QLabel *labelCellMask;
    QLineEdit *lineCellMaskPath;
    QPushButton *buttonCellMask;
    QFrame *horizontalLine_3;
    QFrame *frameCellShapeSettings;
    QHBoxLayout *horizontalLayout;
    QGridLayout *gridCellSpacing;
    QLabel *labelRow;
    QSpinBox *spinCellSpacingRow;
    QLabel *labelSpacing;
    QSpinBox *spinCellAlternateOffsetRow;
    QSpinBox *spinCellAlternateOffsetCol;
    QLabel *labelAlternateOffset;
    QSpinBox *spinCellSpacingCol;
    QLabel *labelColumn;
    QSpacerItem *verticalSpacer_3;
    QFrame *verticalLine_1;
    QGridLayout *gridCellFlipping;
    QLabel *labelHorizontal;
    QLabel *labelColumnFlipping;
    QCheckBox *checkCellRowFlipH;
    QLabel *labelRowFlipping;
    QLabel *labelVertical;
    QCheckBox *checkCellColFlipH;
    QCheckBox *checkCellRowFlipV;
    QCheckBox *checkCellColFlipV;
    QSpacerItem *verticalSpacer_2;
    QFrame *verticalLine_2;
    QGridLayout *gridAlternateSpacing;
    QSpinBox *spinCellAlternateSpacingRow;
    QCheckBox *checkCellAlternateSpacingCol;
    QCheckBox *checkCellAlternateSpacingRow;
    QSpinBox *spinCellAlternateSpacingCol;
    QLabel *labelAlternateSpacing;
    QSpacerItem *verticalSpacer;
    QFrame *verticalLine_3;
    QSpacerItem *horizontalSpacer_8;
    QFrame *horizontalLine_2;
    GridViewer *cellShapeViewer;

    void setupUi(QWidget *CellShapeEditor)
    {
        if (CellShapeEditor->objectName().isEmpty())
            CellShapeEditor->setObjectName(QString::fromUtf8("CellShapeEditor"));
        CellShapeEditor->resize(1080, 720);
        CellShapeEditor->setStyleSheet(QString::fromUtf8("QWidget {\n"
"background-color: rgb(60, 60, 60);\n"
"color: rgb(255, 255, 255);\n"
"border-color: rgb(0, 0, 0);\n"
"}"));
        verticalLayout_7 = new QVBoxLayout(CellShapeEditor);
        verticalLayout_7->setObjectName(QString::fromUtf8("verticalLayout_7"));
        frameSaveLoad = new QFrame(CellShapeEditor);
        frameSaveLoad->setObjectName(QString::fromUtf8("frameSaveLoad"));
        QSizePolicy sizePolicy(QSizePolicy::Preferred, QSizePolicy::Preferred);
        sizePolicy.setHorizontalStretch(0);
        sizePolicy.setVerticalStretch(0);
        sizePolicy.setHeightForWidth(frameSaveLoad->sizePolicy().hasHeightForWidth());
        frameSaveLoad->setSizePolicy(sizePolicy);
        frameSaveLoad->setFrameShape(QFrame::StyledPanel);
        frameSaveLoad->setFrameShadow(QFrame::Raised);
        horizontalLayout_5 = new QHBoxLayout(frameSaveLoad);
        horizontalLayout_5->setObjectName(QString::fromUtf8("horizontalLayout_5"));
        horizontalLayout_5->setContentsMargins(0, 0, 0, 0);
        buttonSaveCell = new QPushButton(frameSaveLoad);
        buttonSaveCell->setObjectName(QString::fromUtf8("buttonSaveCell"));

        horizontalLayout_5->addWidget(buttonSaveCell);

        buttonLoadCell = new QPushButton(frameSaveLoad);
        buttonLoadCell->setObjectName(QString::fromUtf8("buttonLoadCell"));

        horizontalLayout_5->addWidget(buttonLoadCell);

        horizontalSpacer_1 = new QSpacerItem(0, 0, QSizePolicy::Expanding, QSizePolicy::Minimum);

        horizontalLayout_5->addItem(horizontalSpacer_1);


        verticalLayout_7->addWidget(frameSaveLoad);

        horizontalLine_1 = new QFrame(CellShapeEditor);
        horizontalLine_1->setObjectName(QString::fromUtf8("horizontalLine_1"));
        horizontalLine_1->setFrameShape(QFrame::HLine);
        horizontalLine_1->setFrameShadow(QFrame::Sunken);

        verticalLayout_7->addWidget(horizontalLine_1);

        frameCellName = new QFrame(CellShapeEditor);
        frameCellName->setObjectName(QString::fromUtf8("frameCellName"));
        sizePolicy.setHeightForWidth(frameCellName->sizePolicy().hasHeightForWidth());
        frameCellName->setSizePolicy(sizePolicy);
        frameCellName->setFrameShape(QFrame::StyledPanel);
        frameCellName->setFrameShadow(QFrame::Raised);
        horizontalLayout_6 = new QHBoxLayout(frameCellName);
        horizontalLayout_6->setObjectName(QString::fromUtf8("horizontalLayout_6"));
        horizontalLayout_6->setContentsMargins(0, 0, 0, 0);
        labelCellName = new QLabel(frameCellName);
        labelCellName->setObjectName(QString::fromUtf8("labelCellName"));

        horizontalLayout_6->addWidget(labelCellName);

        lineCellName = new QLineEdit(frameCellName);
        lineCellName->setObjectName(QString::fromUtf8("lineCellName"));
        lineCellName->setStyleSheet(QString::fromUtf8("QLineEdit {\n"
"border: 1px solid dimgray;\n"
"}"));

        horizontalLayout_6->addWidget(lineCellName);


        verticalLayout_7->addWidget(frameCellName);

        frameCellMask = new QFrame(CellShapeEditor);
        frameCellMask->setObjectName(QString::fromUtf8("frameCellMask"));
        sizePolicy.setHeightForWidth(frameCellMask->sizePolicy().hasHeightForWidth());
        frameCellMask->setSizePolicy(sizePolicy);
        frameCellMask->setFrameShape(QFrame::StyledPanel);
        frameCellMask->setFrameShadow(QFrame::Raised);
        horizontalLayout_8 = new QHBoxLayout(frameCellMask);
        horizontalLayout_8->setObjectName(QString::fromUtf8("horizontalLayout_8"));
        horizontalLayout_8->setContentsMargins(0, 0, 0, 0);
        labelCellMask = new QLabel(frameCellMask);
        labelCellMask->setObjectName(QString::fromUtf8("labelCellMask"));
        QSizePolicy sizePolicy1(QSizePolicy::Fixed, QSizePolicy::Fixed);
        sizePolicy1.setHorizontalStretch(0);
        sizePolicy1.setVerticalStretch(0);
        sizePolicy1.setHeightForWidth(labelCellMask->sizePolicy().hasHeightForWidth());
        labelCellMask->setSizePolicy(sizePolicy1);

        horizontalLayout_8->addWidget(labelCellMask);

        lineCellMaskPath = new QLineEdit(frameCellMask);
        lineCellMaskPath->setObjectName(QString::fromUtf8("lineCellMaskPath"));
        lineCellMaskPath->setStyleSheet(QString::fromUtf8("QLineEdit {\n"
"border: 1px solid dimgray;\n"
"}"));
        lineCellMaskPath->setReadOnly(true);
        lineCellMaskPath->setCursorMoveStyle(Qt::LogicalMoveStyle);
        lineCellMaskPath->setClearButtonEnabled(false);

        horizontalLayout_8->addWidget(lineCellMaskPath);

        buttonCellMask = new QPushButton(frameCellMask);
        buttonCellMask->setObjectName(QString::fromUtf8("buttonCellMask"));
        QSizePolicy sizePolicy2(QSizePolicy::Minimum, QSizePolicy::Minimum);
        sizePolicy2.setHorizontalStretch(0);
        sizePolicy2.setVerticalStretch(0);
        sizePolicy2.setHeightForWidth(buttonCellMask->sizePolicy().hasHeightForWidth());
        buttonCellMask->setSizePolicy(sizePolicy2);

        horizontalLayout_8->addWidget(buttonCellMask);


        verticalLayout_7->addWidget(frameCellMask);

        horizontalLine_3 = new QFrame(CellShapeEditor);
        horizontalLine_3->setObjectName(QString::fromUtf8("horizontalLine_3"));
        horizontalLine_3->setFrameShape(QFrame::HLine);
        horizontalLine_3->setFrameShadow(QFrame::Sunken);

        verticalLayout_7->addWidget(horizontalLine_3);

        frameCellShapeSettings = new QFrame(CellShapeEditor);
        frameCellShapeSettings->setObjectName(QString::fromUtf8("frameCellShapeSettings"));
        frameCellShapeSettings->setFrameShape(QFrame::StyledPanel);
        frameCellShapeSettings->setFrameShadow(QFrame::Raised);
        horizontalLayout = new QHBoxLayout(frameCellShapeSettings);
        horizontalLayout->setObjectName(QString::fromUtf8("horizontalLayout"));
        horizontalLayout->setContentsMargins(0, 0, 0, 0);
        gridCellSpacing = new QGridLayout();
        gridCellSpacing->setObjectName(QString::fromUtf8("gridCellSpacing"));
        labelRow = new QLabel(frameCellShapeSettings);
        labelRow->setObjectName(QString::fromUtf8("labelRow"));

        gridCellSpacing->addWidget(labelRow, 0, 1, 1, 1, Qt::AlignHCenter|Qt::AlignVCenter);

        spinCellSpacingRow = new QSpinBox(frameCellShapeSettings);
        spinCellSpacingRow->setObjectName(QString::fromUtf8("spinCellSpacingRow"));
        spinCellSpacingRow->setStyleSheet(QString::fromUtf8("QSpinBox {\n"
"border: 1px solid dimgray;\n"
"}"));
        spinCellSpacingRow->setButtonSymbols(QAbstractSpinBox::NoButtons);
        spinCellSpacingRow->setKeyboardTracking(false);
        spinCellSpacingRow->setMinimum(1);
        spinCellSpacingRow->setMaximum(10000);
        spinCellSpacingRow->setSingleStep(1);

        gridCellSpacing->addWidget(spinCellSpacingRow, 1, 1, 1, 1, Qt::AlignHCenter|Qt::AlignVCenter);

        labelSpacing = new QLabel(frameCellShapeSettings);
        labelSpacing->setObjectName(QString::fromUtf8("labelSpacing"));
        sizePolicy.setHeightForWidth(labelSpacing->sizePolicy().hasHeightForWidth());
        labelSpacing->setSizePolicy(sizePolicy);

        gridCellSpacing->addWidget(labelSpacing, 1, 0, 1, 1, Qt::AlignRight|Qt::AlignVCenter);

        spinCellAlternateOffsetRow = new QSpinBox(frameCellShapeSettings);
        spinCellAlternateOffsetRow->setObjectName(QString::fromUtf8("spinCellAlternateOffsetRow"));
        spinCellAlternateOffsetRow->setStyleSheet(QString::fromUtf8("QSpinBox {\n"
"border: 1px solid dimgray;\n"
"}"));
        spinCellAlternateOffsetRow->setButtonSymbols(QAbstractSpinBox::NoButtons);
        spinCellAlternateOffsetRow->setKeyboardTracking(false);
        spinCellAlternateOffsetRow->setMinimum(0);
        spinCellAlternateOffsetRow->setMaximum(10000);
        spinCellAlternateOffsetRow->setSingleStep(1);
        spinCellAlternateOffsetRow->setValue(0);

        gridCellSpacing->addWidget(spinCellAlternateOffsetRow, 2, 1, 1, 1, Qt::AlignHCenter|Qt::AlignVCenter);

        spinCellAlternateOffsetCol = new QSpinBox(frameCellShapeSettings);
        spinCellAlternateOffsetCol->setObjectName(QString::fromUtf8("spinCellAlternateOffsetCol"));
        spinCellAlternateOffsetCol->setStyleSheet(QString::fromUtf8("QSpinBox {\n"
"border: 1px solid dimgray;\n"
"}"));
        spinCellAlternateOffsetCol->setButtonSymbols(QAbstractSpinBox::NoButtons);
        spinCellAlternateOffsetCol->setKeyboardTracking(false);
        spinCellAlternateOffsetCol->setMinimum(0);
        spinCellAlternateOffsetCol->setMaximum(10000);
        spinCellAlternateOffsetCol->setSingleStep(1);

        gridCellSpacing->addWidget(spinCellAlternateOffsetCol, 2, 2, 1, 1, Qt::AlignHCenter|Qt::AlignVCenter);

        labelAlternateOffset = new QLabel(frameCellShapeSettings);
        labelAlternateOffset->setObjectName(QString::fromUtf8("labelAlternateOffset"));
        sizePolicy.setHeightForWidth(labelAlternateOffset->sizePolicy().hasHeightForWidth());
        labelAlternateOffset->setSizePolicy(sizePolicy);

        gridCellSpacing->addWidget(labelAlternateOffset, 2, 0, 1, 1, Qt::AlignRight|Qt::AlignVCenter);

        spinCellSpacingCol = new QSpinBox(frameCellShapeSettings);
        spinCellSpacingCol->setObjectName(QString::fromUtf8("spinCellSpacingCol"));
        spinCellSpacingCol->setStyleSheet(QString::fromUtf8("QSpinBox {\n"
"border: 1px solid dimgray;\n"
"}"));
        spinCellSpacingCol->setButtonSymbols(QAbstractSpinBox::NoButtons);
        spinCellSpacingCol->setKeyboardTracking(false);
        spinCellSpacingCol->setMinimum(1);
        spinCellSpacingCol->setMaximum(10000);
        spinCellSpacingCol->setSingleStep(1);

        gridCellSpacing->addWidget(spinCellSpacingCol, 1, 2, 1, 1, Qt::AlignHCenter|Qt::AlignVCenter);

        labelColumn = new QLabel(frameCellShapeSettings);
        labelColumn->setObjectName(QString::fromUtf8("labelColumn"));

        gridCellSpacing->addWidget(labelColumn, 0, 2, 1, 1, Qt::AlignHCenter|Qt::AlignVCenter);

        verticalSpacer_3 = new QSpacerItem(0, 0, QSizePolicy::Minimum, QSizePolicy::Minimum);

        gridCellSpacing->addItem(verticalSpacer_3, 3, 0, 1, 3);


        horizontalLayout->addLayout(gridCellSpacing);

        verticalLine_1 = new QFrame(frameCellShapeSettings);
        verticalLine_1->setObjectName(QString::fromUtf8("verticalLine_1"));
        verticalLine_1->setFrameShape(QFrame::VLine);
        verticalLine_1->setFrameShadow(QFrame::Sunken);

        horizontalLayout->addWidget(verticalLine_1);

        gridCellFlipping = new QGridLayout();
        gridCellFlipping->setObjectName(QString::fromUtf8("gridCellFlipping"));
        gridCellFlipping->setContentsMargins(0, 0, -1, -1);
        labelHorizontal = new QLabel(frameCellShapeSettings);
        labelHorizontal->setObjectName(QString::fromUtf8("labelHorizontal"));

        gridCellFlipping->addWidget(labelHorizontal, 0, 1, 1, 1, Qt::AlignHCenter|Qt::AlignVCenter);

        labelColumnFlipping = new QLabel(frameCellShapeSettings);
        labelColumnFlipping->setObjectName(QString::fromUtf8("labelColumnFlipping"));

        gridCellFlipping->addWidget(labelColumnFlipping, 2, 0, 1, 1, Qt::AlignRight|Qt::AlignVCenter);

        checkCellRowFlipH = new QCheckBox(frameCellShapeSettings);
        checkCellRowFlipH->setObjectName(QString::fromUtf8("checkCellRowFlipH"));
        checkCellRowFlipH->setEnabled(true);
        checkCellRowFlipH->setLayoutDirection(Qt::RightToLeft);
        checkCellRowFlipH->setStyleSheet(QString::fromUtf8(""));
        checkCellRowFlipH->setCheckable(true);

        gridCellFlipping->addWidget(checkCellRowFlipH, 1, 1, 1, 1, Qt::AlignHCenter|Qt::AlignVCenter);

        labelRowFlipping = new QLabel(frameCellShapeSettings);
        labelRowFlipping->setObjectName(QString::fromUtf8("labelRowFlipping"));
        labelRowFlipping->setEnabled(true);

        gridCellFlipping->addWidget(labelRowFlipping, 1, 0, 1, 1, Qt::AlignRight|Qt::AlignVCenter);

        labelVertical = new QLabel(frameCellShapeSettings);
        labelVertical->setObjectName(QString::fromUtf8("labelVertical"));

        gridCellFlipping->addWidget(labelVertical, 0, 2, 1, 1, Qt::AlignHCenter|Qt::AlignVCenter);

        checkCellColFlipH = new QCheckBox(frameCellShapeSettings);
        checkCellColFlipH->setObjectName(QString::fromUtf8("checkCellColFlipH"));
        checkCellColFlipH->setEnabled(true);
        checkCellColFlipH->setLayoutDirection(Qt::RightToLeft);
        checkCellColFlipH->setStyleSheet(QString::fromUtf8(""));
        checkCellColFlipH->setCheckable(true);

        gridCellFlipping->addWidget(checkCellColFlipH, 2, 1, 1, 1, Qt::AlignHCenter|Qt::AlignVCenter);

        checkCellRowFlipV = new QCheckBox(frameCellShapeSettings);
        checkCellRowFlipV->setObjectName(QString::fromUtf8("checkCellRowFlipV"));
        checkCellRowFlipV->setEnabled(true);
        checkCellRowFlipV->setLayoutDirection(Qt::RightToLeft);
        checkCellRowFlipV->setStyleSheet(QString::fromUtf8(""));

        gridCellFlipping->addWidget(checkCellRowFlipV, 1, 2, 1, 1, Qt::AlignHCenter|Qt::AlignVCenter);

        checkCellColFlipV = new QCheckBox(frameCellShapeSettings);
        checkCellColFlipV->setObjectName(QString::fromUtf8("checkCellColFlipV"));
        checkCellColFlipV->setEnabled(true);
        checkCellColFlipV->setLayoutDirection(Qt::RightToLeft);
        checkCellColFlipV->setStyleSheet(QString::fromUtf8(""));

        gridCellFlipping->addWidget(checkCellColFlipV, 2, 2, 1, 1, Qt::AlignHCenter|Qt::AlignVCenter);

        verticalSpacer_2 = new QSpacerItem(0, 8, QSizePolicy::Minimum, QSizePolicy::Minimum);

        gridCellFlipping->addItem(verticalSpacer_2, 3, 0, 1, 3);


        horizontalLayout->addLayout(gridCellFlipping);

        verticalLine_2 = new QFrame(frameCellShapeSettings);
        verticalLine_2->setObjectName(QString::fromUtf8("verticalLine_2"));
        verticalLine_2->setFrameShape(QFrame::VLine);
        verticalLine_2->setFrameShadow(QFrame::Sunken);

        horizontalLayout->addWidget(verticalLine_2);

        gridAlternateSpacing = new QGridLayout();
        gridAlternateSpacing->setObjectName(QString::fromUtf8("gridAlternateSpacing"));
        spinCellAlternateSpacingRow = new QSpinBox(frameCellShapeSettings);
        spinCellAlternateSpacingRow->setObjectName(QString::fromUtf8("spinCellAlternateSpacingRow"));
        spinCellAlternateSpacingRow->setEnabled(false);
        spinCellAlternateSpacingRow->setStyleSheet(QString::fromUtf8("QSpinBox {\n"
"border: 1px solid dimgray;\n"
"}\n"
"\n"
"QSpinBox:disabled {\n"
"	background-color: rgb(30, 30, 30);\n"
"	color: rgb(60, 60, 60);\n"
"	border-color: rgb(0, 0, 0);\n"
"}"));
        spinCellAlternateSpacingRow->setButtonSymbols(QAbstractSpinBox::NoButtons);
        spinCellAlternateSpacingRow->setKeyboardTracking(false);
        spinCellAlternateSpacingRow->setMinimum(0);
        spinCellAlternateSpacingRow->setMaximum(10000);
        spinCellAlternateSpacingRow->setSingleStep(1);

        gridAlternateSpacing->addWidget(spinCellAlternateSpacingRow, 1, 1, 1, 1, Qt::AlignHCenter|Qt::AlignVCenter);

        checkCellAlternateSpacingCol = new QCheckBox(frameCellShapeSettings);
        checkCellAlternateSpacingCol->setObjectName(QString::fromUtf8("checkCellAlternateSpacingCol"));
        checkCellAlternateSpacingCol->setLayoutDirection(Qt::LeftToRight);

        gridAlternateSpacing->addWidget(checkCellAlternateSpacingCol, 0, 2, 1, 1, Qt::AlignHCenter|Qt::AlignVCenter);

        checkCellAlternateSpacingRow = new QCheckBox(frameCellShapeSettings);
        checkCellAlternateSpacingRow->setObjectName(QString::fromUtf8("checkCellAlternateSpacingRow"));
        checkCellAlternateSpacingRow->setLayoutDirection(Qt::LeftToRight);

        gridAlternateSpacing->addWidget(checkCellAlternateSpacingRow, 0, 1, 1, 1, Qt::AlignHCenter|Qt::AlignVCenter);

        spinCellAlternateSpacingCol = new QSpinBox(frameCellShapeSettings);
        spinCellAlternateSpacingCol->setObjectName(QString::fromUtf8("spinCellAlternateSpacingCol"));
        spinCellAlternateSpacingCol->setEnabled(false);
        spinCellAlternateSpacingCol->setStyleSheet(QString::fromUtf8("QSpinBox {\n"
"border: 1px solid dimgray;\n"
"}\n"
"\n"
"QSpinBox:disabled {\n"
"	background-color: rgb(30, 30, 30);\n"
"	color: rgb(60, 60, 60);\n"
"	border-color: rgb(0, 0, 0);\n"
"}"));
        spinCellAlternateSpacingCol->setButtonSymbols(QAbstractSpinBox::NoButtons);
        spinCellAlternateSpacingCol->setKeyboardTracking(false);
        spinCellAlternateSpacingCol->setMinimum(0);
        spinCellAlternateSpacingCol->setMaximum(10000);
        spinCellAlternateSpacingCol->setSingleStep(1);

        gridAlternateSpacing->addWidget(spinCellAlternateSpacingCol, 1, 2, 1, 1, Qt::AlignHCenter|Qt::AlignVCenter);

        labelAlternateSpacing = new QLabel(frameCellShapeSettings);
        labelAlternateSpacing->setObjectName(QString::fromUtf8("labelAlternateSpacing"));

        gridAlternateSpacing->addWidget(labelAlternateSpacing, 1, 0, 1, 1, Qt::AlignRight|Qt::AlignVCenter);

        verticalSpacer = new QSpacerItem(0, 18, QSizePolicy::Minimum, QSizePolicy::Minimum);

        gridAlternateSpacing->addItem(verticalSpacer, 2, 0, 1, 3);


        horizontalLayout->addLayout(gridAlternateSpacing);

        verticalLine_3 = new QFrame(frameCellShapeSettings);
        verticalLine_3->setObjectName(QString::fromUtf8("verticalLine_3"));
        verticalLine_3->setFrameShape(QFrame::VLine);
        verticalLine_3->setFrameShadow(QFrame::Sunken);

        horizontalLayout->addWidget(verticalLine_3);

        horizontalSpacer_8 = new QSpacerItem(1, 10, QSizePolicy::Expanding, QSizePolicy::Minimum);

        horizontalLayout->addItem(horizontalSpacer_8);


        verticalLayout_7->addWidget(frameCellShapeSettings);

        horizontalLine_2 = new QFrame(CellShapeEditor);
        horizontalLine_2->setObjectName(QString::fromUtf8("horizontalLine_2"));
        horizontalLine_2->setFrameShape(QFrame::HLine);
        horizontalLine_2->setFrameShadow(QFrame::Sunken);

        verticalLayout_7->addWidget(horizontalLine_2);

        cellShapeViewer = new GridViewer(CellShapeEditor);
        cellShapeViewer->setObjectName(QString::fromUtf8("cellShapeViewer"));
        QSizePolicy sizePolicy3(QSizePolicy::Expanding, QSizePolicy::Expanding);
        sizePolicy3.setHorizontalStretch(0);
        sizePolicy3.setVerticalStretch(0);
        sizePolicy3.setHeightForWidth(cellShapeViewer->sizePolicy().hasHeightForWidth());
        cellShapeViewer->setSizePolicy(sizePolicy3);
        cellShapeViewer->setStyleSheet(QString::fromUtf8("background-color: rgb(25, 25, 25);"));

        verticalLayout_7->addWidget(cellShapeViewer);


        retranslateUi(CellShapeEditor);

        QMetaObject::connectSlotsByName(CellShapeEditor);
    } // setupUi

    void retranslateUi(QWidget *CellShapeEditor)
    {
#if QT_CONFIG(statustip)
        buttonSaveCell->setStatusTip(QCoreApplication::translate("CellShapeEditor", "Save Cell Shape to .mcs", nullptr));
#endif // QT_CONFIG(statustip)
#if QT_CONFIG(whatsthis)
        buttonSaveCell->setWhatsThis(QCoreApplication::translate("CellShapeEditor", "<html><head/><body><p>Opens a file dialog to allow user to save the current cell shape as a .mcs file.</p></body></html>", nullptr));
#endif // QT_CONFIG(whatsthis)
        buttonSaveCell->setText(QCoreApplication::translate("CellShapeEditor", "Save Cell Shape", nullptr));
#if QT_CONFIG(statustip)
        buttonLoadCell->setStatusTip(QCoreApplication::translate("CellShapeEditor", "Load a Cell Shape from .mcs", nullptr));
#endif // QT_CONFIG(statustip)
#if QT_CONFIG(whatsthis)
        buttonLoadCell->setWhatsThis(QCoreApplication::translate("CellShapeEditor", "<html><head/><body><p>Opens file dialog to select a .mcs file to load.</p></body></html>", nullptr));
#endif // QT_CONFIG(whatsthis)
        buttonLoadCell->setText(QCoreApplication::translate("CellShapeEditor", "Load Cell Shape", nullptr));
#if QT_CONFIG(statustip)
        labelCellName->setStatusTip(QCoreApplication::translate("CellShapeEditor", "Name of the Cell Shape", nullptr));
#endif // QT_CONFIG(statustip)
        labelCellName->setText(QCoreApplication::translate("CellShapeEditor", "Cell Shape Name:", nullptr));
#if QT_CONFIG(statustip)
        lineCellName->setStatusTip(QCoreApplication::translate("CellShapeEditor", "Name of the Cell Shape", nullptr));
#endif // QT_CONFIG(statustip)
#if QT_CONFIG(whatsthis)
        lineCellName->setWhatsThis(QCoreApplication::translate("CellShapeEditor", "<html><head/><body><p>Name of the Cell Shape. This is also used as the filename when saving the Cell Shape.</p></body></html>", nullptr));
#endif // QT_CONFIG(whatsthis)
        lineCellName->setPlaceholderText(QString());
#if QT_CONFIG(statustip)
        labelCellMask->setStatusTip(QCoreApplication::translate("CellShapeEditor", "Image file to be used as mask for Cell Shape", nullptr));
#endif // QT_CONFIG(statustip)
        labelCellMask->setText(QCoreApplication::translate("CellShapeEditor", "Cell Mask:", nullptr));
#if QT_CONFIG(statustip)
        lineCellMaskPath->setStatusTip(QCoreApplication::translate("CellShapeEditor", "Displays path and filename of cell mask image", nullptr));
#endif // QT_CONFIG(statustip)
#if QT_CONFIG(whatsthis)
        lineCellMaskPath->setWhatsThis(QCoreApplication::translate("CellShapeEditor", "<html><head/><body><p>Image is loaded as a greyscale and thresholded into binary image.</p><p>White is the active area, black is inactive.</p></body></html>", nullptr));
#endif // QT_CONFIG(whatsthis)
#if QT_CONFIG(statustip)
        buttonCellMask->setStatusTip(QCoreApplication::translate("CellShapeEditor", "Opens a file dialog to allow an image file to be chosen", nullptr));
#endif // QT_CONFIG(statustip)
#if QT_CONFIG(whatsthis)
        buttonCellMask->setWhatsThis(QCoreApplication::translate("CellShapeEditor", "<html><head/><body><p>Image is loaded as a greyscale and thresholded into binary image.</p><p>White is the active area, black is inactive.</p></body></html>", nullptr));
#endif // QT_CONFIG(whatsthis)
        buttonCellMask->setText(QCoreApplication::translate("CellShapeEditor", "Browse", nullptr));
        labelRow->setText(QCoreApplication::translate("CellShapeEditor", "Row", nullptr));
#if QT_CONFIG(statustip)
        spinCellSpacingRow->setStatusTip(QCoreApplication::translate("CellShapeEditor", "Controls spacing between rows of cells in pixels", nullptr));
#endif // QT_CONFIG(statustip)
        spinCellSpacingRow->setSuffix(QCoreApplication::translate("CellShapeEditor", "px", nullptr));
#if QT_CONFIG(statustip)
        labelSpacing->setStatusTip(QCoreApplication::translate("CellShapeEditor", "Controls the spacing between cells", nullptr));
#endif // QT_CONFIG(statustip)
#if QT_CONFIG(whatsthis)
        labelSpacing->setWhatsThis(QString());
#endif // QT_CONFIG(whatsthis)
        labelSpacing->setText(QCoreApplication::translate("CellShapeEditor", "Spacing:", nullptr));
#if QT_CONFIG(statustip)
        spinCellAlternateOffsetRow->setStatusTip(QCoreApplication::translate("CellShapeEditor", "Controls horizontal offset for alternate rows in pixels", nullptr));
#endif // QT_CONFIG(statustip)
        spinCellAlternateOffsetRow->setSuffix(QCoreApplication::translate("CellShapeEditor", "px", nullptr));
#if QT_CONFIG(statustip)
        spinCellAlternateOffsetCol->setStatusTip(QCoreApplication::translate("CellShapeEditor", "Controls vertical offset for alternate columns in pixel", nullptr));
#endif // QT_CONFIG(statustip)
        spinCellAlternateOffsetCol->setSuffix(QCoreApplication::translate("CellShapeEditor", "px", nullptr));
#if QT_CONFIG(statustip)
        labelAlternateOffset->setStatusTip(QCoreApplication::translate("CellShapeEditor", "Controls the offset applied to every other cell", nullptr));
#endif // QT_CONFIG(statustip)
        labelAlternateOffset->setText(QCoreApplication::translate("CellShapeEditor", "Alternate Offset:", nullptr));
#if QT_CONFIG(statustip)
        spinCellSpacingCol->setStatusTip(QCoreApplication::translate("CellShapeEditor", "Controls spacing between columns of cells in pixels", nullptr));
#endif // QT_CONFIG(statustip)
        spinCellSpacingCol->setSuffix(QCoreApplication::translate("CellShapeEditor", "px", nullptr));
        labelColumn->setText(QCoreApplication::translate("CellShapeEditor", "Column", nullptr));
#if QT_CONFIG(statustip)
        labelHorizontal->setStatusTip(QCoreApplication::translate("CellShapeEditor", "Controls horizontal cell shape flipping", nullptr));
#endif // QT_CONFIG(statustip)
        labelHorizontal->setText(QCoreApplication::translate("CellShapeEditor", "Horizontal", nullptr));
#if QT_CONFIG(statustip)
        labelColumnFlipping->setStatusTip(QCoreApplication::translate("CellShapeEditor", "Controls cell shape column flipping", nullptr));
#endif // QT_CONFIG(statustip)
        labelColumnFlipping->setText(QCoreApplication::translate("CellShapeEditor", "Column Flipping:", nullptr));
#if QT_CONFIG(statustip)
        checkCellRowFlipH->setStatusTip(QCoreApplication::translate("CellShapeEditor", "Controls horizontal cell shape row flipping", nullptr));
#endif // QT_CONFIG(statustip)
#if QT_CONFIG(whatsthis)
        checkCellRowFlipH->setWhatsThis(QCoreApplication::translate("CellShapeEditor", "<html><head/><body><p>Every other row of cells will use a cell shape flipped horizontally.</p></body></html>", nullptr));
#endif // QT_CONFIG(whatsthis)
        checkCellRowFlipH->setText(QString());
#if QT_CONFIG(statustip)
        labelRowFlipping->setStatusTip(QCoreApplication::translate("CellShapeEditor", "Controls cell shape row flipping", nullptr));
#endif // QT_CONFIG(statustip)
        labelRowFlipping->setText(QCoreApplication::translate("CellShapeEditor", "Row Flipping:", nullptr));
#if QT_CONFIG(statustip)
        labelVertical->setStatusTip(QCoreApplication::translate("CellShapeEditor", "Controls vertical cell shape flipping", nullptr));
#endif // QT_CONFIG(statustip)
        labelVertical->setText(QCoreApplication::translate("CellShapeEditor", "Vertical", nullptr));
#if QT_CONFIG(statustip)
        checkCellColFlipH->setStatusTip(QCoreApplication::translate("CellShapeEditor", "Controls horizontal cell shape column flipping", nullptr));
#endif // QT_CONFIG(statustip)
#if QT_CONFIG(whatsthis)
        checkCellColFlipH->setWhatsThis(QCoreApplication::translate("CellShapeEditor", "<html><head/><body><p>Every other column of cells will use a cell shape flipped horizontally.</p></body></html>", nullptr));
#endif // QT_CONFIG(whatsthis)
        checkCellColFlipH->setText(QString());
#if QT_CONFIG(statustip)
        checkCellRowFlipV->setStatusTip(QCoreApplication::translate("CellShapeEditor", "Controls vertical cell shape row flipping", nullptr));
#endif // QT_CONFIG(statustip)
#if QT_CONFIG(whatsthis)
        checkCellRowFlipV->setWhatsThis(QCoreApplication::translate("CellShapeEditor", "<html><head/><body><p>Every other row of cells will use a cell shape flipped vertically.</p></body></html>", nullptr));
#endif // QT_CONFIG(whatsthis)
        checkCellRowFlipV->setText(QString());
#if QT_CONFIG(statustip)
        checkCellColFlipV->setStatusTip(QCoreApplication::translate("CellShapeEditor", "Controls vertical cell shape column flipping", nullptr));
#endif // QT_CONFIG(statustip)
#if QT_CONFIG(whatsthis)
        checkCellColFlipV->setWhatsThis(QCoreApplication::translate("CellShapeEditor", "<html><head/><body><p>Every other column of cells will use a cell shape flipped vertically.</p></body></html>", nullptr));
#endif // QT_CONFIG(whatsthis)
        checkCellColFlipV->setText(QString());
#if QT_CONFIG(statustip)
        spinCellAlternateSpacingRow->setStatusTip(QCoreApplication::translate("CellShapeEditor", "Controls spacing of alternate rows of cells in pixels", nullptr));
#endif // QT_CONFIG(statustip)
#if QT_CONFIG(whatsthis)
        spinCellAlternateSpacingRow->setWhatsThis(QCoreApplication::translate("CellShapeEditor", "<html><head/><body><p>When enabled every other row will use the alternate spacing instead of normal spacing.</p></body></html>", nullptr));
#endif // QT_CONFIG(whatsthis)
        spinCellAlternateSpacingRow->setSuffix(QCoreApplication::translate("CellShapeEditor", "px", nullptr));
#if QT_CONFIG(statustip)
        checkCellAlternateSpacingCol->setStatusTip(QCoreApplication::translate("CellShapeEditor", "Enables alternate column spacing", nullptr));
#endif // QT_CONFIG(statustip)
        checkCellAlternateSpacingCol->setText(QCoreApplication::translate("CellShapeEditor", "Column", nullptr));
#if QT_CONFIG(statustip)
        checkCellAlternateSpacingRow->setStatusTip(QCoreApplication::translate("CellShapeEditor", "Enables alternate row spacing", nullptr));
#endif // QT_CONFIG(statustip)
        checkCellAlternateSpacingRow->setText(QCoreApplication::translate("CellShapeEditor", "Row", nullptr));
#if QT_CONFIG(statustip)
        spinCellAlternateSpacingCol->setStatusTip(QCoreApplication::translate("CellShapeEditor", "Controls spacing of alternate rows of cells in pixels", nullptr));
#endif // QT_CONFIG(statustip)
#if QT_CONFIG(whatsthis)
        spinCellAlternateSpacingCol->setWhatsThis(QCoreApplication::translate("CellShapeEditor", "<html><head/><body><p>When enabled every other column will use the alternate spacing instead of normal spacing.</p></body></html>", nullptr));
#endif // QT_CONFIG(whatsthis)
        spinCellAlternateSpacingCol->setSuffix(QCoreApplication::translate("CellShapeEditor", "px", nullptr));
        labelAlternateSpacing->setText(QCoreApplication::translate("CellShapeEditor", "Alternate Spacing:", nullptr));
#if QT_CONFIG(statustip)
        cellShapeViewer->setStatusTip(QCoreApplication::translate("CellShapeEditor", "Displays a grid preview of the Cell Shape", nullptr));
#endif // QT_CONFIG(statustip)
#if QT_CONFIG(whatsthis)
        cellShapeViewer->setWhatsThis(QCoreApplication::translate("CellShapeEditor", "<html><head/><body><p>Displays a grid preview of the Cell Shape.</p><p>When the edge detect check box is enabled an edge-detected version of the cell shape is used to form the grid preview.</p></body></html>", nullptr));
#endif // QT_CONFIG(whatsthis)
        (void)CellShapeEditor;
    } // retranslateUi

};

namespace Ui {
    class CellShapeEditor: public Ui_CellShapeEditor {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_CELLSHAPEEDITOR_H
