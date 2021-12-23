/********************************************************************************
** Form generated from reading UI file 'ImageSquarer.ui'
**
** Created by: Qt User Interface Compiler version 5.15.0
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef UI_IMAGESQUARER_H
#define UI_IMAGESQUARER_H

#include <QtCore/QVariant>
#include <QtWidgets/QApplication>
#include <QtWidgets/QGraphicsView>
#include <QtWidgets/QHBoxLayout>
#include <QtWidgets/QMainWindow>
#include <QtWidgets/QMenuBar>
#include <QtWidgets/QPushButton>
#include <QtWidgets/QSpacerItem>
#include <QtWidgets/QStatusBar>
#include <QtWidgets/QVBoxLayout>
#include <QtWidgets/QWidget>

QT_BEGIN_NAMESPACE

class Ui_ImageSquarer
{
public:
    QWidget *centralwidget;
    QVBoxLayout *verticalLayout;
    QGraphicsView *graphicsView;
    QHBoxLayout *horizontalLayout;
    QSpacerItem *horizontalSpacer;
    QPushButton *pushSkip;
    QPushButton *pushCrop;
    QMenuBar *menubar;
    QStatusBar *statusbar;

    void setupUi(QMainWindow *ImageSquarer)
    {
        if (ImageSquarer->objectName().isEmpty())
            ImageSquarer->setObjectName(QString::fromUtf8("ImageSquarer"));
        ImageSquarer->resize(1080, 720);
        ImageSquarer->setStyleSheet(QString::fromUtf8("QWidget {\n"
"background-color: rgb(60, 60, 60);\n"
"color: rgb(255, 255, 255);\n"
"border-color: rgb(0, 0, 0);\n"
"}"));
        centralwidget = new QWidget(ImageSquarer);
        centralwidget->setObjectName(QString::fromUtf8("centralwidget"));
        verticalLayout = new QVBoxLayout(centralwidget);
        verticalLayout->setObjectName(QString::fromUtf8("verticalLayout"));
        graphicsView = new QGraphicsView(centralwidget);
        graphicsView->setObjectName(QString::fromUtf8("graphicsView"));

        verticalLayout->addWidget(graphicsView);

        horizontalLayout = new QHBoxLayout();
        horizontalLayout->setObjectName(QString::fromUtf8("horizontalLayout"));
        horizontalLayout->setContentsMargins(-1, 0, -1, -1);
        horizontalSpacer = new QSpacerItem(40, 20, QSizePolicy::Expanding, QSizePolicy::Minimum);

        horizontalLayout->addItem(horizontalSpacer);

        pushSkip = new QPushButton(centralwidget);
        pushSkip->setObjectName(QString::fromUtf8("pushSkip"));

        horizontalLayout->addWidget(pushSkip);

        pushCrop = new QPushButton(centralwidget);
        pushCrop->setObjectName(QString::fromUtf8("pushCrop"));

        horizontalLayout->addWidget(pushCrop);


        verticalLayout->addLayout(horizontalLayout);

        ImageSquarer->setCentralWidget(centralwidget);
        menubar = new QMenuBar(ImageSquarer);
        menubar->setObjectName(QString::fromUtf8("menubar"));
        menubar->setGeometry(QRect(0, 0, 1080, 20));
        ImageSquarer->setMenuBar(menubar);
        statusbar = new QStatusBar(ImageSquarer);
        statusbar->setObjectName(QString::fromUtf8("statusbar"));
        ImageSquarer->setStatusBar(statusbar);

        retranslateUi(ImageSquarer);

        QMetaObject::connectSlotsByName(ImageSquarer);
    } // setupUi

    void retranslateUi(QMainWindow *ImageSquarer)
    {
        ImageSquarer->setWindowTitle(QCoreApplication::translate("ImageSquarer", "Image Squarer", nullptr));
        pushSkip->setText(QCoreApplication::translate("ImageSquarer", "Skip", nullptr));
        pushCrop->setText(QCoreApplication::translate("ImageSquarer", "Crop", nullptr));
    } // retranslateUi

};

namespace Ui {
    class ImageSquarer: public Ui_ImageSquarer {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_IMAGESQUARER_H
