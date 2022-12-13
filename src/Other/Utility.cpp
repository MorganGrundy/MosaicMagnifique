#pragma once

#include "Utility.h"

#include "Logger.h"

namespace Utility
{
    QString GetApplicationDir()
    {
        static QString applicationDir;
        //Only load the application dir once
        if (applicationDir.isEmpty())
        {
            //Application dir from QCoreApplication is empty, so assume we have no QCoreApplication
            if (QCoreApplication::applicationDirPath().isEmpty())
            {
                //Create a temporary QCoreApplication
                int argc = 0;
                QCoreApplication tmp(argc, 0);
                applicationDir = QCoreApplication::applicationDirPath();
            }
            else
                applicationDir = QCoreApplication::applicationDirPath();
        }
        return applicationDir;
    }
}

int MessageBox::exec()
{
    Utility::MsgType msgType;
    switch (icon())
    {
    case Icon::Information: msgType = Utility::MsgType::INFO; break;
    case Icon::Warning: msgType = Utility::MsgType::WARNING; break;
    case Icon::Critical: msgType = Utility::MsgType::CRITICAL; break;
    default: msgType = Utility::MsgType::INFO; break;
    }

    g_Logger.write(msgType, "Displaying message box \"" + windowTitle() + "\": \"" + text() + "\"");
    const int ret = QMessageBox::exec();
    g_Logger.write(msgType, "Message box closed.");
    return ret;
}

QMessageBox::StandardButton MessageBox::information(QWidget *parent, const QString &title, const QString &text, StandardButtons buttons, StandardButton defaultButton)
{
    g_Logger.write(Utility::MsgType::INFO, "Displaying message box \"" + title + "\": \"" + text + "\"");
    const StandardButton ret = QMessageBox::information(parent, title, text, buttons, defaultButton);
    g_Logger.write(Utility::MsgType::INFO, "Message box closed.");
    return ret;
}

QMessageBox::StandardButton MessageBox::question(QWidget *parent, const QString &title, const QString &text, StandardButtons buttons, StandardButton defaultButton)
{
    g_Logger.write(Utility::MsgType::INFO, "Displaying message box \"" + title + "\": \"" + text + "\"");
    const StandardButton ret = QMessageBox::question(parent, title, text, buttons, defaultButton);
    g_Logger.write(Utility::MsgType::INFO, "Message box closed.");
    return ret;
}

QMessageBox::StandardButton MessageBox::warning(QWidget *parent, const QString &title, const QString &text, StandardButtons buttons, StandardButton defaultButton)
{
    g_Logger.write(Utility::MsgType::INFO, "Displaying message box \"" + title + "\": \"" + text + "\"");
    const StandardButton ret = QMessageBox::warning(parent, title, text, buttons, defaultButton);
    g_Logger.write(Utility::MsgType::WARNING, "Message box closed.");
    return ret;
}

QMessageBox::StandardButton MessageBox::critical(QWidget *parent, const QString &title, const QString &text, StandardButtons buttons, StandardButton defaultButton)
{
    g_Logger.write(Utility::MsgType::INFO, "Displaying message box \"" + title + "\": \"" + text + "\"");
    const StandardButton ret = QMessageBox::critical(parent, title, text, buttons, defaultButton);
    g_Logger.write(Utility::MsgType::CRITICAL, "Message box closed.");
    return ret;
}

void MessageBox::about(QWidget *parent, const QString &title, const QString &text)
{
    g_Logger.write(Utility::MsgType::INFO, "Displaying about message box \"" + title + "\": \"" + text + "\"");
    QMessageBox::about(parent, title, text);
    g_Logger.write(Utility::MsgType::INFO, "About message box closed.");
}
