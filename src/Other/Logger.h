#pragma once

#include <mutex>

#include <qdir.h>
#include <QtCore/qfile.h>
#include <QApplication>
#include <QtCore/qtextstream.h>
#include <QtCore/qdatetime.h>

#include "..\Resources\VersionInfo.h"


#define LogDebug(x) g_Logger.write(Logger::LogType::DEBUG, x)
#define LogInfo(x) g_Logger.write(Logger::LogType::INFO, x)
#define LogWarn(x) g_Logger.write(Logger::LogType::WARNING, x);
#define LogCritical(x) g_Logger.write(Logger::LogType::CRITICAL, x);
#define LogFatal(x) g_Logger.write(Logger::LogType::FATAL, x);

class Logger
{
public:
    enum class LogType { FATAL = 0, CRITICAL = 1, DEBUG = 2, WARNING = 3, INFO = 4 };

	explicit Logger()
    {
        QDir folder(GetApplicationDir() + "/Logs/");
        if (!folder.exists())
            folder.mkpath(".");

        m_logDirectory = folder.absolutePath() + "/";
        write(LogType::INFO, "Mosaic Magnifique " VERSION_STR);
    }

    ~Logger()
    {
        write(LogType::INFO, "Mosaic Magnifique Closing...");
    }

    void write(const LogType type, const QString &msg)
    {
        std::unique_lock lock(m_mutex);

        QFile file(GetName());

        if (!file.open(QIODevice::Append | QIODevice::Text))
            qFatal("Logger failed to open log file");

        QTextStream out(&file);

        //Write date time
        out << QDateTime::currentDateTime().toString("[[yyyy-MM-dd hh:mm:ss]]");

        //Write type
        switch (type)
        {
        case LogType::INFO: out << "[INFO] "; break;
        case LogType::WARNING: out << "[WARN] "; break;
        case LogType::DEBUG: out << "[DEBUG] "; break;
        case LogType::CRITICAL: out << "[CRITICAL] "; break;
        case LogType::FATAL: out << "[FATAL] "; break;
        }

        //Write message
        out << msg << Qt::endl;
    }

private:
    QString m_logDirectory;
    std::mutex m_mutex;

    QString GetName()
    {
        return m_logDirectory + QDateTime::currentDateTime().toString("yyyy-MM-dd") + ".txt";
    }

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
} g_Logger;