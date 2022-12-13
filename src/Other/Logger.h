#pragma once

#include <mutex>

#include <qdir.h>
#include <QtCore/qfile.h>
#include <QApplication>
#include <QtCore/qtextstream.h>
#include <QtCore/qdatetime.h>

#include "..\Resources\VersionInfo.h"
#include "Utility.h"

//Debug logging should only be done in debug builds
#ifdef _DEBUG
#define LogDebug(x) g_Logger.write(Utility::MsgType::DEBUG, x)
#else
#define LogDebug(x)  
#endif
#define LogInfo(x) g_Logger.write(Utility::MsgType::INFO, x)
#define LogWarn(x) g_Logger.write(Utility::MsgType::WARNING, x);
#define LogCritical(x) g_Logger.write(Utility::MsgType::CRITICAL, x);
#define LogFatal(x) g_Logger.write(Utility::MsgType::FATAL, x);

class Logger
{
public:
	explicit Logger()
    {
        QDir folder(Utility::GetApplicationDir() + "/Logs/");
        if (!folder.exists())
            folder.mkpath(".");

        m_logDirectory = folder.absolutePath() + "/";
        write(Utility::MsgType::INFO, "Mosaic Magnifique " VERSION_STR);
    }

    ~Logger()
    {
        write(Utility::MsgType::INFO, "Mosaic Magnifique Closing...");
    }

    void write(const Utility::MsgType type, const QString &msg)
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
        case Utility::MsgType::INFO: out << "[INFO] "; break;
        case Utility::MsgType::WARNING: out << "[WARN] "; break;
        case Utility::MsgType::DEBUG: out << "[DEBUG] "; break;
        case Utility::MsgType::CRITICAL: out << "[CRITICAL] "; break;
        case Utility::MsgType::FATAL: out << "[FATAL] "; break;
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
};

extern Logger g_Logger;