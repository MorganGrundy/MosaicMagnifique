#pragma once

#include <QString>
#include <QCoreApplication>
#include <QMessageBox>

namespace Utility
{
    QString GetApplicationDir();

    enum class MsgType { FATAL = 0, CRITICAL = 1, DEBUG = 2, WARNING = 3, INFO = 4 };
}

class MessageBox : public QMessageBox
{
public:
    int exec() override;

    static StandardButton information(QWidget *parent, const QString &title,
        const QString &text, StandardButtons buttons = Ok,
        StandardButton defaultButton = NoButton);
    static StandardButton question(QWidget *parent, const QString &title,
        const QString &text, StandardButtons buttons = StandardButtons(Yes | No),
        StandardButton defaultButton = NoButton);
    static StandardButton warning(QWidget *parent, const QString &title,
        const QString &text, StandardButtons buttons = Ok,
        StandardButton defaultButton = NoButton);
    static StandardButton critical(QWidget *parent, const QString &title,
        const QString &text, StandardButtons buttons = Ok,
        StandardButton defaultButton = NoButton);
    static void about(QWidget *parent, const QString &title, const QString &text);
};