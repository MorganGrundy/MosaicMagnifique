#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <memory>
#include <QMap>
#include <QListWidgetItem>
#include <QProgressBar>

QT_BEGIN_NAMESPACE
namespace Ui { class MainWindow; }
QT_END_NAMESPACE

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    MainWindow(QWidget *t_parent = nullptr);
    ~MainWindow();

public slots:
    void addImages();
    void deleteImages();
    void updateCellSize();
    void saveLibrary();
    void loadLibrary();

private:
    Ui::MainWindow *ui;
    QProgressBar *progressBar;

    QString imageTypes;
    int imageSize;
    QMap<QListWidgetItem, QPixmap> originalImages;
};
#endif // MAINWINDOW_H
