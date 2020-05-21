#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <memory>
#include <QMap>
#include <QListWidgetItem>
#include <QProgressBar>
#include <opencv2/core/mat.hpp>

#include "cellshape.h"

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
    //Custom Cell Shapes tab
    void saveCellShape();
    void loadCellShape();
    void cellNameChanged(const QString &text);
    void selectCellMask();
    void cellSpacingColChanged(int t_value);
    void cellSpacingRowChanged(int t_value);
    void cellAlternateColOffsetChanged(int t_value);
    void cellAlternateRowOffsetChanged(int t_value);
    void cellFlipHorizontalChanged(bool t_state);
    void cellFlipVerticalChanged(bool t_state);

    //Image Library tab
    void addImages();
    void deleteImages();
    void updateCellSize();
    void saveLibrary();
    void loadLibrary();

    //Generator Settings tab
    void selectMainImage();
    void photomosaicSizeLink();
    void photomosaicWidthChanged(int i);
    void photomosaicHeightChanged(int i);
    void loadImageSize();
    void enableCellShape(bool t_state);

    void generatePhotomosaic();

private:
    Ui::MainWindow *ui;
    QProgressBar *progressBar;

    double photomosaicSizeRatio;

    int imageSize;
    QMap<QListWidgetItem, std::pair<cv::Mat, cv::Mat>> allImages;

    cv::Mat mainImage;
};
#endif // MAINWINDOW_H
