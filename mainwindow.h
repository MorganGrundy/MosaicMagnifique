#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <memory>
#include <QMap>
#include <QListWidgetItem>
#include <QProgressBar>
#include <opencv2/core/mat.hpp>

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
    void selectCellMask();
    void cellColOffsetXChanged(int t_value);
    void cellColOffsetYChanged(int t_value);
    void cellRowOffsetXChanged(int t_value);
    void cellRowOffsetYChanged(int t_value);

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
    void enableCellShape(int t_state);
    void selectCellFolder();

    void generatePhotomosaic();

private:
    Ui::MainWindow *ui;
    QProgressBar *progressBar;

    cv::Mat cellMask;

    double photomosaicSizeRatio;

    int imageSize;
    QMap<QListWidgetItem, std::pair<cv::Mat, cv::Mat>> allImages;
};

//Outputs a OpenCV mat to a QDataStream
//Can be used to save a OpenCV mat to a file
QDataStream &operator<<(QDataStream &t_out, const cv::Mat &t_mat);

//Inputs a OpenCV mat from a QDataStream
//Can be used to load a OpenCV mat from a file
QDataStream &operator>>(QDataStream &t_in, cv::Mat &t_mat);
#endif // MAINWINDOW_H
