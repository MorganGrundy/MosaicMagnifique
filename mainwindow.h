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
    void tabChanged(int t_index);

    //Custom Cell Shapes tab
    void saveCellShape();
    void loadCellShape();
    void cellNameChanged(const QString &text);
    void selectCellMask();
    //Cell spacing
    void cellSpacingColChanged(int t_value);
    void cellSpacingRowChanged(int t_value);
    //Cell alternate offset
    void cellAlternateColOffsetChanged(int t_value);
    void cellAlternateRowOffsetChanged(int t_value);
    //Cell flipping
    void cellColumnFlipHorizontalChanged(bool t_state);
    void cellColumnFlipVerticalChanged(bool t_state);
    void cellRowFlipHorizontalChanged(bool t_state);
    void cellRowFlipVerticalChanged(bool t_state);
    //Cell alternate spacing
    void enableCellAlternateRowSpacing(bool t_state);
    void enableCellAlternateColSpacing(bool t_state);
    void cellAlternateRowSpacingChanged(int t_value);
    void cellAlternateColSpacingChanged(int t_value);

    //Image Library tab
    void addImages();
    void deleteImages();
    void updateCellSize();
    void saveLibrary();
    void loadLibrary();

    //Generator Settings tab
    void selectMainImage();
    void compareColours();
    void photomosaicSizeLink();
    void photomosaicWidthChanged(int i);
    void photomosaicHeightChanged(int i);
    void loadImageSize();
    void enableCellShape(bool t_state);
    void CUDADeviceChanged(int t_index);

    void generatePhotomosaic();

private:
    Ui::MainWindow *ui;
    QProgressBar *progressBar;

    bool cellShapeChanged;

    double photomosaicSizeRatio;

    int imageSize;
    QMap<QListWidgetItem, std::pair<cv::Mat, cv::Mat>> allImages;

    cv::Mat mainImage;
};
#endif // MAINWINDOW_H
