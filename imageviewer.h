#ifndef IMAGEVIEWER_H
#define IMAGEVIEWER_H

#include <QMainWindow>

#include <opencv2/core/mat.hpp>

namespace Ui {
class ImageViewer;
}

class ImageViewer : public QMainWindow
{
    Q_OBJECT

public:
    explicit ImageViewer(QWidget *t_parent = nullptr);
    explicit ImageViewer(QWidget *t_parent, const cv::Mat &t_image, const double t_duration);
    ~ImageViewer();

public slots:
    //Allows user to save the Photomosaic as an image file
    void saveImage();

private:
    Ui::ImageViewer *ui;

    const cv::Mat image;
};

#endif // IMAGEVIEWER_H
