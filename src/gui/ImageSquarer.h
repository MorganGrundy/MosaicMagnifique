#pragma once

#include <QMainWindow>
#include <QGraphicsScene>
#include <opencv2/core/mat.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/objdetect.hpp>

#include "CropGraphicsObject.h"

namespace Ui {
class ImageSquarer;
}

class ImageSquarer : public QMainWindow
{
    Q_OBJECT

public:
    explicit ImageSquarer(QWidget *parent = nullptr);
    ~ImageSquarer();

    //Allows user to crop input image
    //Return false if window was closed
    bool square(const cv::Mat &t_in, cv::Mat &t_out);

    //Clears scene
    void clear();

public slots:
    //Apply the entered crop to the current image
    void cropCurrentImage();
    //Skip the current image
    void skipCurrentImage();

signals:
    //Sends image crop
    void imageCrop(const cv::Rect &t_crop);
    //Sent when user closes window
    void windowClosed();

protected:
    //Triggers close signal
    void closeEvent(QCloseEvent *event) override;

private:
    Ui::ImageSquarer *ui;
    std::shared_ptr<QGraphicsScene> scene;
    std::shared_ptr<CropGraphicsObject> cropObject;
};