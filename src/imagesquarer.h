#ifndef IMAGESQUARER_H
#define IMAGESQUARER_H

#include <QMainWindow>
#include <QGraphicsScene>
#include <opencv2/core/mat.hpp>

#include "cropgraphicsobject.h"

namespace Ui {
class ImageSquarer;
}

class ImageSquarer : public QMainWindow
{
    Q_OBJECT

public:
    explicit ImageSquarer(QWidget *parent = nullptr);
    ~ImageSquarer();

    //Show given image for cropping
    void show(const cv::Mat &t_image);

    //Clears scene
    void clear();

public slots:
    //Apply the entered crop to the current image
    void cropCurrentImage();
    //Skip the current image
    void skipCurrentImage();
    //Cancel cropping of images
    void cancelCurrentImage();

signals:
    //Sends image crop
    void imageCrop(const cv::Rect &t_crop);
    //Sent when user cancels crop
    void cancelCrop();

protected:
    //Triggers cancel crop
    void closeEvent(QCloseEvent *event) override;

private:
    Ui::ImageSquarer *ui;
    std::shared_ptr<QGraphicsScene> scene;
    std::shared_ptr<CropGraphicsObject> cropObject;
};

#endif // IMAGESQUARER_H
