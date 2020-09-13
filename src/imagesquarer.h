#ifndef IMAGESQUARER_H
#define IMAGESQUARER_H

#include <QMainWindow>
#include <QGraphicsScene>
#include <opencv2/core/mat.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/objdetect.hpp>

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

    //Crop image to square, such that maximum number of features in crop
    //Returns false if no features found
    bool squareToFeatures(const cv::Mat &t_in, cv::Mat &t_out);

    //Crop image to square, such that maximum entropy in crop
    bool squareToEntropy(const cv::Mat &t_in, cv::Mat &t_out);

    //Crop image to square, such that maximum number of objects in crop
    bool squareToCascadeClassifier(const cv::Mat &t_in, cv::Mat &t_out);

    //Loads cascade classifier from file
    bool loadCascade(const std::string &t_file);

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

    //Pointer to feature detector
    std::shared_ptr<cv::FastFeatureDetector> m_featureDetector;

    //Cascade classifier for detecting faces, or other objects
    cv::CascadeClassifier m_cascadeClassifier;
};

#endif // IMAGESQUARER_H
