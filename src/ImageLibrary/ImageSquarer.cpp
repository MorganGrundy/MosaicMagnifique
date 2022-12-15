#include "ImageSquarer.h"
#include "ui_ImageSquarer.h"

#include <QDebug>
#include <opencv2/imgproc.hpp>

#include "..\Other\ImageUtility.h"
#include "..\Other\Logger.h"

ImageSquarer::ImageSquarer(QWidget *parent) :
    QMainWindow(parent), ui(new Ui::ImageSquarer)
{
    LogInfo("Opened Image Squarer.");
    ui->setupUi(this);

    connect(ui->pushCrop, &QPushButton::released, this, &ImageSquarer::cropCurrentImage);
    connect(ui->pushSkip, &QPushButton::released, this, &ImageSquarer::skipCurrentImage);

    //Ensures view is centered
    Qt::Alignment alignment;
    alignment.setFlag(Qt::AlignmentFlag::AlignHCenter);
    alignment.setFlag(Qt::AlignmentFlag::AlignVCenter);
    ui->graphicsView->setAlignment(alignment);

    //Create scene
    scene = std::make_shared<QGraphicsScene>(ui->graphicsView);
}

ImageSquarer::~ImageSquarer()
{
    delete ui;
    LogInfo("Closed Image Squarer.");
}

//Allows user to crop input image
//Return false if window was closed
bool ImageSquarer::square(const cv::Mat &t_in, cv::Mat &t_out)
{
    //Clear scene
    cropObject.reset();
    scene->clear();

    //Update scene size
    scene->setSceneRect(0, 0, t_in.cols, t_in.rows);

    //Add image to scene
    const QPixmap pixmap = ImageUtility::matToQPixmap(t_in);
    scene->addPixmap(pixmap);

    //Add moveable crop to scene
    cropObject = std::make_shared<CropGraphicsObject>(t_in.cols, t_in.rows);
    scene->addItem(cropObject.get());

    //Set scene in graphics view
    ui->graphicsView->setScene(scene.get());

    //Display window
    QWidget::show();
    ui->graphicsView->fitInView(ui->graphicsView->sceneRect(), Qt::KeepAspectRatio);

    //Stores if image was cropped or skipped, stays false if window closed
    bool imageWasCropped = false;

    //Wait in loop till user crops image or closes window
    QEventLoop loop;
    //Image cropped or skipped, set output image
    const QMetaObject::Connection connCrop =
        connect(this, &ImageSquarer::imageCrop,
                [&t_in, &t_out, &loop, &imageWasCropped](const cv::Rect &t_crop)
                {
                    imageWasCropped = true;
                    t_out = t_in(t_crop);
                    loop.quit();
                });
    //Window closed, image not cropped
    const QMetaObject::Connection connCancel = connect(this, &ImageSquarer::windowClosed,
                                                       &loop, &QEventLoop::quit);
    loop.exec();

    //Remove connections after loop exits
    disconnect(connCrop);
    disconnect(connCancel);

    return imageWasCropped;
}

//Clears scene
void ImageSquarer::clear()
{
    cropObject.reset();
    scene->clear();
    ui->graphicsView->update();
}

//Apply the entered crop to the current image
void ImageSquarer::cropCurrentImage()
{
    const QRect crop = cropObject->getCrop();
    clear();
    emit imageCrop(cv::Rect(crop.x(), crop.y(), crop.width(), crop.height()));
}

//Skip the current image
void ImageSquarer::skipCurrentImage()
{
    clear();
    emit imageCrop(cv::Rect());
}

//Triggers close signal
void ImageSquarer::closeEvent([[maybe_unused]] QCloseEvent *event)
{
    emit windowClosed();
}
