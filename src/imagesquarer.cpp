#include "imagesquarer.h"
#include "ui_imagesquarer.h"

#include <QDebug>

#include "imageutility.h"

ImageSquarer::ImageSquarer(QWidget *parent) :
    QMainWindow(parent), ui(new Ui::ImageSquarer)
{
    ui->setupUi(this);

    connect(ui->pushCrop, &QPushButton::released, this, &ImageSquarer::cropCurrentImage);
    connect(ui->pushSkip, &QPushButton::released, this, &ImageSquarer::skipCurrentImage);
    connect(ui->pushCancel, &QPushButton::released, this, &ImageSquarer::cancelCurrentImage);

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
}

//Show given image for cropping
void ImageSquarer::show(const cv::Mat &t_image)
{
    //Clear scene
    cropObject.reset();
    scene->clear();

    //Update scene size
    scene->setSceneRect(0, 0, t_image.cols, t_image.rows);

    //Add image to scene
    const QPixmap pixmap = ImageUtility::matToQPixmap(t_image);
    scene->addPixmap(pixmap);

    //Add crop to scene
    cropObject = std::make_shared<CropGraphicsObject>(t_image.cols, t_image.rows);
    scene->addItem(cropObject.get());

    //Set scene in graphics view
    ui->graphicsView->setScene(scene.get());

    //Display window
    QWidget::show();
    ui->graphicsView->fitInView(ui->graphicsView->sceneRect(), Qt::KeepAspectRatio);
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

//Cancel cropping of images
void ImageSquarer::cancelCurrentImage()
{
    clear();
    emit cancelCrop();
}

//Triggers cancel crop
void ImageSquarer::closeEvent([[maybe_unused]] QCloseEvent *event)
{
    emit cancelCrop();
}
