#include "imagesquarer.h"
#include "ui_imagesquarer.h"

#include <QDebug>
#include <opencv2/imgproc.hpp>

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

    //Create feature detector
    m_featureDetector = cv::FastFeatureDetector::create();
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


//Crop image to square, such that maximum number of features in crop
//Returns false if no features found
bool ImageSquarer::squareToFeatures(const cv::Mat &t_in, cv::Mat &t_out)
{
    //Check for empty image
    if (t_in.empty())
    {
        qDebug() << Q_FUNC_INFO << "input image was empty.";
        return false;
    }

    //Check if image already square
    if (t_in.rows == t_in.cols)
    {
        t_out = t_in;
        return true;
    }

    //Detect features
    std::vector<cv::KeyPoint> keyPoints;
    m_featureDetector->detect(t_in, keyPoints);
    if (keyPoints.empty())
    {
        qDebug() << Q_FUNC_INFO << "detected no features in input image.";
        return false;
    }

    //Find shortest side, use as size of cropped image
    const int cropSize = std::min(t_in.rows,  t_in.cols);

    //Stores current best crop of image and it's feature count
    cv::Rect bestCrop;
    size_t bestCropFeatureCount = 0;
    //Distance between crop center and keypoint average
    double bestCropDistFromCenter = 0;
    //Find crop with highest feature count
    for (int cropOffset = 0; cropOffset + cropSize < std::max(t_in.rows, t_in.cols);
         cropOffset += 4)
    {
        const cv::Rect crop((t_in.rows > t_in.cols) ? cv::Point(0, cropOffset)
                                                    : cv::Point(cropOffset, 0),
                            cv::Size(cropSize, cropSize));

        //Calculate average position of all keypoints in crop
        cv::Point2f keypointAverage(0, 0);
        //Count features in crop
        size_t cropFeatureCount = 0;
        for (auto keyPoint : keyPoints)
        {
            if (crop.contains(keyPoint.pt))
            {
                keypointAverage += keyPoint.pt;
                ++cropFeatureCount;
            }
        }
        keypointAverage = cv::Point2f(keypointAverage.x / cropFeatureCount,
                                      keypointAverage.y / cropFeatureCount);
        //Calculate distance between keypoint average and crop center
        const double distFromCenter = std::sqrt(
            std::pow(keypointAverage.x - (crop.x + crop.width / 2), 2) +
            std::pow(keypointAverage.y - (crop.y + crop.height / 2), 2));

        //New best crop if more features, or equal features but average closer to crop center
        if (cropFeatureCount > bestCropFeatureCount ||
            (cropFeatureCount == bestCropFeatureCount && distFromCenter < bestCropDistFromCenter))
        {
            bestCropFeatureCount = cropFeatureCount;
            bestCropDistFromCenter = distFromCenter;
            bestCrop = crop;
        }
    }

    //Copy best crop of image to output
    t_out = t_in(bestCrop);
    return true;
}

//Crop image to square, such that maximum entropy in crop
bool ImageSquarer::squareToEntropy(const cv::Mat &t_in, cv::Mat &t_out)
{
    //Check for empty image
    if (t_in.empty())
    {
        qDebug() << Q_FUNC_INFO << "input image was empty.";
        return false;
    }

    //Check if image already square
    if (t_in.rows == t_in.cols)
    {
        t_out = t_in;
        return true;
    }

    //Find shortest side, use as size of cropped image
    const int cropSize = std::min(t_in.rows,  t_in.cols);
    //Checking every possible crop takes a long time, so only check some
    const int cropStepSize = cropSize / 16;

    //Stores current best crop of image and it's entropy value
    cv::Rect bestCrop;
    double bestCropEntropy = 0;
    //Find crop with highest entropy
    for (int cropOffset = 0; cropOffset + cropSize < std::max(t_in.rows, t_in.cols);
         cropOffset += cropStepSize)
    {
        const cv::Rect crop((t_in.rows > t_in.cols) ? cv::Point(0, cropOffset)
                                                    : cv::Point(cropOffset, 0),
                            cv::Size(cropSize, cropSize));

        double cropEntropy = ImageUtility::calculateEntropy(t_in(crop));

        //New best crop if higher entropy
        if (cropEntropy > bestCropEntropy)
        {
            bestCropEntropy = cropEntropy;
            bestCrop = crop;
        }
    }

    t_out = t_in(bestCrop);
    return true;
}

//Crop image to square, such that maximum number of objects in crop
bool ImageSquarer::squareToCascadeClassifier(const cv::Mat &t_in, cv::Mat &t_out)
{
    //Check for empty image
    if (t_in.empty())
    {
        qDebug() << Q_FUNC_INFO << "input image was empty.";
        return false;
    }

    //Check if image already square
    if (t_in.rows == t_in.cols)
    {
        t_out = t_in;
        return true;
    }

    //Check that cascade classifier is loaded
    if (m_cascadeClassifier.empty())
    {
        qDebug() << Q_FUNC_INFO << "Cascade classifier is empty.";
        return false;
    }

    //Find shortest side, use as size of cropped image
    const int cropSize = std::min(t_in.rows,  t_in.cols);

    //Convert image to grayscale
    cv::Mat gray;
    cv::cvtColor(t_in, gray, cv::COLOR_BGR2GRAY);
    cv::equalizeHist(gray, gray);

    //Detect objects
    std::vector<cv::Rect> objects;
    m_cascadeClassifier.detectMultiScale(gray, objects);
    if (objects.empty())
    {
        qDebug() << Q_FUNC_INFO << "detected no objects in input image.";
        return false;
    }

    //Stores current best crop of image, and it's badness value
    cv::Rect bestCrop;
    double bestCropBadnessValue = std::numeric_limits<double>::max();
    //Find crop with lowest badness value
    for (int cropOffset = 0; cropOffset + cropSize < std::max(t_in.rows, t_in.cols);
         cropOffset += 8)
    {
        const cv::Rect crop((t_in.rows > t_in.cols) ? cv::Point(0, cropOffset)
                                                    : cv::Point(cropOffset, 0),
                            cv::Size(cropSize, cropSize));

        //Calculate how well objects fit in crop
        double cropBadnessValue = (objects.empty()) ? std::numeric_limits<double>::max() : 0;
        for (auto object : objects)
        {
            //Calculate rect of object visible in crop
            const cv::Rect objectVisible = crop & object;

            //Calculate distance between object and crop center
            const cv::Point objectCenter(object.x + object.width / 2, object.y + object.height / 2);
            const cv::Point distFromCenter(crop.x + crop.width / 2 - objectCenter.x,
                                           crop.y + crop.height / 2 - objectCenter.y);

            //Calculate how well object fits in crop, scales with distance from center
            double objectBadnessValue = std::sqrt(std::pow(distFromCenter.x, 2) +
                                                  std::pow(distFromCenter.y, 2));

            //Increase badness value if object not fully visible in crop
            if (objectVisible.area() < object.area())
            {
                //Increase badness value even more if object not visible at all
                if (objectVisible.area() > 0)
                    objectBadnessValue *= 5.0 * (object.area() / objectVisible.area());
                else
                    objectBadnessValue *= 10.0 * object.area();
            }

            cropBadnessValue += objectBadnessValue;
        }

        //If badness value less than current best then new best crop
        if (cropBadnessValue < bestCropBadnessValue)
        {
            bestCropBadnessValue = cropBadnessValue;
            bestCrop = crop;
        }
    }

    //Copy best crop of image to output
    t_out = t_in(bestCrop);
    return true;
}

//Loads cascade classifier from file
bool ImageSquarer::loadCascade(const std::string &t_file)
{
    return m_cascadeClassifier.load(t_file);
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
