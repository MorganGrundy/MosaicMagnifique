#include "gridviewer.h"

#include <QPainter>
#include <QDebug>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <QWheelEvent>

GridViewer::GridViewer(QWidget *parent)
    : QWidget(parent), MIN_ZOOM{0.1}, MAX_ZOOM{10}, zoom{1}
{
    layout = new QGridLayout(this);

    label = new QLabel("Zoom:", this);
    label->setStyleSheet("QWidget {"
                         "background-color: rgb(60, 60, 60);"
                         "color: rgb(255, 255, 255);"
                         "border-color: rgb(0, 0, 0);"
                         "}");
    layout->addWidget(label, 0, 0);

    spinBox = new QDoubleSpinBox(this);
    spinBox->setStyleSheet("QWidget {"
                           "background-color: rgb(60, 60, 60);"
                           "color: rgb(255, 255, 255);"
                           "}"
                           "QDoubleSpinBox {"
                           "border: 1px solid dimgray;"
                           "}");
    spinBox->setRange(MIN_ZOOM * 100, MAX_ZOOM * 100);
    spinBox->setValue(zoom * 100);
    spinBox->setSuffix("%");
    spinBox->setButtonSymbols(QDoubleSpinBox::PlusMinus);
    connect(spinBox, SIGNAL(valueChanged(double)), this, SLOT(zoomChanged(double)));
    layout->addWidget(spinBox, 0, 1);

    hSpacer = new QSpacerItem(10, 10, QSizePolicy::Expanding, QSizePolicy::Minimum);
    layout->addItem(hSpacer, 0, 2);

    vSpacer = new QSpacerItem(10, 10, QSizePolicy::Minimum, QSizePolicy::Expanding);
    layout->addItem(vSpacer, 1, 0);
}

//Sets the cell mask
void GridViewer::setCellMask(const cv::Mat &t_mask)
{
    cv::Mat tmp = t_mask.clone();

    //Mask is binary
    if (t_mask.type() == CV_8UC1)
    {
        //Convert to RGBA
        cv::cvtColor(tmp, tmp, cv::COLOR_GRAY2RGBA);

        //Replace black with transparent
        int channels = tmp.channels();
        int nRows = tmp.rows;
        int nCols = tmp.cols * channels;
        if (tmp.isContinuous())
        {
            nCols *= nRows;
            nRows = 1;
        }

        uchar *p;
        for (int i = 0; i < nRows; ++i)
        {
            p = tmp.ptr<uchar>(i);
            for (int j = 0; j < nCols; j += channels)
            {
                if (p[j] == 0)
                    p[j+3] = 0;
            }
        }
        cellMask = tmp;
    }
    //Mask is RGBA
    else if (t_mask.type() == CV_8UC4)
        cellMask = tmp;
    else
        qDebug() << "GridViewer::setCellMask(const cv::Mat &) unsupported mask type";
}

//Generates grid preview
void GridViewer::updatePreview()
{
    //No cell mask, no grid
    if (cellMask.empty())
    {
        grid = QImage();
        return;
    }

    cv::Mat result(height() / zoom, width() / zoom, CV_8UC4, cv::Scalar(0, 0, 0, 0));
    const cv::Point gridSize(result.cols / colOffset.x + 1, result.rows / rowOffset.y + 1);

    //Create all cells in grid
    for (int x = 0; x < gridSize.x; ++x)
    {
        for (int y = 0; y < gridSize.y; ++y)
        {
            const cv::Rect roi = cv::Rect(x * colOffset.x + (y % 2 == 1) * rowOffset.x,
                         y * rowOffset.y + (x % 2 == 1) * colOffset.y,
                         cellMask.cols, cellMask.rows) & cv::Rect(0, 0, result.cols, result.rows);
            cv::Mat part(result, roi);
            cv::Mat mask(cellMask, cv::Rect(0, 0, part.cols, part.rows));
            cv::bitwise_or(part, mask, part);
        }
    }

    grid = QImage(result.data, result.cols, result.rows, static_cast<int>(result.step),
                  QImage::Format_RGBA8888).copy();
    update();
}

//Called when the spinbox value is changed, updates grid zoom
void GridViewer::zoomChanged(double t_value)
{
    zoom = t_value / 100.0;
    updatePreview();
}

//Displays grid
void GridViewer::paintEvent(QPaintEvent *event)
{
    QPainter painter(this);

    if (!grid.isNull())
        painter.drawImage(QRectF(QPointF(0,0), QSizeF(width(), height())), grid);
}

//Generate new grid with new size
void GridViewer::resizeEvent(QResizeEvent *event)
{
    updatePreview();
}

//Change zoom of grid preview based on mouse scrollwheel movement
//Ctrl is a modifier key that allows for faster zooming (x10)
void GridViewer::wheelEvent(QWheelEvent *event)
{
    zoom += event->delta() / ((event->modifiers().testFlag(Qt::ControlModifier)) ? 1200.0 : 12000.0);
    zoom = std::clamp(zoom, MIN_ZOOM, MAX_ZOOM);

    spinBox->blockSignals(true);
    spinBox->setValue(zoom * 100);
    spinBox->blockSignals(false);

    updatePreview();
}
