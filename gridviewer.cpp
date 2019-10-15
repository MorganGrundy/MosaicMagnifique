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

    labelZoom = new QLabel("Zoom:", this);
    labelZoom->setStyleSheet("QWidget {"
                             "background-color: rgb(60, 60, 60);"
                             "color: rgb(255, 255, 255);"
                             "border-color: rgb(0, 0, 0);"
                             "}");
    layout->addWidget(labelZoom, 0, 0);

    spinZoom = new QDoubleSpinBox(this);
    spinZoom->setStyleSheet("QWidget {"
                           "background-color: rgb(60, 60, 60);"
                           "color: rgb(255, 255, 255);"
                           "}"
                           "QDoubleSpinBox {"
                           "border: 1px solid dimgray;"
                           "}");
    spinZoom->setRange(MIN_ZOOM * 100, MAX_ZOOM * 100);
    spinZoom->setValue(zoom * 100);
    spinZoom->setSuffix("%");
    spinZoom->setButtonSymbols(QDoubleSpinBox::PlusMinus);
    connect(spinZoom, SIGNAL(valueChanged(double)), this, SLOT(zoomChanged(double)));
    layout->addWidget(spinZoom, 0, 1);

    checkEdgeDetect = new QCheckBox("Edge Detect:", this);
    checkEdgeDetect->setLayoutDirection(Qt::LayoutDirection::RightToLeft);
    checkEdgeDetect->setStyleSheet("QWidget {"
                                   "background-color: rgb(60, 60, 60);"
                                   "color: rgb(255, 255, 255);"
                                   "border-color: rgb(0, 0, 0);"
                                   "}");
    connect(checkEdgeDetect, SIGNAL(stateChanged(int)), this, SLOT(edgeDetectChanged(int)));
    layout->addWidget(checkEdgeDetect, 0, 2);

    hSpacer = new QSpacerItem(10, 10, QSizePolicy::Expanding, QSizePolicy::Minimum);
    layout->addItem(hSpacer, 0, 3);

    vSpacer = new QSpacerItem(10, 10, QSizePolicy::Minimum, QSizePolicy::Expanding);
    layout->addItem(vSpacer, 1, 0);
}

//Changes state of edge detection in grid preview
void GridViewer::setEdgeDetect(bool t_state)
{
    checkEdgeDetect->setChecked(t_state);
}

//Generates grid preview
void GridViewer::updatePreview()
{
    //No cell mask, no grid
    if (cellShape.getCellMask().empty())
    {
        grid = QImage();
        return;
    }

    cv::Mat result(height() / zoom, width() / zoom, CV_8UC4, cv::Scalar(0, 0, 0, 0));
    const cv::Point gridSize(result.cols / cellShape.getColSpacing() + 1,
                             result.rows / cellShape.getRowSpacing() + 1);

    //Create all cells in grid
    for (int x = 0; x < gridSize.x; ++x)
    {
        for (int y = 0; y < gridSize.y; ++y)
        {
            const cv::Rect roi = cv::Rect(x * cellShape.getColSpacing() +
                                          (y % 2 == 1) * cellShape.getAlternateRowOffset(),
                                          y * cellShape.getRowSpacing() +
                                          (x % 2 == 1) * cellShape.getAlternateColOffset(),
                                          cellShape.getCellMask().cols,
                                          cellShape.getCellMask().rows) & cv::Rect(0, 0,
                                                                                   result.cols,
                                                                                   result.rows);
            cv::Mat part(result, roi);
            cv::Mat mask;
            if (edgeDetectedCell.empty())
                mask = cv::Mat(cellShape.getCellMask(), cv::Rect(0, 0, part.cols, part.rows));
            else
                mask = cv::Mat(edgeDetectedCell, cv::Rect(0, 0, part.cols, part.rows));
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

//Called when the checkbox state changes
//If checked then edge detects cell mask and updates grid preview else forgets edge detected mask
void GridViewer::edgeDetectChanged(int t_state)
{
    if (t_state && !cellShape.getCellMask().empty())
    {
        //Add single pixel black transparent border to mask so that Canny cannot leave open edges
        cv::Mat maskWithBorder;
        cv::copyMakeBorder(cellShape.getCellMask(), maskWithBorder, 1, 1, 1, 1, cv::BORDER_CONSTANT,
                           cv::Scalar(0));
        //Use Canny to detect edge of cell mask and convert to RGBA
        cv::Mat result;
        cv::Canny(maskWithBorder, result, 100.0, 155.0);
        cv::cvtColor(result, result, cv::COLOR_GRAY2RGBA);

        //Make black pixels transparent
        int channels = result.channels();
        int nRows = result.rows;
        int nCols = result.cols * channels;
        if (result.isContinuous())
        {
            nCols *= nRows;
            nRows = 1;
        }

        uchar *p;
        for (int i = 0; i < nRows; ++i)
        {
            p = result.ptr<uchar>(i);
            for (int j = 0; j < nCols; j += channels)
            {
                if (p[j] == 0)
                    p[j+3] = 0;
            }
        }
        edgeDetectedCell = result;
    }
    else
        edgeDetectedCell.release();

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

    spinZoom->blockSignals(true);
    spinZoom->setValue(zoom * 100);
    spinZoom->blockSignals(false);

    updatePreview();
}
