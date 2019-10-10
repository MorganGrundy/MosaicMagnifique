#include "gridviewer.h"

#include <QPainter>
#include <QDebug>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

GridViewer::GridViewer(QWidget *parent)
    : QWidget(parent) {}

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

    cv::Mat result(height(), width(), CV_8UC4, cv::Scalar(0, 0, 0, 0));
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
