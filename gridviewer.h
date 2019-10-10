#ifndef GRIDVIEWER_H
#define GRIDVIEWER_H

#include <QWidget>
#include <opencv2/core/mat.hpp>

class GridViewer : public QWidget
{
    Q_OBJECT
public:
    explicit GridViewer(QWidget *parent = nullptr);
    void setCellMask(const cv::Mat &t_mask);
    void updatePreview();

    cv::Point colOffset, rowOffset;
    QImage grid;


protected:
    void paintEvent(QPaintEvent *event) override;
    void resizeEvent(QResizeEvent *event) override;

    cv::Mat cellMask;
};

#endif // GRIDVIEWER_H
