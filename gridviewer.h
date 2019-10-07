#ifndef GRIDVIEWER_H
#define GRIDVIEWER_H

#include <QWidget>
#include <opencv2/core/mat.hpp>

class GridViewer : public QWidget
{
    Q_OBJECT
public:
    explicit GridViewer(QWidget *parent = nullptr);

protected:
    void paintEvent(QPaintEvent *event);

private:
    cv::Mat cellMask;
};

#endif // GRIDVIEWER_H
