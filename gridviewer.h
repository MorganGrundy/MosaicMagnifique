#ifndef GRIDVIEWER_H
#define GRIDVIEWER_H

#include <QWidget>
#include <opencv2/core/mat.hpp>
#include <QGridLayout>
#include <QLabel>
#include <QDoubleSpinBox>
#include <QSpacerItem>

class GridViewer : public QWidget
{
    Q_OBJECT
public:
    explicit GridViewer(QWidget *parent = nullptr);
    void setCellMask(const cv::Mat &t_mask);
    void updatePreview();

    cv::Point colOffset, rowOffset;
    QImage grid;

public slots:
    void zoomChanged(double t_value);

protected:
    void paintEvent(QPaintEvent *event) override;
    void resizeEvent(QResizeEvent *event) override;
    void wheelEvent(QWheelEvent *event) override;

private:
    QGridLayout *layout;
    QLabel *label;
    QDoubleSpinBox *spinBox;
    QSpacerItem *hSpacer, *vSpacer;

    cv::Mat cellMask;

    const double MIN_ZOOM, MAX_ZOOM;
    double zoom;
};

#endif // GRIDVIEWER_H
