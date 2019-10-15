#ifndef GRIDVIEWER_H
#define GRIDVIEWER_H

#include <QWidget>
#include <opencv2/core/mat.hpp>
#include <QGridLayout>
#include <QLabel>
#include <QDoubleSpinBox>
#include <QCheckBox>
#include <QSpacerItem>

#include "cellshape.h"

class GridViewer : public QWidget
{
    Q_OBJECT
public:
    explicit GridViewer(QWidget *parent = nullptr);
    void setEdgeDetect(bool t_state);
    void updatePreview();

    CellShape cellShape;

public slots:
    void zoomChanged(double t_value);
    void edgeDetectChanged(int t_state);

protected:
    void paintEvent(QPaintEvent *event) override;
    void resizeEvent(QResizeEvent *event) override;
    void wheelEvent(QWheelEvent *event) override;

private:
    QGridLayout *layout;
    QLabel *labelZoom;
    QDoubleSpinBox *spinZoom;
    QCheckBox *checkEdgeDetect;
    QSpacerItem *hSpacer, *vSpacer;

    cv::Mat edgeDetectedCell;
    QImage grid;

    const double MIN_ZOOM, MAX_ZOOM;
    double zoom;
};

#endif // GRIDVIEWER_H
