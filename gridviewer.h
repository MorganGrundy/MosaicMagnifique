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
    void updateGrid();

    void setCellShape(const CellShape &t_cellShape);
    CellShape &getCellShape();

    void setMinimumCellSize(const size_t t_size);

    void setBackground(const cv::Mat &t_background);

public slots:
    void zoomChanged(double t_value);
    void edgeDetectChanged(int t_state);

protected:
    void paintEvent(QPaintEvent *event) override;
    void resizeEvent(QResizeEvent *event) override;
    void wheelEvent(QWheelEvent *event) override;

private:
    void updateCell(const CellShape &t_cellShape, const int t_x, const int t_y,
                    cv::Mat &t_grid, cv::Mat &t_edgeGrid);

    QGridLayout *layout;
    QLabel *labelZoom;
    QDoubleSpinBox *spinZoom;
    QCheckBox *checkEdgeDetect;
    QSpacerItem *hSpacer, *vSpacer;

    CellShape cellShape;
    size_t minimumCellSize;

    cv::Mat backImage;
    QImage background;

    cv::Mat cell, cellFlippedH, cellFlippedV, cellFlippedHV;
    cv::Mat edgeCell, edgeCellFlippedH, edgeCellFlippedV, edgeCellFlippedHV;
    QImage grid, edgeGrid;

    const double MIN_ZOOM, MAX_ZOOM;
    double zoom;

    const int padGrid = 2;
};

#endif // GRIDVIEWER_H
