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

    void setSizeSteps(const size_t t_steps, const bool t_reset = false);

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
                    cv::Mat &t_grid, cv::Mat &t_edgeGrid, size_t t_step = 0);

    cv::Mat &getCellMask(size_t t_sizeStep, bool t_flipHorizontal, bool t_flipVertical,
                         bool t_edge);

    QGridLayout *layout;
    QLabel *labelZoom;
    QDoubleSpinBox *spinZoom;
    QCheckBox *checkEdgeDetect;
    QSpacerItem *hSpacer, *vSpacer;

    CellShape cellShape;
    size_t sizeSteps;

    cv::Mat backImage;
    QImage background;

    std::vector<std::vector<cv::Mat>> cells;
    std::vector<std::vector<cv::Mat>> edgeCells;

    QImage grid, edgeGrid;

    const double MIN_ZOOM, MAX_ZOOM;
    double zoom;

    const int padGrid = 2;
};

#endif // GRIDVIEWER_H
