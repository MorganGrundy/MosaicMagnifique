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
#include "gridbounds.h"
#include "cellgrid.h"

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

    void setDetail(const int t_detail = 100, const bool t_reset = false);

    CellGrid::mosaicBestFit getGridState() const;

public slots:
    void zoomChanged(double t_value);
    void edgeDetectChanged(int t_state);

protected:
    void paintEvent(QPaintEvent *event) override;
    void resizeEvent(QResizeEvent *event) override;
    void wheelEvent(QWheelEvent *event) override;

private:
    bool createCell(const int t_x, const int t_y,
                    cv::Mat &t_grid, cv::Mat &t_edgeGrid, const GridBounds &t_bounds,
                    size_t t_step = 0);

    CellGrid::cellBestFit findCellState(const int x, const int y, const GridBounds &t_bounds,
                                        const size_t t_step = 0) const;

    void createGrid(const CellGrid::mosaicBestFit &states,
                    const int gridHeight, const int gridWidth);

    cv::Mat &getEdgeCell(size_t t_sizeStep, bool t_flipHorizontal, bool t_flipVertical);

    QGridLayout *layout;
    QLabel *labelZoom;
    QDoubleSpinBox *spinZoom;
    QCheckBox *checkEdgeDetect;
    QSpacerItem *hSpacer, *vSpacer;

    size_t sizeSteps;

    cv::Mat backImage, detailImage;
    QImage background;

    double detail;

    std::vector<CellShape> cells;
    std::vector<CellShape> detailCells;
    std::vector<std::vector<cv::Mat>> edgeCells;

    QImage grid, edgeGrid;

    const double MIN_ZOOM, MAX_ZOOM;
    double zoom;

    const int padGrid = 2;

    CellGrid::mosaicBestFit gridState;
};

#endif // GRIDVIEWER_H
