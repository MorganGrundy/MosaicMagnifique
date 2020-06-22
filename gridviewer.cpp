#include "gridviewer.h"

#include <QPainter>
#include <QDebug>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <QWheelEvent>

#include "cellgrid.h"
#include "utilityfuncs.h"

GridViewer::GridViewer(QWidget *parent)
    : QWidget(parent), sizeSteps{0}, cells{std::vector<cv::Mat>(4)},
      edgeCells{std::vector<cv::Mat>(4)},
      MIN_ZOOM{0.5}, MAX_ZOOM{10}, zoom{1}
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
    checkEdgeDetect->setCheckState(Qt::Checked);
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

//Creates cell in grid at given position
//Returns false if cell entropy exceeds threshold
bool GridViewer::createCell(const CellShape &t_cellShape, const int t_x, const int t_y,
                            cv::Mat &t_grid, cv::Mat &t_edgeGrid, const GridBounds &t_bounds,
                            size_t t_step)
{
    const cv::Rect unboundedRect = CellGrid::getRectAt(t_cellShape, t_x, t_y);

    //Cell bounded positions (in background area)
    int yStart, yEnd, xStart, xEnd;

    bool inBounds = false;
    for (auto it = t_bounds.cbegin(); it != t_bounds.cend(); ++it)
    {
        yStart = std::clamp(unboundedRect.y, it->y, it->br().y);
        yEnd = std::clamp(unboundedRect.br().y, it->y, it->br().y);
        xStart = std::clamp(unboundedRect.x, it->x, it->br().x);
        xEnd = std::clamp(unboundedRect.br().x, it->x, it->br().x);

        //Cell in bounds
        if (yStart != yEnd && xStart != xEnd)
        {
            inBounds = true;
            break;
        }
    }

    //Cell completely out of bounds, just skip
    if (!inBounds)
        return true;

    const cv::Rect roi = cv::Rect(xStart, yStart, xEnd - xStart, yEnd - yStart);

    cv::Mat gridPart(t_grid, roi), edgeGridPart(t_edgeGrid, roi);

    //Calculate if and how current cell is flipped
    auto [flipHorizontal, flipVertical] = CellGrid::getFlipStateAt(t_cellShape,
            t_x, t_y, padGrid);

    //Create bounded mask
    cv::Mat mask(getCellMask(t_step, flipHorizontal, flipVertical, false),
                 cv::Range(yStart - unboundedRect.y, yEnd - unboundedRect.y),
                 cv::Range(xStart - unboundedRect.x, xEnd - unboundedRect.x));

    if (!backImage.empty() && t_step < sizeSteps)
    {
        //If cell entropy exceeds threshold halve cell size
        const cv::Mat cellImage(backImage, roi);
        if (CellGrid::calculateEntropy(mask, cellImage) >= CellGrid::MAX_ENTROPY() * 0.7)
        {
            fprintf(stderr, "Entropy (%i, %i)", t_x, t_y);

            return false;
        }
    }

    //Create bounded edge mask
    cv::Mat edgeMask(getCellMask(t_step, flipHorizontal, flipVertical, true),
                     cv::Range(yStart - unboundedRect.y, yEnd - unboundedRect.y),
                     cv::Range(xStart - unboundedRect.x, xEnd - unboundedRect.x));

    //Copy cell to grid
    cv::bitwise_or(gridPart, mask, gridPart);
    cv::bitwise_or(edgeGridPart, edgeMask, edgeGridPart);

    return true;
}

//Generates grid preview
void GridViewer::updateGrid()
{
    //No cell mask, no grid
    if (cellShape.getCellMask(0, 0).empty())
    {
        grid = QImage();
        return;
    }

    //Calculate grid size
    const int gridHeight = (!background.isNull()) ? background.height()
                                                  : static_cast<int>(height() / MIN_ZOOM);
    const int gridWidth = (!background.isNull()) ? background.width()
                                                 : static_cast<int>(width() / MIN_ZOOM);

    cv::Mat newGrid(gridHeight, gridWidth, CV_8UC4, cv::Scalar(0, 0, 0, 0));
    cv::Mat newEdgeGrid(gridHeight, gridWidth, CV_8UC4, cv::Scalar(0, 0, 0, 0));

    GridBounds bounds;
    bounds.addBound(gridHeight, gridWidth);

    const cv::Point gridSize = CellGrid::calculateGridSize(cellShape,
                                                           newGrid.cols, newGrid.rows, padGrid);

    //Create all cells in grid
    for (int y = -padGrid; y < gridSize.y - padGrid; ++y)
    {
        for (int x = -padGrid; x < gridSize.x - padGrid; ++x)
        {
            if (!createCell(cellShape, x, y, newGrid, newEdgeGrid, bounds))
            {
                //Add to vector
            }
        }
    }

    grid = QImage(newGrid.data, newGrid.cols, newGrid.rows, static_cast<int>(newGrid.step),
                  QImage::Format_RGBA8888).copy();
    edgeGrid = QImage(newEdgeGrid.data, newEdgeGrid.cols, newEdgeGrid.rows,
                      static_cast<int>(newEdgeGrid.step), QImage::Format_RGBA8888).copy();
    update();
}

//Sets the cell shape
void GridViewer::setCellShape(const CellShape &t_cellShape)
{
    cellShape = t_cellShape;

    if (!cellShape.getCellMask(0, 0).empty())
    {
        cv::Mat cellMask;
        //Converts cell mask to RGBA
        cv::cvtColor(cellShape.getCellMask(0, 0), cellMask, cv::COLOR_GRAY2RGBA);
        //Make black pixels transparent
        int channels = cellMask.channels();
        int nRows = cellMask.rows;
        int nCols = cellMask.cols * channels;
        if (cellMask.isContinuous())
        {
            nCols *= nRows;
            nRows = 1;
        }
        uchar *p;
        for (int i = 0; i < nRows; ++i)
        {
            p = cellMask.ptr<uchar>(i);
            for (int j = 0; j < nCols; j += channels)
            {
                if (p[j] == 0)
                    p[j+3] = 0;
            }
        }
        getCellMask(0, false, false, false) = cellMask;

        //Add single pixel black transparent border to mask so that Canny cannot leave open edges
        cv::Mat maskWithBorder;
        cv::copyMakeBorder(cellMask, maskWithBorder, 1, 1, 1, 1, cv::BORDER_CONSTANT,
                           cv::Scalar(0));
        //Use Canny to detect edge of cell mask and convert to RGBA
        cv::Canny(maskWithBorder, cellMask, 100.0, 155.0);
        cv::cvtColor(cellMask, cellMask, cv::COLOR_GRAY2RGBA);

        //Make black pixels transparent
        channels = cellMask.channels();
        nRows = cellMask.rows;
        nCols = cellMask.cols * channels;
        if (cellMask.isContinuous())
        {
            nCols *= nRows;
            nRows = 1;
        }
        for (int i = 0; i < nRows; ++i)
        {
            p = cellMask.ptr<uchar>(i);
            for (int j = 0; j < nCols; j += channels)
            {
                if (p[j] == 0)
                    p[j+3] = 0;
            }
        }
        getCellMask(0, false, false, true) = cellMask;

        //Create flipped cell and edge cell
        cv::flip(getCellMask(0, false, false, false), getCellMask(0, true, false, false), 1);
        cv::flip(getCellMask(0, false, false, true), getCellMask(0, true, false, true), 1);
        cv::flip(getCellMask(0, false, false, false), getCellMask(0, false, true, false), 0);
        cv::flip(getCellMask(0, false, false, true), getCellMask(0, false, true, true), 0);
        cv::flip(getCellMask(0, false, false, false), getCellMask(0, true, true, false), -1);
        cv::flip(getCellMask(0, false, false, true), getCellMask(0, true, true, true), -1);

        setSizeSteps(sizeSteps, true);
    }
    else
    {
        cells.clear();
        edgeCells.clear();
    }
}

//Returns reference to a cell mask
cv::Mat &GridViewer::getCellMask(size_t t_sizeStep, bool t_flipHorizontal, bool t_flipVertical,
                                 bool t_edge)
{
    if (t_edge)
        return edgeCells.at(t_sizeStep).at(t_flipHorizontal + t_flipVertical * 2);
    else
        return cells.at(t_sizeStep).at(t_flipHorizontal + t_flipVertical * 2);
}

//Returns a reference to the cell shape
CellShape &GridViewer::getCellShape()
{
    return cellShape;
}

//Sets the minimum cell size
void GridViewer::setSizeSteps(const size_t t_steps, const bool t_reset)
{
    //Resize vectors
    cells.resize(t_steps + 1, std::vector<cv::Mat>(4));
    edgeCells.resize(t_steps + 1, std::vector<cv::Mat>(4));

    //Create cell masks for each step size
    int cellSize = getCellMask(0, false, false, false).rows;
    for (size_t step = 1; step <= t_steps; ++step)
    {
        cellSize /= 2;

        //Only create if reset or is a new step size
        if (t_reset || step > sizeSteps)
        {
            //Create cell masks
            getCellMask(step, false, false, false) = UtilityFuncs::resizeImage(
                        getCellMask(0, false, false, false), cellSize, cellSize,
                        UtilityFuncs::ResizeType::EXCLUSIVE);

            cv::flip(getCellMask(step, false, false, false),
                     getCellMask(step, true, false, false), 1);
            cv::flip(getCellMask(step, false, false, false),
                     getCellMask(step, false, true, false), 0);
            cv::flip(getCellMask(step, false, false, false),
                     getCellMask(step, true, true, false), -1);

            //Create edge cell masks
            //Add single pixel black transparent border to mask so that Canny cannot leave open edges
            cv::Mat &newEdgeMask = getCellMask(step, false, false, true);
            cv::Mat maskWithBorder;
            cv::copyMakeBorder(getCellMask(step, false, false, false), maskWithBorder, 1, 1, 1, 1,
                               cv::BORDER_CONSTANT, cv::Scalar(0));
            //Use Canny to detect edge of cell mask and convert to RGBA
            cv::Canny(maskWithBorder, newEdgeMask, 100.0, 155.0);
            cv::cvtColor(newEdgeMask, newEdgeMask, cv::COLOR_GRAY2RGBA);

            //Make black pixels transparent
            int channels = newEdgeMask.channels();
            int nRows = newEdgeMask.rows;
            int nCols = newEdgeMask.cols * channels;
            if (newEdgeMask.isContinuous())
            {
                nCols *= nRows;
                nRows = 1;
            }
            uchar *p;
            for (int i = 0; i < nRows; ++i)
            {
                p = newEdgeMask.ptr<uchar>(i);
                for (int j = 0; j < nCols; j += channels)
                {
                    if (p[j] == 0)
                        p[j+3] = 0;
                }
            }

            cv::flip(newEdgeMask, getCellMask(step, true, false, true), 1);
            cv::flip(newEdgeMask, getCellMask(step, false, true, true), 0);
            cv::flip(newEdgeMask, getCellMask(step, true, true, true), -1);
        }
    }
    sizeSteps = t_steps;
}

//Sets the background image in grid
void GridViewer::setBackground(const cv::Mat &t_background)
{
    backImage = t_background;
    if (t_background.empty())
        background = QImage();
    else
    {
        cv::Mat inverted(t_background.rows, t_background.cols, t_background.type());
        cv::cvtColor(t_background, inverted, cv::COLOR_BGR2RGB);

        background = QImage(inverted.data, inverted.cols, inverted.rows,
                            static_cast<int>(inverted.step), QImage::Format_RGB888).copy();
    }
}

//Called when the spinbox value is changed, updates grid zoom
void GridViewer::zoomChanged(double t_value)
{
    zoom = t_value / 100.0;
    update();
}

//Called when the checkbox state changes
void GridViewer::edgeDetectChanged(int /*t_state*/)
{
    update();
}

//Displays grid
void GridViewer::paintEvent(QPaintEvent * /*event*/)
{
    QPainter painter(this);

    double ratio;
    QRectF sourceRect(0, 0, 0, 0);

    //Draw background
    if (!background.isNull())
    {
        //Calculate ratio between background and GridViewer size
        ratio = background.width() / static_cast<double>(width());
        if (height() * ratio < background.height())
            ratio = background.height() / static_cast<double>(height());

        sourceRect.setSize(QSizeF((ratio * width()) / zoom, (ratio * height()) / zoom));

        painter.drawImage(QRectF(QPointF(0,0), QSizeF(width(), height())), background, sourceRect);
    }
    else
        sourceRect.setSize(QSizeF(width() / zoom, height() / zoom));

    //Draw grid
    if (checkEdgeDetect->isChecked())
    {
        if (!edgeGrid.isNull())
        {
            QImage croppedGrid = edgeGrid.copy(0, 0, background.width(), background.height());
            painter.drawImage(QRectF(QPointF(0,0), QSizeF(width(), height())),
                              croppedGrid, sourceRect);
        }
    }
    else if (!grid.isNull())
    {
        QImage croppedGrid = grid.copy(0, 0, background.width(), background.height());
        painter.drawImage(QRectF(QPointF(0,0), QSizeF(width(), height())), croppedGrid, sourceRect);
    }
}

//Generate new grid with new size
void GridViewer::resizeEvent(QResizeEvent * /*event*/)
{
    updateGrid();
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

    update();
}
