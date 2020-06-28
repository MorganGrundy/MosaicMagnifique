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

    //Check that cell is within a bound
    bool inBounds = false;
    for (auto it = t_bounds.cbegin(); it != t_bounds.cend() && !inBounds; ++it)
    {
        yStart = std::clamp(unboundedRect.y, it->y, it->br().y);
        yEnd = std::clamp(unboundedRect.br().y, it->y, it->br().y);
        xStart = std::clamp(unboundedRect.x, it->x, it->br().x);
        xEnd = std::clamp(unboundedRect.br().x, it->x, it->br().x);

        //Cell in bounds
        if (yStart != yEnd && xStart != xEnd)
            inBounds = true;
    }

    //Cell completely out of bounds, just skip
    if (!inBounds)
        return true;

    //Bound cell within grid dimensions
    yStart = std::clamp(unboundedRect.y, 0, t_grid.rows);
    yEnd = std::clamp(unboundedRect.br().y, 0, t_grid.rows);
    xStart = std::clamp(unboundedRect.x, 0, t_grid.cols);
    xEnd = std::clamp(unboundedRect.br().x, 0, t_grid.cols);

    const cv::Rect roi = cv::Rect(xStart, yStart, xEnd - xStart, yEnd - yStart);

    cv::Mat gridPart(t_grid, roi), edgeGridPart(t_edgeGrid, roi);

    //Calculate if and how current cell is flipped
    auto [flipHorizontal, flipVertical] = CellGrid::getFlipStateAt(t_cellShape,
            t_x, t_y, padGrid);

    //Create bounded mask
    const cv::Mat mask(getCellMask(t_step, flipHorizontal, flipVertical, false),
                       cv::Range(yStart - unboundedRect.y, yEnd - unboundedRect.y),
                       cv::Range(xStart - unboundedRect.x, xEnd - unboundedRect.x));

    if (!backImage.empty() && t_step < sizeSteps)
    {
        //If cell entropy exceeds threshold halve cell size
        const cv::Mat cellImage(backImage, roi);
        if (CellGrid::calculateEntropy(mask, cellImage) >= CellGrid::MAX_ENTROPY() * 0.7)
            return false;
    }

    //Create bounded edge mask
    const cv::Mat edgeMask(getCellMask(t_step, flipHorizontal, flipVertical, true),
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

    std::vector<cv::Mat> newGrid, newEdgeGrid;

    //Stores grid bounds starting with full grid size
    std::vector<GridBounds> bounds(2);
    //Determines which bound is active
    int activeBound = 0;
    bounds.at(activeBound).addBound(gridHeight, gridWidth);

    CellShape currentCellShape(cellShape);
    for (size_t step = 0; step <= sizeSteps && !bounds.at(activeBound).empty(); ++step)
    {
        //Create new grids
        newGrid.push_back(cv::Mat(gridHeight, gridWidth, CV_8UC1, cv::Scalar(0)));
        newEdgeGrid.push_back(cv::Mat(gridHeight, gridWidth, CV_8UC4, cv::Scalar(0, 0, 0, 0)));

        const cv::Point gridSize = CellGrid::calculateGridSize(currentCellShape,
                                                               gridWidth, gridHeight, padGrid);

        //Clear previous bounds
        bounds.at(!activeBound).clear();
        //Create all cells in grid
        for (int y = -padGrid; y < gridSize.y - padGrid; ++y)
        {
            for (int x = -padGrid; x < gridSize.x - padGrid; ++x)
            {
                if (!createCell(currentCellShape, x, y, newGrid.at(step), newEdgeGrid.at(step),
                                bounds.at(activeBound), step))
                {
                    //Get cell bounds
                    cv::Rect cellBounds = CellGrid::getRectAt(currentCellShape, x, y);

                    //Bound cell within grid dimensions
                    int yStart = std::clamp(cellBounds.y, 0, gridHeight);
                    int yEnd = std::clamp(cellBounds.br().y, 0, gridHeight);
                    int xStart = std::clamp(cellBounds.x, 0, gridWidth);
                    int xEnd = std::clamp(cellBounds.br().x, 0, gridWidth);

                    //Update cell bounds
                    cellBounds.y = yStart;
                    cellBounds.x = xStart;
                    cellBounds.height = yEnd - yStart;
                    cellBounds.width = xEnd - xStart;

                    //Add to inactive bounds
                    bounds.at(!activeBound).addBound(cellBounds);
                }
            }
        }

        //Swap active and inactive bounds
        activeBound = !activeBound;

        //New bounds
        if (!bounds.at(activeBound).empty())
        {
            bounds.at(activeBound).mergeBounds();

            //Split cell size
            currentCellShape = currentCellShape.resized(
                        currentCellShape.getCellMask(0, 0).cols / 2,
                        currentCellShape.getCellMask(0, 0).rows / 2);
        }
    }

    //Combine grids
    for (size_t i = newGrid.size() - 1; i > 0; --i)
    {
        cv::Mat mask;
        cv::bitwise_not(newGrid.at(i - 1), mask);

        cv::bitwise_or(newGrid.at(i - 1), newGrid.at(i), newGrid.at(i - 1), mask);
        cv::bitwise_or(newEdgeGrid.at(i - 1), newEdgeGrid.at(i), newEdgeGrid.at(i - 1), mask);
    }

    //Make black pixels transparent
    UtilityFuncs::matMakeTransparent(newGrid.at(0), newGrid.at(0), 0);

    grid = QImage(newGrid.at(0).data, gridWidth, gridHeight, static_cast<int>(newGrid.at(0).step),
                  QImage::Format_RGBA8888).copy();
    edgeGrid = QImage(newEdgeGrid.at(0).data, gridWidth, gridHeight,
                      static_cast<int>(newEdgeGrid.at(0).step), QImage::Format_RGBA8888).copy();
    update();
}

//Sets the cell shape
void GridViewer::setCellShape(const CellShape &t_cellShape)
{
    cellShape = t_cellShape;

    if (!cellShape.getCellMask(0, 0).empty())
    {
        getCellMask(0, false, false, false) = cellShape.getCellMask(0, 0);

        cv::Mat cellMask;
        UtilityFuncs::edgeDetect(cellShape.getCellMask(0, 0), cellMask);

        //Make black pixels transparent
        UtilityFuncs::matMakeTransparent(cellMask, getCellMask(0, false, false, true), 0);

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
            cv::Mat cellMask;
            UtilityFuncs::edgeDetect(getCellMask(step, false, false, false), cellMask);
            //Make black pixels transparent
            UtilityFuncs::matMakeTransparent(cellMask, getCellMask(step, false, false, true), 0);

            cv::flip(getCellMask(0, false, false, true), getCellMask(step, true, false, true), 1);
            cv::flip(getCellMask(0, false, false, true), getCellMask(step, false, true, true), 0);
            cv::flip(getCellMask(0, false, false, true), getCellMask(step, true, true, true), -1);
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
    if (background.isNull())
        updateGrid();
    else
        update();
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
