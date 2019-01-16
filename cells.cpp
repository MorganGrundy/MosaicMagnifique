#ifndef CELLS_CPP_
#define CELLS_CPP_

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <sys/ioctl.h>
#include <cstdio>
#include <unistd.h>
#include <boost/filesystem.hpp>

#include "shared.hpp"

using namespace std;
using namespace cv;
using namespace boost::filesystem;

Mat cellMask; //Bit mask for cell shape at max cell size
Mat cellMaskCmp; //Bit mask for cell shape at cell size
int cellOffsetX, cellOffsetY; //Offset to interjoin cells
int cellOffsetCmpX, cellOffsetCmpY; //Offset to interjoin cells

//Resizes img inclusively using INTER_NEAREST to targetSize
//Returns resize factor
double resizeInterNearest(Mat& img, Mat& result, int targetSize)
{
    //Calculates resize factor for max cell size
    double resizeFactor = ((double) targetSize / img.rows);
    if (targetSize < resizeFactor * img.cols)
        resizeFactor = ((double) targetSize / img.cols);

    //Resizes cell mask to max cell size
    resize(img, result, Size(resizeFactor * img.cols, resizeFactor * img.rows), 0, 0, INTER_NEAREST);
    return resizeFactor;
}

int loadCellShape()
{
    String cellMaskName, cellOffsetName;
    switch(CELL_SHAPE)
    {
        default: case 0: cellOffsetX = MAX_CELL_SIZE; cellOffsetY = MAX_CELL_SIZE; cellOffsetCmpX = CELL_SIZE; cellOffsetCmpY = CELL_SIZE; return 0; //Square
        case 1: cellMaskName = "./Cells/Hexagon.png";
                cellOffsetName = "./Cells/HexagonOffset.png";
                break; //Hexagon
    }

    //Loads and checks cell mask
    Mat tmpCellMask = imread(cellMaskName, IMREAD_COLOR);
    if (tmpCellMask.empty()) // Check for invalid input
    {
        cout <<  "Could not open or find the cell mask image" << endl;
        return -1;
    }
    cvtColor(tmpCellMask, tmpCellMask, COLOR_BGR2GRAY);
    //Get original cell mask size
    cellOffsetX = tmpCellMask.cols;
    cellOffsetY = tmpCellMask.rows;
    //Resizes cell mask to max cell size
    double resizeFactorMax = resizeInterNearest(tmpCellMask, cellMask, MAX_CELL_SIZE);

    //Resizes cell mask to cell size
    double resizeFactorMin = resizeInterNearest(tmpCellMask, cellMaskCmp, CELL_SIZE);

    bool not01 = false;
    int main_rows = cellMask.rows;
    int main_cols = cellMask.cols * cellMask.channels();
    uchar* p_main;
    for (int row = 0; row < main_rows; ++row)
    {
        p_main = cellMask.ptr<uchar>(row);
        for (int col = 0; col < main_cols * cellMask.channels(); ++col)
            if ((int) p_main[col] != 0 && (int) p_main[col] != 255)
                not01 = true;
    }
    if (cellMask.channels() != 1)
    {
        cout << "Mask channels not 1: " << cellMask.channels() << endl;
        return -1;
    }
    if (not01 == true)
    {
        cout << "Mask contains value other than 0 or 255" << endl;
        return -1;
    }

    //Loads and checks cell offset
    Mat cellOffset = imread(cellOffsetName, IMREAD_COLOR);
    if (cellOffset.empty()) // Check for invalid input
    {
        cout <<  "Could not open or find the cell offset image" << endl;
        return -1;
    }
    cellOffsetX = cellOffset.cols * resizeFactorMax;
    cellOffsetY = cellOffset.rows * resizeFactorMax;
    cellOffsetCmpX = cellOffset.cols * resizeFactorMin;
    cellOffsetCmpY = cellOffset.rows * resizeFactorMin;

    cout << "Cell offset: " << cellOffsetX << "," << cellOffsetY << endl;

    return 0;
}

#endif //CELLS_CPP_