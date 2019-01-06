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

Mat cellMask; //Bit mask for cell shape
int cellOffsetX, cellOffsetY; //Offset to interjoin cells

int loadCellShape()
{
    String cellMaskName, cellOffsetName;
    switch(CELL_SHAPE)
    {
        default: case 0: return 0; //Square
        case 1: cellMaskName = "./Cells/Hexagon.png";
                cellOffsetName = "./Cells/HexagonOffset.png";
                break; //Hexagon
    }

    //Loads and checks cell mask
    cellMask = imread(cellMaskName, IMREAD_UNCHANGED);
    if (cellMask.empty()) // Check for invalid input
    {
        cout <<  "Could not open or find the cell mask image" << endl;
        return -1;
    }
    //Get original cell mask size
    cellOffsetX = cellMask.cols;
    cellOffsetY = cellMask.rows;
    //resizeImageInclusive(cellMask, cellMask, CELL_SIZE, CELL_SIZE);
    //Calculates resize factor
    double resizeFactor = ((double) (MAX_ZOOM / MIN_ZOOM) * CELL_SIZE / cellMask.rows);
    if ((MAX_ZOOM / MIN_ZOOM) * CELL_SIZE < resizeFactor * cellMask.cols)
        resizeFactor = ((double) (MAX_ZOOM / MIN_ZOOM) * CELL_SIZE / cellMask.cols);

    //Resizes image
    resize(cellMask, cellMask, Size(resizeFactor * cellMask.cols, resizeFactor * cellMask.rows), 0, 0, INTER_NEAREST);

    //Loads and checks cell offset
    Mat cellOffset = imread(cellOffsetName, IMREAD_UNCHANGED);
    if (cellOffset.empty()) // Check for invalid input
    {
        cout <<  "Could not open or find the cell offset image" << endl;
        return -1;
    }
    cellOffsetX = (cellOffset.cols * cellMask.cols) / cellOffsetX;
    cellOffsetY = (cellOffset.rows * cellMask.rows) / cellOffsetY;

    cout << "Cell offset: " << cellOffsetX << "," << cellOffsetY << endl;

    imshow("Cell shape", cellMask);
    waitKey(0); // Wait for a keystroke in the window
    destroyAllWindows();

    return 0;
}

#endif //CELLS_CPP_
