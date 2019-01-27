#ifndef CELLS_HPP_
#define CELLS_HPP_

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

extern Mat cellMask; //Bit mask for cell shape at max cell size
extern Mat cellMaskCmp; //Bit mask for cell shape at cell size
extern int cellOffsetX [2], cellOffsetY [2]; //Offset to interjoin cells at max cell size
extern int cellOffsetCmpX [2], cellOffsetCmpY [2]; //Offset to interjoin cells at cell size

int loadCellShape();

#endif //CELLS_HPP_
