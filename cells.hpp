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

extern Mat cellMask; //Bit mask for cell shape
extern int cellOffsetX, cellOffsetY; //Offset to interjoin cells

int loadCellShape();

#endif //CELLS_HPP_
