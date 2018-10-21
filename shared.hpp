#ifndef SHARED_HPP_
#define SHARED_HPP_

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <sys/ioctl.h>
#include <cstdio>
#include <unistd.h>
#include <boost/filesystem.hpp>

using namespace cv;
using namespace std;
using namespace boost::filesystem;

#define MAX_HEIGHT 1080
#define MAX_WIDTH 1920

#define CELL_SIZE 5

#define MIN_ZOOM 100
#define MAX_ZOOM 1000

#define REPEAT_RANGE 5
#define REPEAT_ADDITION 1

//Image formats that are always supported and some additional commons
String IMG_FORMATS_ARR[15] = {".bmp",".dib",".pbm",".pgm",".ppm",".pxm",".pnm",".sr",".ras",".hdr",".pic",".jpeg",".jpg",".jpe",".png"};
const vector<String> IMG_FORMATS(&IMG_FORMATS_ARR[0], &IMG_FORMATS_ARR[0] + 15);

void progressBar(int cur, int max, int width);

//Converts the given value in degrees into radians
double deg2Rad(double deg);

// Resizes input image (img) such that
// (height = targetHeight && width <= targetWidth) || (height <= targetHeight && width = targetWidth)
// and puts the resized image in result
void resizeImageInclusive(Mat& img, Mat& result, int targetHeight, int targetWidth);

// Resizes input image (img) such that
// (height = targetHeight && width >= targetWidth) || (height >= targetHeight && width = targetWidth)
// and puts the resized image in result
void resizeImageExclusive(Mat& img, Mat& result, int targetHeight, int targetWidth);

//Given filepath creates list of filepaths to accepted images in given filepath
bool read_image_names(path img_in_path, vector<String> *fn);

#endif //SHARED_HPP_
