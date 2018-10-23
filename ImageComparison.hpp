#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

using namespace cv;
using namespace std;

// Returns the index in images of the image with the least variance from main_img, using CIE76 colour difference
int findBestImageCIE76(Mat& main_img, vector<Mat> images, vector<int> repeats);

// Returns the index in images of the image with the least variance from main_img, using CIE2000 colour difference
int findBestImageCIE2000(Mat& main_img, vector<Mat> images, vector<int> repeats);
