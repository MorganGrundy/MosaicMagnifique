#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

using namespace cv;
using namespace std;

void populateRepeats(vector< vector<int> > gridIndex, int y, int x, vector<int> *repeats);

// Returns the index in images of the image with the least variance from main_img, using CIE76 colour difference
int findBestImageCIE76(Mat& main_img, vector<Mat> images, vector<int> repeats);

//Returns 2D vector of Mat that make up the best fitting images of main_img cells using CIE76 colour difference
vector< vector<Mat> > findBestImagesCIE76(Mat& main_img, vector<Mat>& images, vector<Mat>& imagesMax, int no_of_cell_x, int no_of_cell_y, int window_width);

// Returns the index in images of the image with the least variance from main_img, using CIE2000 colour difference
int findBestImageCIE2000(Mat& main_img, vector<Mat> images, vector<int> repeats);

//Returns 2D vector of Mat that make up the best fitting images of main_img cells using CIE2000 colour difference
vector< vector<Mat> > findBestImagesCIE2000(Mat& main_img, vector<Mat>& images, vector<Mat>& imagesMax, int no_of_cell_x, int no_of_cell_y, int window_width);
