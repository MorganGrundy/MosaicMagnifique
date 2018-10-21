#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>

#include <sys/ioctl.h>
#include <cstdio>
#include <unistd.h>

#include "shared.hpp"

using namespace cv;
using namespace std;

// Resizes input image (img) such that
// (height = targetHeight && width >= targetWidth) || (height >= targetHeight && width = targetWidth)
// and puts the resized image in result
void resizeImageExclusive(Mat& img, Mat& result, int targetHeight, int targetWidth)
{
    //Calculates resize factor
    double resizeFactor = ((double) targetHeight / img.rows);
    if (targetWidth > resizeFactor * img.cols)
        resizeFactor = ((double) targetWidth / img.cols);

    //Resizes image
    if (resizeFactor < 1)
        resize(img, result, Size(round(resizeFactor * img.cols), round(resizeFactor * img.rows)), 0, 0, INTER_AREA);
    else if (resizeFactor > 1)
        resize(img, result, Size(round(resizeFactor * img.cols), round(resizeFactor * img.rows)), 0, 0, INTER_CUBIC);
    else
        result = img;
}

int main(int argc, char** argv)
{
    struct winsize w;
    ioctl(STDOUT_FILENO, TIOCGWINSZ, &w);

    if(argc < 3)
    {
        cout << argv[0] << " images_in_path images_out_path" << endl;
        return -1;
    }

    //Get list of photos in given folder
    vector<String> fn;
    String filepath(argv[1]);
    filepath += "/*.jpg";
    glob(filepath, fn, false); //Populates fn with all the .jpg at filepath

    cout << "Processing " << fn.size() << " images at size " << CELL_SIZE << ":" << endl;

    double t = getTickCount();

    //Read in images and preprocess them
    Mat temp_img;
    int max_cell_size = CELL_SIZE * (MAX_ZOOM / 100.0);
    for (size_t i = 0; i < fn.size(); i++)
    {
        temp_img = imread(fn[i], IMREAD_COLOR);
        resizeImageExclusive(temp_img, temp_img, max_cell_size, max_cell_size);
        String file_out = argv[2] + fn[i].substr(fn[i].find_last_of("/"), fn[i].length());
        imwrite(file_out, temp_img);
        progressBar(i, fn.size() - 1, w.ws_col);
    }
    t = (getTickCount() - t) / getTickFrequency();
    cout << "\nTime passed in seconds for read: " << t << endl;

    return 0;
}
