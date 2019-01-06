#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <sstream>
#include <iterator>
#include <string>
#include <cmath>
#include <climits>

#include "shared.hpp"
#include "ImageComparison.hpp"
#include "cells.hpp"

using namespace cv;
using namespace std;

int actual_height, actual_width;

Mat mosaic;

int lastZoom;
int cur_zoom = MIN_ZOOM;

int x_offset, y_offset;

bool fast_mode = false;

//Callback for trackbars
void showMosaic(int pos, void *userdata)
{
    //Calculates number of pixels needed for a border in y and x to prevent viewing exceeding image bounds
    int borderSizeY = (mosaic.rows * MIN_ZOOM) / (2 * cur_zoom); // (image height / 2) / (zoom / min zoom)
    int borderSizeX = (mosaic.cols * MIN_ZOOM) / (2 * cur_zoom); // (image width / 2) / (zoom / min zoom)

    //Wraps y and x offsets to be within the border
    int y_wrapped = wrap(y_offset * (MAX_ZOOM / MIN_ZOOM), borderSizeY, mosaic.rows - borderSizeY);
    int x_wrapped = wrap(x_offset * (MAX_ZOOM / MIN_ZOOM), borderSizeX, mosaic.cols - borderSizeX);

    //Calculates bounding box of current view
    int y_min = y_wrapped - (MIN_ZOOM * mosaic.rows) / (2 * cur_zoom);
    int y_max = y_wrapped + (MIN_ZOOM * mosaic.rows) / (2 * cur_zoom);
    int x_min = x_wrapped - (MIN_ZOOM * mosaic.cols) / (2 * cur_zoom);
    int x_max = x_wrapped + (MIN_ZOOM * mosaic.cols) / (2 * cur_zoom);

    //Creates new Mat that points to a subsection of the image, only whats in the bounding box
    Mat focusMosaic = mosaic(Range(y_min, y_max), Range(x_min, x_max));
    //Resizes subsection of image to fit window
    resizeImageInclusive(focusMosaic, focusMosaic, MAX_HEIGHT, MAX_WIDTH);
    imshow("Mosaic", focusMosaic); //Display mosaic
}

uchar encodeValueToPixel(int value)
{
  //int max8bit = pow(2, 8) - 1;
  return 0;
}

//Forms and writes photomosaic
void writePhotomosaic(vector<Mat> images, vector< vector<unsigned int> > mosaic)
{
  int imageSize = CELL_SIZE * MAX_ZOOM / MIN_ZOOM;
  int noImagesY = mosaic.size() % imageSize;
  int noImagesX = images.size() / noImagesY;

  int resultWidth = 1 + mosaic[0].size() + noImagesX * imageSize;

  Mat result(mosaic[0].size(), resultWidth, CV_8UC3);

  //result.ptr(0,0) = encodeValueToPixel(images.size());
}

int main(int argc, char** argv)
{
    //Reads args
    if(argc < 4)
    {
        cout << argv[0] << " path/to/main.image path/to/images path/to/result.image [-flags]" << endl << endl;

        //Outputs the accepted image types
        cout << "Accepted image types:" << endl;
        ostringstream oss;
        copy(IMG_FORMATS.begin(), IMG_FORMATS.end()-1, ostream_iterator<String>(oss, ", "));
        oss << IMG_FORMATS.back();
        cout << oss.str() << endl << endl;

        //Outputs flags and descriptions
        cout << "Flags:" << endl;
        cout << "-fast, -f: Switches colour difference algorithm from CIEDE2000 to CIE76 (less accurate, but faster)" << endl;
        cout << "-cell_size x, -s x: Uses the integer in the next argument (x) as the cell size in pixels. Default: " << CELL_SIZE << endl;
        cout << "-repeat_range x, -rr x: Uses the integer in the next argument (x) as the range in cells that repeats will be looked for. Default: " << REPEAT_RANGE << endl;
        cout << "-repeat_addition x, -ra x: Uses the integer in the next argument (x) as the value to add to variant for each repeat in range. Default: " << REPEAT_ADDITION << endl;
        cout << "-cell_shape x, -cs x: Uses the integer in the next argument (x) as the cell shape. 0 = square, 1 = hexagon. Default: " << CELL_SHAPE << endl;

        return -1;
    }
    if (argc > 4) //Reads flags
    {
      for (int i = 4; i < argc; ++i)
      {
        string flag = argv[i];
        if (flag == "-f" || flag == "-fast")
          fast_mode = true;
        else if (i + 1 < argc) //Flags that require two arguments
        {
          string other = argv[i + 1];
          if (flag == "-s" || flag == "-cell_size")
          {
            CELL_SIZE = stoi(other);
            MAX_CELL_SIZE = CELL_SIZE * (MAX_ZOOM / 100.0);
            i++;
          }
          else if (flag == "-rr" || flag == "-repeat_range")
          {
            REPEAT_RANGE = stoi(other);
            i++;
          }
          else if (flag == "-ra" || flag == "-repeat_addition")
          {
            REPEAT_ADDITION = stoi(other);
            i++;
          }
          else if (flag == "-cs" || flag == "-cell_shape")
          {
            CELL_SHAPE = stoi(other);
            i++;
          }
        }
      }
    }

    if (loadCellShape() == -1)
        return -1;

    path img_out_path(argv[3]);

    //Loads and checks main image
    String imageName = argv[1];
    Mat mainIm = imread(imageName, IMREAD_COLOR);
    if (mainIm.empty()) // Check for invalid input
    {
        cout <<  "Could not open or find the image" << endl;
        return -1;
    }

    //Resizes main image and converts to Lab colour space
    cout << "Original size: " << mainIm.size() << endl;
    resizeImageInclusive(mainIm, mainIm, MAX_HEIGHT, MAX_WIDTH);
    cout << "Resized: " << mainIm.size() << endl;
    cvtColor(mainIm, mainIm, COLOR_BGR2Lab);

    //Get list of photos in given folder
    path img_in_path(argv[2]);
    vector<String> fn;
    if (!read_image_names(img_in_path, &fn))
      return 0;

    //Calculate number of cells
    int no_of_cell_x = mainIm.cols / CELL_SIZE;
    int no_of_cell_y = mainIm.rows / CELL_SIZE;
    cout << "Number of cells: (" << no_of_cell_x << ", " << no_of_cell_y << ")"<< endl;
////////////////////////////////////////////////////////////////////////////////
////    READ + PREPROCESS IMAGES
////////////////////////////////////////////////////////////////////////////////
// Worst case complexity:
// O(Ni),
// where Ni = Number of images
    double t = getTickCount();

    //Read in images and preprocess them
    vector<Mat> images(fn.size());
    vector<Mat> imagesMax(fn.size());
    for (size_t i = 0; i < fn.size(); ++i)
    {
        //Reads image, resizes to min zoom and max zoom (min used for compare)
        images[i] = imread(fn[i], IMREAD_COLOR);
        imageToSquare(images[i]);
        resizeImageExclusive(images[i], imagesMax[i], MAX_CELL_SIZE, MAX_CELL_SIZE);
        resizeImageExclusive(imagesMax[i], images[i], CELL_SIZE, CELL_SIZE);

        //Converts to Lab colour space
        cvtColor(images[i], images[i], COLOR_BGR2Lab);
    }
    t = (getTickCount() - t) / getTickFrequency();
    cout << "Time passed in seconds for read: " << t << endl;
////////////////////////////////////////////////////////////////////////////////
////    CREATE MOSAIC PARTS
////////////////////////////////////////////////////////////////////////////////
// Worst case complexity:
// O(rows * cols ((REPEAT_RANGE^2) / 2) * Ni) ~
// O(rows * cols * Ni),
// where Ni = number of images,
// rows = main image rows
// cols = main image cols
    //Used to determine width of window for progress bar
    struct winsize w;
    ioctl(STDOUT_FILENO, TIOCGWINSZ, &w);

    t = getTickCount();
    //Creates 2D grid of image cells and splits main image between them
    //Finds best match for each cell and creates a 2D vector of the best matches
    vector< vector<Mat> > result;
    if (fast_mode)
      result = findBestImagesCIE76(mainIm, images, imagesMax, no_of_cell_x, no_of_cell_y, w.ws_col);
    else
      result = findBestImagesCIE2000(mainIm, images, imagesMax, no_of_cell_x, no_of_cell_y, w.ws_col);

    t = (getTickCount() - t) / getTickFrequency();
    cout << "Time passed in seconds for split & find: " << t << endl;
////////////////////////////////////////////////////////////////////////////////
////    COMBINE PARTS INTO MOSAIC
////////////////////////////////////////////////////////////////////////////////
    t = getTickCount();
    //Combines all results into single image (mosaic)
    vector<Mat> mosaicRows(no_of_cell_y);
    for (int y = 0; y < no_of_cell_y; ++y)
      hconcat(result[y], mosaicRows[y]);
    vconcat(mosaicRows, mosaic);

    t = (getTickCount() - t) / getTickFrequency();
    cout << "Time passed in seconds for concat: " << t << endl;

    //Writes mosaic result at reduced size
    {
      Mat mosaicSmall;
      resizeImageExclusive(mosaic, mosaicSmall, MAX_WIDTH * 2, MAX_HEIGHT * 2);
      imwrite(img_out_path.string(), mosaicSmall);
    }
////////////////////////////////////////////////////////////////////////////////
////    DISPLAY RESULT
////////////////////////////////////////////////////////////////////////////////
    //Display original image
    namedWindow("Original", WINDOW_AUTOSIZE);
    cvtColor(mainIm, mainIm, COLOR_Lab2BGR);
    imshow("Original", mainIm);

    //Calculates resize factor for image
    double resizeFactor = ((double) MAX_HEIGHT / mosaic.rows);
    if (MAX_WIDTH < resizeFactor * mosaic.cols)
        resizeFactor = ((double) MAX_WIDTH / mosaic.cols);

    //Calculates actual size of image
    actual_height = mosaic.rows * resizeFactor;
    actual_width = mosaic.cols * resizeFactor;

    //Initialises x and y offset at center of image
    x_offset = actual_width / 2;
    y_offset = actual_height / 2;

    //Create window with trackbars for zoom and x, y offset
    namedWindow("Mosaic", WINDOW_AUTOSIZE);
    createTrackbar("Zoom %", "Mosaic", &cur_zoom, MAX_ZOOM, showMosaic);
    setTrackbarMin("Zoom %", "Mosaic", MIN_ZOOM);
    createTrackbar("X focus", "Mosaic", &x_offset, actual_width, showMosaic);
    createTrackbar("Y focus", "Mosaic", &y_offset, actual_height, showMosaic);

    showMosaic(0, NULL);

    waitKey(0); // Wait for a keystroke in the window
    destroyAllWindows();
    return 0;
}
