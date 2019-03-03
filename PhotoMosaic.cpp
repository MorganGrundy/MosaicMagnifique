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

Mat mosaic;

int lastZoom;
int cur_zoom = MIN_ZOOM;

int x_offset, y_offset;

bool fast_mode = true;

//Callback for trackbars
void showMosaic(int pos, void *userdata)
{
    //Calculates number of pixels needed for a border in y and x to prevent viewing exceeding image bounds
    int borderSizeY = floor((mosaic.rows * MIN_ZOOM) / (2 * cur_zoom)); // (image height / 2) / (zoom / min zoom)
    int borderSizeX = floor((mosaic.cols * MIN_ZOOM) / (2 * cur_zoom)); // (image width / 2) / (zoom / min zoom)

    //Wraps y and x offsets to be within the border
    int y_wrapped = wrap(y_offset, borderSizeY, mosaic.rows - borderSizeY);
    int x_wrapped = wrap(x_offset, borderSizeX, mosaic.cols - borderSizeX);

    //Calculates bounding box of current view
    int y_min = y_wrapped - (MIN_ZOOM * mosaic.rows) / (2 * cur_zoom);
    int y_max = y_wrapped + (MIN_ZOOM * mosaic.rows) / (2 * cur_zoom);
    int x_min = x_wrapped - (MIN_ZOOM * mosaic.cols) / (2 * cur_zoom);
    int x_max = x_wrapped + (MIN_ZOOM * mosaic.cols) / (2 * cur_zoom);

    //Creates new Mat that points to a subsection of the image, only whats in the bounding box
    Mat focusMosaic = mosaic(Range(y_min, y_max), Range(x_min, x_max));
    //Resizes subsection of image to fit window
    resizeImageInclusive(focusMosaic, focusMosaic, DISPLAY_HEIGHT, DISPLAY_WIDTH);
    imshow("Mosaic", focusMosaic); //Display mosaic
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
        copy(IMG_FORMATS.begin(), IMG_FORMATS.end()-1, ostream_iterator<string>(oss, ", "));
        oss << IMG_FORMATS.back();
        cout << oss.str() << endl << endl;

        //Outputs flags and descriptions
        cout << "Flags:" << endl;
        cout << "-info, -i: Extra information is output" << endl;
        cout << "-cie2000, -c: Switches colour difference algorithm from CIE76 to CIEDE2000 (more accurate, but slower, doesn't guarantee better result)" << endl;
        cout << "-cell_size x, -s x: Uses the integer in the next argument (x) as the cell size in pixels. Default: " << CELL_SIZE << endl;
        cout << "-repeat_range x, -rr x: Uses the integer in the next argument (x) as the range in cells that repeats will be looked for. Default: " << REPEAT_RANGE << endl;
        cout << "-repeat_addition x, -ra x: Uses the integer in the next argument (x) as the value to add to variant for each repeat in range. Default: " << REPEAT_ADDITION << endl;
        cout << "-cell_shape x, -cs x: Uses the string in the next argument (x) as the cell name in ./Cells/x. Default: " << CELL_SHAPE << endl;
        cout << "-mosaic_size x1 x2, -ms x1 x2: Uses the integer in x1 as image height and x2 as image width. Default: Original image size" << endl;

        return -1;
    }
    if (argc > 4) //Reads flags
    {
      for (int i = 4; i < argc; ++i)
      {
        string flag = argv[i];

        if (i + 2 < argc) //Flags that require three arguments
        {
          string x1 = argv[i + 1];
          string x2 = argv[i + 2];
          // MOSAIC SIZE
          if (flag == "-mosaic_size" || flag == "-ms")
          {
            try
            {
              if (stoi(x1) <= 0 || stoi(x2) <= 0)
              {
                cout << "Mosaic size must be positive integers" << endl;
                return -1;
              }
              TARGET_HEIGHT = stoi(x1);
              TARGET_WIDTH = stoi(x2);
              cout << x1 << " x " << x2 << endl;
              cout << TARGET_HEIGHT << " x " << TARGET_WIDTH << endl;
              i += 2;
              continue;
            }
            catch(exception const& e)
            {
              cout << "Mosaic size was not a valid integer" << endl;
              return -1;
            }
          }
        }

        if (flag == "-i" || flag == "-info")
          extraInfo = true;
        else if (flag == "-c" || flag == "-cie2000")
          fast_mode = false;
        else if (i + 1 < argc) //Flags that require two arguments
        {
          string x = argv[i + 1];
          // CELL SIZE
          if (flag == "-s" || flag == "-cell_size")
          {
            try
            {
              if (stoi(x) < MIN_CELL_SIZE)
              {
                  cout << "Minimum cell size is " << MIN_CELL_SIZE << endl;
                  return -1;
              }
              CELL_SIZE = stoi(x);
              MAX_CELL_SIZE = CELL_SIZE * (MAX_ZOOM / MIN_ZOOM);
              i++;
            }
            catch(exception const& e)
            {
              cout << "Cell size was not a valid integer" << endl;
              return -1;
            }
          }
          // REPEAT RANGE
          else if (flag == "-rr" || flag == "-repeat_range")
          {
            try
            {
              if (stoi(x) < MIN_REPEAT_RANGE)
              {
                cout << "Minimum repeat range is " << MIN_REPEAT_RANGE << endl;
                return -1;
              }
              REPEAT_RANGE = stoi(x);
              i++;
            }
            catch(exception const& e)
            {
              cout << "Repeat range was not a valid integer" << endl;
              return -1;
            }
          }
          // REPEAT ADDITION
          else if (flag == "-ra" || flag == "-repeat_addition")
          {
            try
            {
              REPEAT_ADDITION = stoi(x);
              i++;
            }
            catch(exception const& e)
            {
              cout << "Repeat addition was not a valid integer" << endl;
              return -1;
            }
          }
          // CELL SHAPE
          else if (flag == "-cs" || flag == "-cell_shape")
          {
            CELL_SHAPE = x;
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
    Mat originalIm = imread(imageName, IMREAD_COLOR);
    if (originalIm.empty()) // Check for invalid input
    {
        cout <<  "Could not open or find the image" << endl;
        return -1;
    }

    //If mosaic size not specified then set to original image size
    if (TARGET_HEIGHT == 0)
    {
        TARGET_HEIGHT = originalIm.rows;
        TARGET_WIDTH = originalIm.cols;
    }

    //Resizes main image and converts to Lab colour space
    Mat mainIm;
    cout << "Original size: " << originalIm.size() << endl;
    resizeImageInclusive(originalIm, mainIm, (TARGET_HEIGHT * MIN_ZOOM) / MAX_ZOOM, (TARGET_WIDTH * MIN_ZOOM) / MAX_ZOOM);
    cout << "Resized: " << mainIm.size() << endl;
    cvtColor(mainIm, mainIm, COLOR_BGR2Lab);

    //Resizes original image to fit in display window
    resizeImageInclusive(originalIm, originalIm, DISPLAY_HEIGHT, DISPLAY_WIDTH);

    //Get list of photos in given folder
    path img_in_path(argv[2]);
    vector<String> fn;
    if (!read_image_names(img_in_path, &fn))
      return 0;

    //Calculate number of cells
    int no_of_cell_x = round(mainIm.cols / (double)cellOffsetCmpX[1]);
    int no_of_cell_y = round(mainIm.rows / (double)cellOffsetCmpY[0]);

    if (extraInfo)
    {
        cout << "Cell size: " << CELL_SIZE << endl;

        cout << "Cell offset X: (" << cellOffsetX[0] << ", " << cellOffsetX[1] << ")" << endl;
        cout << "Cell offset Y: (" << cellOffsetY[0] << ", " << cellOffsetY[1] << ")" << endl;

        cout << "Cell cmp offset X: (" << cellOffsetCmpX[0] << ", " << cellOffsetCmpX[1] << ")" << endl;
        cout << "Cell cmp offset Y: (" << cellOffsetCmpY[0] << ", " << cellOffsetCmpY[1] << ")" << endl;

        cout << "Number of cells: (" << no_of_cell_x + padRows * 2 << ", " << no_of_cell_y + padCols * 2 << ")"<< endl;
    }

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
    Mat tmp_img;
    for (size_t i = 0; i < fn.size(); ++i)
    {
        //Reads image, resizes to min zoom and max zoom (min used for compare)
        images[i] = imread(fn[i], IMREAD_COLOR);
        imageToSquare(images[i]);
        if (CELL_SHAPE == "square")
            resizeImageExclusive(images[i], imagesMax[i], MAX_CELL_SIZE, MAX_CELL_SIZE);
        else
        {
            resizeImageExclusive(images[i], tmp_img, MAX_CELL_SIZE, MAX_CELL_SIZE);
            tmp_img.copyTo(imagesMax[i], cellMask);
        }
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
    if (CELL_SHAPE == "square")
    {
        //Combines all results into single image (mosaic)
        vector<Mat> mosaicRows(no_of_cell_y);
        for (int y = 0; y < no_of_cell_y; ++y)
          hconcat(result[y], mosaicRows[y]);
        vconcat(mosaicRows, mosaic);
    }
    else
    {
        mosaic = Mat(no_of_cell_y * cellOffsetY[0], no_of_cell_x * cellOffsetX[1], mainIm.type(), cvScalar(0));

        //For all cells
        for (int y = -padCols; y < no_of_cell_y + padCols; ++y)
        {
            for (int x = -padRows; x < no_of_cell_x + padRows; ++x)
            {
                //Cell y start position
                int yUnboundedStart = y * cellOffsetY[0] + ((abs(x % 2) == 1) ? cellOffsetX[0] : 0);
                //Cell bounded y start position (in mosaic area)
                int yStart = wrap(yUnboundedStart, 0, mosaic.rows - 1);

                //Cell y end position
                int yUnboundedEnd = y * cellOffsetY[0] + cellMask.rows + ((abs(x % 2) == 1) ? cellOffsetX[0] : 0);
                //Cell bounded y end position (in mosaic area)
                int yEnd = wrap(yUnboundedEnd, 0, mosaic.rows - 1);

                //Cell x start position
                int xUnboundedStart = x * cellOffsetX[1] + ((abs(y % 2) == 1) ? cellOffsetY[1] : 0);
                //Cell bounded x start position (in mosaic area)
                int xStart = wrap(xUnboundedStart, 0, mosaic.cols - 1);

                //Cell x end position
                int xUnboundedEnd = x * cellOffsetX[1] + cellMask.cols + ((abs(y % 2) == 1) ? cellOffsetY[1] : 0);
                //Cell bounded x end position (in mosaic area)
                int xEnd = wrap(xUnboundedEnd, 0, mosaic.cols - 1);

                //Cell completely out of bounds, just skip
                if (yStart == yEnd || xStart == xEnd)
                    continue;

                //Creates a mat that is the cell area actually visible in mosaic
                Mat cellBounded = result[y + padCols][x + padRows](Range(yStart - yUnboundedStart, yEnd - yUnboundedStart), Range(xStart - xUnboundedStart, xEnd - xUnboundedStart));

                //Creates mask bounded same as cell (so that size equals)
                Mat maskBounded = cellMask(Range(yStart - yUnboundedStart, yEnd - yUnboundedStart), Range(xStart - xUnboundedStart, xEnd - xUnboundedStart));

                //Copys cell to mosaic using mask
                cellBounded.copyTo(mosaic(Range(yStart, yEnd), Range(xStart, xEnd)), maskBounded);
            }
        }
    }

    t = (getTickCount() - t) / getTickFrequency();
    cout << "Time passed in seconds for concat: " << t << endl;

    //Writes mosaic result
    imwrite(img_out_path.string(), mosaic);
////////////////////////////////////////////////////////////////////////////////
////    DISPLAY RESULT
////////////////////////////////////////////////////////////////////////////////
    //Display original image
    namedWindow("Original", WINDOW_AUTOSIZE);
    imshow("Original", originalIm);

    resizeImageInclusive(mosaic, mosaic, DISPLAY_HEIGHT, DISPLAY_WIDTH);

    //Initialises x and y offset at center of image
    x_offset = mosaic.cols / 2;
    y_offset = mosaic.rows / 2;

    //Create window with trackbars for zoom and x, y offset
    namedWindow("Mosaic", WINDOW_AUTOSIZE);
    createTrackbar("Zoom %", "Mosaic", &cur_zoom, MAX_ZOOM, showMosaic);
    setTrackbarMin("Zoom %", "Mosaic", MIN_ZOOM);
    createTrackbar("X focus", "Mosaic", &x_offset, mosaic.cols, showMosaic);
    createTrackbar("Y focus", "Mosaic", &y_offset, mosaic.rows, showMosaic);

    showMosaic(0, NULL);

    waitKey(0); // Wait for a keystroke in the window
    destroyAllWindows();
    return 0;
}
