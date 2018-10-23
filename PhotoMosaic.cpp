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

using namespace cv;
using namespace std;

int actual_height, actual_width;

Mat mosaic;

int lastZoom;
int cur_zoom = MIN_ZOOM;

int x_offset, y_offset;

bool fast_mode = false;

// Returns the index in images of the image with the least variance from main_img
int findBestImage(Mat& main_img, vector<Mat> images, vector<int> repeats)
{
    int main_rows = main_img.rows;
    int main_cols = main_img.cols * main_img.channels();
    if (main_img.isContinuous())
    {
        main_cols *= main_rows;
        main_rows = 1;
    }

    //Index of current image with lowest variant
    int best_fit = -1;
    //Initial best_variant set higher than max variant
    long double best_variant = LDBL_MAX;
    for (int i = 0; i < (int) images.size(); ++i)
    {
        int im_rows = images[i].rows;

        int im_cols = images[i].cols;
        if (images[i].isContinuous())
        {
            im_cols *= im_rows;
            im_rows = 1;
        }

        uchar* p_main;
        uchar* p_im;
        long double variant = 0;
        //Calculates sum of difference between corresponding pixel values
        for (int row = 0; row < main_rows; ++row)
        {
            p_main = main_img.ptr<uchar>(row);
            p_im = images[i].ptr<uchar>(row);
            for (int col = 0; col < main_cols; col += main_img.channels())
            {
                //CIE76
                variant += sqrt(pow(p_main[col] - p_im[col], 2) +
                                pow(p_main[col + 1] - p_im[col + 1], 2) +
                                pow(p_main[col + 2] - p_im[col + 2], 2));
                /*
                double C1Star = sqrt(pow(p_main[col + 1], 2) + pow(p_main[col + 2], 2));
                double C2Star = sqrt(pow(p_im[col + 1], 2) + pow(p_im[col + 2], 2));

                double deltaLPrime = p_im[col] - p_main[col];
                double LDash = (p_main[col] + p_im[col]) / 2;
                double CDash = (C1Star + C2Star) / 2;

                double a1Prime = p_main[col + 1] + (p_main[col + 1] / 2) * (1 - sqrt(pow(CDash, 7) / (pow(CDash, 7) + pow(25, 7))));
                double a2Prime = p_im[col + 1] + (p_im[col + 1] / 2) * (1 - sqrt(pow(CDash, 7) / (pow(CDash, 7) + pow(25, 7))));

                double C1Prime = sqrt(pow(a1Prime, 2) + pow(p_main[col + 2], 2));
                double C2Prime = sqrt(pow(a2Prime, 2) + pow(p_im[col + 2], 2));

                double CDashPrime = (C1Prime + C2Prime) / 2;
                double deltaCPrime = C2Prime - C1Prime;

                double h1Prime = atan2(p_main[col + 2], a1Prime) + M_PI;
                double h2Prime = atan2(p_im[col + 2], a2Prime) + M_PI;

                double deltahPrime = h2Prime - h1Prime;
                if (abs(h1Prime - h2Prime) > M_PI)
                {
                  if (h1Prime + h2Prime < 2 * M_PI)
                    deltahPrime += 2 * M_PI;
                  else
                    deltahPrime -= 2 * M_PI;
                }
                if (C1Prime == 0 || C2Prime == 0)
                  deltahPrime = 0;

                double deltaHPrime = 2 * sqrt(C1Prime * C2Prime) * sin(deltahPrime / 2);

                double HDashPrime = 0;
                if (C1Prime == 0 || C2Prime == 0)
                  HDashPrime = h1Prime + h2Prime;
                else if (abs(h1Prime - h2Prime) <= M_PI)
                  HDashPrime = (h1Prime + h2Prime) / 2;
                else if (h1Prime + h2Prime < 2 * M_PI)
                  HDashPrime = (h1Prime + h2Prime + 2 * M_PI) / 2;
                else
                  HDashPrime = (h1Prime + h2Prime - 2 * M_PI) / 2;

                double T = 1 - 0.17 * cos(HDashPrime - (deg2Rad(30))) + 0.24 * cos(2 * HDashPrime) + 0.32 * cos(3 * HDashPrime + (deg2Rad(6))) - 0.2 * cos(4 * HDashPrime - (deg2Rad(63)));

                double SL = 1 + ((0.015 * pow(LDash - 50, 2)) / sqrt(20 + pow(LDash - 50, 2)));
                double SC = 1 + 0.045 * CDashPrime;
                double SH = 1 + 0.015 * CDashPrime * T;

                double RT = -2 * sqrt(pow(CDashPrime, 7) / (pow(CDashPrime, 7) + pow(25, 7))) * sin((deg2Rad(60)) * exp(-pow((HDashPrime - deg2Rad(275)) / (deg2Rad(25)), 2)));

                double kL = 1;
                double kC = 1;
                double kH = 1;

                variant += sqrt(pow(deltaLPrime / (kL * SL), 2) + pow(deltaCPrime / (kC * SC), 2) + pow(deltaHPrime / (kH * SH), 2) + RT * (deltaCPrime / (kC * SC)) * (deltaHPrime / (kH * SH)));*/
            }
        }

        variant += REPEAT_ADDITION * repeats[i];

        if (variant < best_variant)
        {
            best_variant = variant;
            best_fit = i;
        }
    }

    return best_fit;
}

// Callback for trackbars
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

void populateRepeats(vector< vector<int> > gridIndex, int y, int x, vector<int> *repeats)
{
    for (int i = 0; i < (int) (*repeats).size(); ++i)
        (*repeats)[i] = 0;

    int startX = wrap(x - REPEAT_RANGE, 0, gridIndex[0].size());
    int endX = wrap(x + REPEAT_RANGE, 0, gridIndex[0].size());
    int startY = wrap(y - REPEAT_RANGE, 0, gridIndex.size());
    //int endY = wrap(y + REPEAT_RANGE, 0, gridIndex.size());

    for (int yPos = startY; yPos < wrap(y - 1, 0, gridIndex.size()); ++yPos)
        for (int xPos = startX; xPos < endX; ++xPos)
            (*repeats)[gridIndex[yPos][xPos]]++;

    for (int xPos = startX; xPos < wrap(x - 1, 0, gridIndex[0].size()); ++xPos)
        (*repeats)[gridIndex[y][xPos]]++;
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
        cout << "-f: Switches colour difference algorithm from CIEDE2000 to CIE76 (less accurate, much faster)" << endl;

        return -1;
    }
    if (argc > 4) //Reads flags
    {
      String flags = argv[4];
      fast_mode = flags == "-f";
    }

    path img_out_path(argv[3]);
    //img_out_path = canonical(img_out_path);

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
////    PREPROCESS IMAGES
////////////////////////////////////////////////////////////////////////////////
    double t = getTickCount();

    //Read in images and preprocess them
    vector<Mat> images(fn.size());
    vector<Mat> imagesMax(fn.size());
    int max_cell_size = CELL_SIZE * (MAX_ZOOM / 100.0);
    for (size_t i = 0; i < fn.size(); ++i)
    {
        //Reads image, resizes to min zoom and max zoom (min used for compare)
        images[i] = imread(fn[i], IMREAD_COLOR);
        resizeImageExclusive(images[i], imagesMax[i], max_cell_size, max_cell_size);
        resizeImageExclusive(imagesMax[i], images[i], CELL_SIZE, CELL_SIZE);

        //Crops images, TEMPORARY
        imagesMax[i] = imagesMax[i](Range(0, max_cell_size), Range(0, max_cell_size));
        images[i] = images[i](Range(0, CELL_SIZE), Range(0, CELL_SIZE));

        //Converts to Lab colour space
        cvtColor(images[i], images[i], COLOR_BGR2Lab);
    }
    t = (getTickCount() - t) / getTickFrequency();
    cout << "Time passed in seconds for read: " << t << endl;
////////////////////////////////////////////////////////////////////////////////
////    CREATE MOSAIC PARTS
////////////////////////////////////////////////////////////////////////////////
    //Used to determine width of window for progress bar
    struct winsize w;
    ioctl(STDOUT_FILENO, TIOCGWINSZ, &w);

    t = getTickCount();
    //Creates 2D grid of image cells and splits main image between them
    //Finds best match for each cell and creates a 2D vector of the best matches
    vector< vector<int> > gridIndex(no_of_cell_y, vector<int>(no_of_cell_x));
    vector< vector<Mat> > result(no_of_cell_y, vector<Mat>(no_of_cell_x));

    vector<int> repeats(images.size());
    for (int y = 0; y < no_of_cell_y; ++y)
    {
        Mat cell;
        for (int x = 0; x < no_of_cell_x; ++x)
        {
            //Creates cell at x,y from main image
            cell = mainIm(Range(y * CELL_SIZE, (y+1) * CELL_SIZE),
                Range(x * CELL_SIZE, (x+1) * CELL_SIZE));

            //Calculates number of repeats around x,y for each image
            populateRepeats(gridIndex, y, x, &repeats);

            //Find best image for cell at x,y
            int temp = findBestImage(cell, images, repeats);
            gridIndex[y][x] = temp;
            result[y][x] = imagesMax[temp];
            
            progressBar(x + (no_of_cell_x - 1) * y, no_of_cell_x * no_of_cell_y - 1, w.ws_col);
        }
    }
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
    return 0;
}
