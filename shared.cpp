#ifndef SHARED_CPP_
#define SHARED_CPP_

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
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

//Displays progress bar
void progressBar(int cur, int max, int width)
{
    int bar_width = width - 8; //-8 for additional characters
    int progress = (cur * bar_width) / max;
    cout << (cur * 100) / max << "% [";
    for (int i = 0; i < bar_width; i++)
        if (i < progress)
            cout << "=";
        else
            cout << " ";
    cout << "]" << " \r";
    cout.flush();
}

// If val exceeds a bound returns bound else returns val
int wrap(int val, int lower_bound, int upper_bound)
{
    if (val > upper_bound)
        return upper_bound;
    else if (val < lower_bound)
        return lower_bound;
    else
        return val;
}

//Converts the given value in degrees into radians
double deg2Rad(double deg)
{
  return (deg * M_PI / 180.0);
}

// Resizes input image (img) such that
// (height = targetHeight && width <= targetWidth) || (height <= targetHeight && width = targetWidth)
// and puts the resized image in result
void resizeImageInclusive(Mat& img, Mat& result, int targetHeight, int targetWidth)
{
    //Calculates resize factor
    double resizeFactor = ((double) targetHeight / img.rows);
    if (targetWidth < resizeFactor * img.cols)
        resizeFactor = ((double) targetWidth / img.cols);

    //Resizes image
    if (resizeFactor < 1)
        resize(img, result, Size(resizeFactor * img.cols, resizeFactor * img.rows), 0, 0, INTER_AREA);
    else if (resizeFactor > 1)
        resize(img, result, Size(resizeFactor * img.cols, resizeFactor * img.rows), 0, 0, INTER_CUBIC);
    else
        result = img;
}

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

//Given filepath creates list of filepaths to accepted images in given filepath
bool read_image_names(path img_in_path, vector<String> *fn)
{
  try
  {
    //If in path is a directory
    if (exists(img_in_path) && is_directory(img_in_path))
    {
      typedef vector<path> PathVec; //Vector of path
      PathVec paths;

      //For all files and directories in img_in_path
      recursive_directory_iterator end_itr;
      for (recursive_directory_iterator it(img_in_path); it != end_itr; ++it)
      {
        if (is_regular_file(*it))
        {
          //If file extension is accepted image format
          String fileExtension = extension(*it);
          if (find(IMG_FORMATS.begin(), IMG_FORMATS.end(), fileExtension) != IMG_FORMATS.end())
          {
            //Adds file to fn
            (*fn).push_back(it->path().string());
          }
        }
      }
    }
    else
    {
      cout << img_in_path << " is not a directory" << endl;
      return false;
    }
  }
  catch (const filesystem_error& ex)
  {
    cout << ex.what() << endl;
    return false;
  }

  return true;
}

#endif //SHARED_CPP_
