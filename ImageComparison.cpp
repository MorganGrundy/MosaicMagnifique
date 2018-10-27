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

// Returns the index in images of the image with the least variance from main_img, using CIE76 colour difference
int findBestImageCIE76(Mat& main_img, vector<Mat> images, vector<int> repeats)
{
    int main_rows = main_img.rows;
    int main_cols = main_img.cols * main_img.channels();

    //Index of current image with lowest variant
    int best_fit = -1;
    //Initial best_variant set higher than max variant
    long double best_variant = LDBL_MAX;
    for (int i = 0; i < (int) images.size(); ++i)
    {
        uchar* p_main;
        uchar* p_im;
        long double variant = REPEAT_ADDITION * repeats[i];
        //Calculates sum of difference between corresponding pixel values
        for (int row = 0; row < main_rows && variant < best_variant; ++row)
        {
            p_main = main_img.ptr<uchar>(row);
            p_im = images[i].ptr<uchar>(row);
            for (int col = 0; col < main_cols && variant < best_variant; col += main_img.channels())
                variant += sqrt(pow(p_main[col] - p_im[col], 2) +
                                pow(p_main[col + 1] - p_im[col + 1], 2) +
                                pow(p_main[col + 2] - p_im[col + 2], 2));
        }
        if (variant < best_variant)
        {
            best_variant = variant;
            best_fit = i;
        }
    }
    return best_fit;
}

//Returns 2D vector of Mat that make up the best fitting images of main_img cells using CIE76 colour difference
vector< vector<Mat> > findBestImagesCIE76(Mat& main_img, vector<Mat>& images, vector<Mat>& imagesMax, int no_of_cell_x, int no_of_cell_y, int window_width)
{
  vector< vector<int> > gridIndex(no_of_cell_y, vector<int>(no_of_cell_x));
  vector< vector<Mat> > result(no_of_cell_y, vector<Mat>(no_of_cell_x));
  vector<int> repeats(images.size());
  Mat cell;
  for (int y = 0; y < no_of_cell_y; ++y)
  {
      for (int x = 0; x < no_of_cell_x; ++x)
      {
          //Creates cell at x,y from main image
          cell = main_img(Range(y * CELL_SIZE, (y+1) * CELL_SIZE),
              Range(x * CELL_SIZE, (x+1) * CELL_SIZE));

          //Calculates number of repeats around x,y for each image
          populateRepeats(gridIndex, y, x, &repeats);

          //Find best image for cell at x,y
          int temp = findBestImageCIE76(cell, images, repeats);
          gridIndex[y][x] = temp;
          result[y][x] = imagesMax[temp];

          progressBar(x + (no_of_cell_x - 1) * y, no_of_cell_x * no_of_cell_y - 1, window_width);
      }
  }
  progressBarClean(window_width);

  return result;
}

// Returns the index in images of the image with the least variance from main_img, using CIE2000 colour difference
int findBestImageCIE2000(Mat& main_img, vector<Mat> images, vector<int> repeats, vector<vector<vector<double> > >& C2Star)
{
  int main_rows = main_img.rows;
  int main_cols = main_img.cols * main_img.channels();

  uchar* p_main;
  vector< vector<double> > C1Star(main_rows, vector<double>(main_cols));
  for (int row = 0; row < main_rows; ++row)
  {
    p_main = main_img.ptr<uchar>(row);
    for (int col = 0; col < main_cols; col += main_img.channels())
      C1Star[row][col] = sqrt(pow(p_main[col + 1], 2) + pow(p_main[col + 2], 2));
  }

  //Index of current image with lowest variant
  int best_fit = -1;
  //Initial best_variant set higher than max variant
  long double best_variant = LDBL_MAX;
  for (int i = 0; i < (int) images.size(); ++i)
  {
    uchar* p_im;
    long double variant = REPEAT_ADDITION * repeats[i];
    //Calculates sum of difference between corresponding pixel values
    for (int row = 0; row < main_rows && variant < best_variant; ++row)
    {
      p_main = main_img.ptr<uchar>(row);
      p_im = images[i].ptr<uchar>(row);
      for (int col = 0; col < main_cols && variant < best_variant; col += main_img.channels())
      {
        //double C2Star = sqrt(pow(p_im[col + 1], 2) + pow(p_im[col + 2], 2));
        double LDash = (p_main[col] + p_im[col]) / 2;
        double CDash = pow((C1Star[row][col] + C2Star[i][row][col]) / 2, 7);

        double a1Prime = p_main[col + 1] + (p_main[col + 1] / 2) * (1 - sqrt(CDash / (CDash + pow(25, 7))));
        double a2Prime = p_im[col + 1] + (p_im[col + 1] / 2) * (1 - sqrt(CDash / (CDash + pow(25, 7))));

        double C1Prime = sqrt(pow(a1Prime, 2) + pow(p_main[col + 2], 2));
        double C2Prime = sqrt(pow(a2Prime, 2) + pow(p_im[col + 2], 2));

        double CDashPrime = (C1Prime + C2Prime) / 2;

        double h1Prime = atan2(p_main[col + 2], a1Prime) + M_PI;
        double h2Prime = atan2(p_im[col + 2], a2Prime) + M_PI;

        double deltahPrime;
        if (C1Prime == 0 || C2Prime == 0)
          deltahPrime = 0;
        else
        {
          deltahPrime = (h2Prime - h1Prime) / 2;
          if (abs(h1Prime - h2Prime) > M_PI)
          {
            if (h1Prime + h2Prime < 2 * M_PI)
              deltahPrime += M_PI;
            else
              deltahPrime -= M_PI;
          }
        }

        double HDashPrime = 0;
        if (C1Prime == 0 || C2Prime == 0)
          HDashPrime = h1Prime + h2Prime;
        else if (abs(h1Prime - h2Prime) <= M_PI)
          HDashPrime = (h1Prime + h2Prime) / 2;
        else if (h1Prime + h2Prime < 2 * M_PI)
          HDashPrime = (h1Prime + h2Prime + 2 * M_PI) / 2;
        else
          HDashPrime = (h1Prime + h2Prime - 2 * M_PI) / 2;

        double T = 1 - 0.17 * cos(HDashPrime - (DEG2RAD(30))) + 0.24 * cos(2 * HDashPrime) + 0.32 * cos(3 * HDashPrime + (DEG2RAD(6))) - 0.2 * cos(4 * HDashPrime - (DEG2RAD(63)));

        double SL = 1 + ((0.015 * pow(LDash - 50, 2)) / sqrt(20 + pow(LDash - 50, 2)));
        double SC = 1 + 0.045 * CDashPrime;

        double deltaCPrime = (C2Prime - C1Prime) / SC;

        double deltaHPrime = 2 * sqrt(C1Prime * C2Prime) * sin(deltahPrime);
        deltaHPrime /= 1 + 0.015 * CDashPrime * T;

        double RT = -2 * sqrt(pow(CDashPrime, 7) / (pow(CDashPrime, 7) + pow(25, 7))) * sin((DEG2RAD(60)) * exp(-pow((HDashPrime - DEG2RAD(275)) / (DEG2RAD(25)), 2)));

        variant += sqrt(pow((p_im[col] - p_main[col]) / SL, 2) + pow(deltaCPrime, 2) + pow(deltaHPrime, 2) + RT * deltaCPrime * deltaHPrime);
      }
    }
    if (variant < best_variant)
    {
      best_variant = variant;
      best_fit = i;
    }
  }
  return best_fit;
}

//Returns 2D vector of Mat that make up the best fitting images of main_img cells using CIE2000 colour difference
vector< vector<Mat> > findBestImagesCIE2000(Mat& main_img, vector<Mat>& images, vector<Mat>& imagesMax, int no_of_cell_x, int no_of_cell_y, int window_width)
{
  int main_rows = main_img.rows;
  int main_cols = main_img.cols * main_img.channels();

  vector<vector<vector<double> > > C2Star(images.size(), vector< vector<double> >(main_rows, vector<double>(main_cols)));
  for (int i = 0; i < (int) images.size(); ++i)
  {
    uchar* p_im;
    for (int row = 0; row < main_rows; ++row)
    {
      p_im = images[i].ptr<uchar>(row);
      for (int col = 0; col < main_cols; col += main_img.channels())
      {
        C2Star[i][row][col] = sqrt(pow(p_im[col + 1], 2) + pow(p_im[col + 2], 2));
      }
    }
  }

  vector< vector<int> > gridIndex(no_of_cell_y, vector<int>(no_of_cell_x));
  vector< vector<Mat> > result(no_of_cell_y, vector<Mat>(no_of_cell_x));
  vector<int> repeats(images.size());
  Mat cell;
  for (int y = 0; y < no_of_cell_y; ++y)
  {
      for (int x = 0; x < no_of_cell_x; ++x)
      {
          //Creates cell at x,y from main image
          cell = main_img(Range(y * CELL_SIZE, (y+1) * CELL_SIZE),
              Range(x * CELL_SIZE, (x+1) * CELL_SIZE));

          //Calculates number of repeats around x,y for each image
          populateRepeats(gridIndex, y, x, &repeats);

          //Find best image for cell at x,y
          int temp = findBestImageCIE2000(cell, images, repeats, C2Star);
          gridIndex[y][x] = temp;
          result[y][x] = imagesMax[temp];

          progressBar(x + (no_of_cell_x - 1) * y, no_of_cell_x * no_of_cell_y - 1, window_width);
      }
  }
  progressBarClean(window_width);

  return result;
}
