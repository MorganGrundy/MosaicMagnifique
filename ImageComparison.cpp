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
#include "cells.hpp"

using namespace cv;
using namespace std;

//Worst case complexity: O((REPEAT_RANGE^2) / 2)
void populateRepeats(vector< vector<int> > gridIndex, int y, int x, vector<int> *repeats)
{
    for (int i = 0; i < (int) (*repeats).size(); ++i) //O(Ni)
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

// Returns the index in images of the image with the least variance from main_img in the given area, using CIE76 colour difference
//Worst case complexity: O(Ni * CELL_SIZE^2), where Ni = number of images
int findBestImageCIE76(Mat& main_img, vector<Mat> images, vector<int> repeats, int yStart, int yEnd, int xStart, int xEnd)
{
    //Index of current image with lowest variant
    int best_fit = -1;
    //Initial best_variant set higher than max variant
    long double best_variant = LDBL_MAX;
    for (int i = 0; i < (int) images.size(); ++i)
    {
        uchar* p_main;
        uchar* p_im;
        uchar* p_mask;
        long double variant = REPEAT_ADDITION * repeats[i];
        //Calculates sum of difference between corresponding pixel values
        for (int row = yStart; row < yEnd && variant < best_variant; ++row)
        {
            p_main = main_img.ptr<uchar>(row);
            p_im = images[i].ptr<uchar>(row);
            p_mask = cellMaskCmp.ptr<uchar>(row);
            for (int col = xStart; col < xEnd * main_img.channels() && variant < best_variant; col += main_img.channels())
                //Only compare if default cell shape or in active mask area
                if (p_mask[col / main_img.channels()] == 255)
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
//Worst case complexity: O(y * x ((REPEAT_RANGE^2) / 2) * Ni * CELL_SIZE^2), where Ni = number of images, x = main image rows / CELL_SIZE, y = main image cols / CELL_SIZE
vector< vector<Mat> > findBestImagesCIE76(Mat& main_img, vector<Mat>& images, vector<Mat>& imagesMax, int no_of_cell_x, int no_of_cell_y, int window_width)
{
  vector< vector<int> > gridIndex(no_of_cell_y + padCols * 2, vector<int>(no_of_cell_x + padRows * 2));

  vector< vector<Mat> > result(no_of_cell_y + padCols * 2, vector<Mat>(no_of_cell_x + padRows * 2));

  vector<int> repeats(images.size());

  Mat cell(cellMaskCmp.rows, cellMaskCmp.cols, main_img.type(), cvScalar(0));

  //For all cells
  for (int y = -padCols; y < no_of_cell_y + padCols; ++y)
  {
      for (int x = -padRows; x < no_of_cell_x + padRows; ++x)
      {
          //Cell y start position
          int yUnboundedStart = y * cellOffsetCmpY[0] + ((abs(x % 2) == 1) ? cellOffsetCmpX[0] : 0);
          //Cell bounded y start position (in image area)
          int yStart = wrap(yUnboundedStart, 0, main_img.rows - 1);

          //Cell y end position
          int yUnboundedEnd = y * cellOffsetCmpY[0] + cellMaskCmp.rows + ((abs(x % 2) == 1) ? cellOffsetCmpX[0] : 0);
          //Cell bounded y end position (in image area)
          int yEnd = wrap(yUnboundedEnd, 0, main_img.rows - 1);

          //Cell x start position
          int xUnboundedStart = x * cellOffsetCmpX[1] + ((abs(y % 2) == 1) ? cellOffsetCmpY[1] : 0);
          //Cell bounded x start position (in image area)
          int xStart = wrap(xUnboundedStart, 0, main_img.cols - 1);

          //Cell x end position
          int xUnboundedEnd = x * cellOffsetCmpX[1] + cellMaskCmp.cols + ((abs(y % 2) == 1) ? cellOffsetCmpY[1] : 0);
          //Cell bounded x end position (in image area)
          int xEnd = wrap(xUnboundedEnd, 0, main_img.cols - 1);

          //Cell completely out of bounds, just skip
          if (yStart == yEnd || xStart == xEnd)
          {
              result[y + padCols][x + padRows] = imagesMax[0];
              continue;
          }

          //Creates cell at x,y from main image
          Mat imageBounded = main_img(Range(yStart, yEnd), Range(xStart, xEnd));
          imageBounded.copyTo(cell(Range(yStart - yUnboundedStart, yEnd - yUnboundedStart), Range(xStart - xUnboundedStart, xEnd - xUnboundedStart)));

          //Calculates number of repeats around x,y for each image
          populateRepeats(gridIndex, y + padCols, x + padRows, &repeats);

          //Find best image for cell at x,y
          int temp = findBestImageCIE76(cell, images, repeats, yStart - yUnboundedStart, yEnd - yUnboundedStart, xStart - xUnboundedStart, xEnd - xUnboundedStart);
          gridIndex[y + padCols][x + padRows] = temp;
          result[y + padCols][x + padRows] = imagesMax[temp];

          progressBar(x + padRows * 2 + (no_of_cell_x - 1) * (y + padCols * 2), (no_of_cell_x + padRows * 2) * (no_of_cell_y + padCols * 2) - 1, window_width);
      }
  }
  progressBarClean(window_width);

  return result;
}

// Returns the index in images of the image with the least variance from main_img in the given area, using CIE2000 colour difference
//Worst case complexity: O(Ni * CELL_SIZE^2), where Ni = number of images
int findBestImageCIE2000(Mat& main_img, vector<Mat> images, vector<int> repeats, int yStart, int yEnd, int xStart, int xEnd)
{
    //Index of current image with lowest variant
    int best_fit = -1;
    //Initial best_variant set higher than max variant
    long double best_variant = LDBL_MAX;
    for (int i = 0; i < (int) images.size(); ++i)
    {
        uchar* p_main;
        uchar* p_im;
        uchar* p_mask;
        long double variant = REPEAT_ADDITION * repeats[i];
        //Calculates sum of difference between corresponding pixel values
        for (int row = yStart; row < yEnd && variant < best_variant; ++row)
        {
            p_main = main_img.ptr<uchar>(row);
            p_im = images[i].ptr<uchar>(row);
            p_mask = cellMaskCmp.ptr<uchar>(row);
            for (int col = xStart; col < xEnd * main_img.channels() && variant < best_variant; col += main_img.channels())
                //Only compare if default cell shape or in active mask area
                if (p_mask[col / main_img.channels()] == 255)
                {
                    const double k_L = 1.0, k_C = 1.0, k_H = 1.0;
                    const double deg360InRad = DEG2RAD(360.0);
                    const double deg180InRad = DEG2RAD(180.0);
                    const double pow25To7 = 6103515625.0; //pow(25, 7)

                    double C1 = sqrt((p_main[col + 1] * p_main[col + 1]) + (p_main[col + 2] * p_main[col + 2]));
                    double C2 = sqrt((p_im[col + 1] * p_im[col + 1]) + (p_im[col + 2] * p_im[col + 2]));
                    double barC = (C1 + C2) / 2.0;

                    double G = 0.5 * (1 - sqrt(pow(barC, 7) / (pow(barC, 7) + pow25To7)));

                    double a1Prime = (1.0 + G) * p_main[col + 1];
                    double a2Prime = (1.0 + G) * p_im[col + 1];

                    double CPrime1 = sqrt((a1Prime * a1Prime) + (p_main[col + 2] * p_main[col + 2]));
                    double CPrime2 = sqrt((a2Prime * a2Prime) + (p_im[col + 2] * p_im[col + 2]));

                    double hPrime1;
                    if (p_main[col + 2] == 0 && a1Prime == 0)
                        hPrime1 = 0.0;
                    else {
                        hPrime1 = atan2(p_main[col + 2], a1Prime);
                        //This must be converted to a hue angle in degrees between 0 and 360 by addition of 2 pi to negative hue angles.
                        if (hPrime1 < 0)
                            hPrime1 += deg360InRad;
                    }
                    double hPrime2;
                    if (p_im[col + 2] == 0 && a2Prime == 0)
                        hPrime2 = 0.0;
                    else {
                        hPrime2 = atan2(p_im[col + 2], a2Prime);
                        //This must be converted to a hue angle in degrees between 0 and 360 by addition of 2pi to negative hue angles.
                        if (hPrime2 < 0)
                            hPrime2 += deg360InRad;
                    }

                    double deltaLPrime = p_im[col] - p_main[col];
                    double deltaCPrime = CPrime2 - CPrime1;

                    double deltahPrime;
                    double CPrimeProduct = CPrime1 * CPrime2;
                    if (CPrimeProduct == 0)
                        deltahPrime = 0;
                    else {
                        //Avoid the fabs() call
                        deltahPrime = hPrime2 - hPrime1;
                        if (deltahPrime < -deg180InRad)
                            deltahPrime += deg360InRad;
                        else if (deltahPrime > deg180InRad)
                            deltahPrime -= deg360InRad;
                    }

                    double deltaHPrime = 2.0 * sqrt(CPrimeProduct) * sin(deltahPrime / 2.0);

                    double barLPrime = (p_main[col] + p_im[col]) / 2.0;
                    double barCPrime = (CPrime1 + CPrime2) / 2.0;

                    double barhPrime, hPrimeSum = hPrime1 + hPrime2;
                    if (CPrime1 * CPrime2 == 0) {
                        barhPrime = hPrimeSum;
                    } else {
                        if (fabs(hPrime1 - hPrime2) <= deg180InRad)
                            barhPrime = hPrimeSum / 2.0;
                        else {
                            if (hPrimeSum < deg360InRad)
                                barhPrime = (hPrimeSum + deg360InRad) / 2.0;
                            else
                                barhPrime = (hPrimeSum - deg360InRad) / 2.0;
                        }
                    }

                    double T = 1.0 - (0.17 * cos(barhPrime - DEG2RAD(30.0))) + (0.24 * cos(2.0 * barhPrime)) + (0.32 * cos((3.0 * barhPrime) + DEG2RAD(6.0))) - (0.20 * cos((4.0 * barhPrime) - DEG2RAD(63.0)));

                    double deltaTheta = DEG2RAD(30.0) * exp(-pow((barhPrime - deg2Rad(275.0)) / deg2Rad(25.0), 2.0));

                    double R_C = 2.0 * sqrt(pow(barCPrime, 7.0) / (pow(barCPrime, 7.0) + pow25To7));

                    double S_L = 1 + ((0.015 * pow(barLPrime - 50.0, 2.0)) / sqrt(20 + pow(barLPrime - 50.0, 2.0)));
                    double S_C = 1 + (0.045 * barCPrime);
                    double S_H = 1 + (0.015 * barCPrime * T);

                    double R_T = (-sin(2.0 * deltaTheta)) * R_C;


                    variant += sqrt(pow(deltaLPrime / (k_L * S_L), 2.0) + pow(deltaCPrime / (k_C * S_C), 2.0) + pow(deltaHPrime / (k_H * S_H), 2.0) + (R_T * (deltaCPrime / (k_C * S_C)) * (deltaHPrime / (k_H * S_H))));
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
//Worst case complexity: O(y * x ((REPEAT_RANGE^2) / 2) * Ni * CELL_SIZE^2), where Ni = number of images, x = main image rows / CELL_SIZE, y = main image cols / CELL_SIZE
vector< vector<Mat> > findBestImagesCIE2000(Mat& main_img, vector<Mat>& images, vector<Mat>& imagesMax, int no_of_cell_x, int no_of_cell_y, int window_width)
{
  vector< vector<int> > gridIndex(no_of_cell_y + padCols * 2, vector<int>(no_of_cell_x + padRows * 2));

  vector< vector<Mat> > result(no_of_cell_y + padCols * 2, vector<Mat>(no_of_cell_x + padRows * 2));

  vector<int> repeats(images.size());

  Mat cell(cellMaskCmp.rows, cellMaskCmp.cols, main_img.type(), cvScalar(0));

  //For all cells
  for (int y = -padCols; y < no_of_cell_y + padCols; ++y)
  {
      for (int x = -padRows; x < no_of_cell_x + padRows; ++x)
      {
          //Cell y start position
          int yUnboundedStart = y * cellOffsetCmpY[0] + ((abs(x % 2) == 1) ? cellOffsetCmpX[0] : 0);
          //Cell bounded y start position (in image area)
          int yStart = wrap(yUnboundedStart, 0, main_img.rows - 1);

          //Cell y end position
          int yUnboundedEnd = y * cellOffsetCmpY[0] + cellMaskCmp.rows + ((abs(x % 2) == 1) ? cellOffsetCmpX[0] : 0);
          //Cell bounded y end position (in image area)
          int yEnd = wrap(yUnboundedEnd, 0, main_img.rows - 1);

          //Cell x start position
          int xUnboundedStart = x * cellOffsetCmpX[1] + ((abs(y % 2) == 1) ? cellOffsetCmpY[1] : 0);
          //Cell bounded x start position (in image area)
          int xStart = wrap(xUnboundedStart, 0, main_img.cols - 1);

          //Cell x end position
          int xUnboundedEnd = x * cellOffsetCmpX[1] + cellMaskCmp.cols + ((abs(y % 2) == 1) ? cellOffsetCmpY[1] : 0);
          //Cell bounded x end position (in image area)
          int xEnd = wrap(xUnboundedEnd, 0, main_img.cols - 1);

          //Cell completely out of bounds, just skip
          if (yStart == yEnd || xStart == xEnd)
          {
              result[y + padCols][x + padRows] = imagesMax[0];
              continue;
          }

          //Creates cell at x,y from main image
          Mat imageBounded = main_img(Range(yStart, yEnd), Range(xStart, xEnd));
          imageBounded.copyTo(cell(Range(yStart - yUnboundedStart, yEnd - yUnboundedStart), Range(xStart - xUnboundedStart, xEnd - xUnboundedStart)));

          //Calculates number of repeats around x,y for each image
          populateRepeats(gridIndex, y + padCols, x + padRows, &repeats);

          //Find best image for cell at x,y
          int temp = findBestImageCIE2000(cell, images, repeats, yStart - yUnboundedStart, yEnd - yUnboundedStart, xStart - xUnboundedStart, xEnd - xUnboundedStart);
          gridIndex[y + padCols][x + padRows] = temp;
          result[y + padCols][x + padRows] = imagesMax[temp];

          progressBar(x + padRows * 2 + (no_of_cell_x - 1) * (y + padCols * 2), (no_of_cell_x + padRows * 2) * (no_of_cell_y + padCols * 2) - 1, window_width);
      }
  }
  progressBarClean(window_width);

  return result;
}
