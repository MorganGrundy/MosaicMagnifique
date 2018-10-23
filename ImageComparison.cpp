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

// Returns the index in images of the image with the least variance from main_img, using CIE76 colour difference
int findBestImageCIE76(Mat& main_img, vector<Mat> images, vector<int> repeats)
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
                variant += sqrt(pow(p_main[col] - p_im[col], 2) +
                                pow(p_main[col + 1] - p_im[col + 1], 2) +
                                pow(p_main[col + 2] - p_im[col + 2], 2));
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

// Returns the index in images of the image with the least variance from main_img, using CIE2000 colour difference
int findBestImageCIE2000(Mat& main_img, vector<Mat> images, vector<int> repeats)
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

                variant += sqrt(pow(deltaLPrime / (kL * SL), 2) + pow(deltaCPrime / (kC * SC), 2) + pow(deltaHPrime / (kH * SH), 2) + RT * (deltaCPrime / (kC * SC)) * (deltaHPrime / (kH * SH)));
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
