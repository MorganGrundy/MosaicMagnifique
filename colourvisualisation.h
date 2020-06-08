#ifndef COLOURVISUALISATION_H
#define COLOURVISUALISATION_H

#include <QMainWindow>
#include <opencv2/core/mat.hpp>

namespace Ui {
class ColourVisualisation;
}

class ColourVisualisation : public QMainWindow
{
    Q_OBJECT

public:
    explicit ColourVisualisation(QWidget *parent = nullptr);
    explicit ColourVisualisation(QWidget *parent, const cv::Mat &t_image, cv::Mat *t_libImages,
                                 const size_t t_noLib);
    ~ColourVisualisation();

private:
    void createColourList();

    Ui::ColourVisualisation *ui;

    cv::Mat colours;

    cv::Mat mainHistogram, libraryHistogram;

    //Number of bins in histogram
    const int noOfBins = 30;
    const int histogramSize[3] = {noOfBins, noOfBins, noOfBins};

    //Histogram range
    const float RGBRanges[2] = {0, 256};
    const float *ranges[3] = {RGBRanges, RGBRanges, RGBRanges};

    const int channels[3] = {0, 1, 2};

    const int iconSize = 100;
};

#endif // COLOURVISUALISATION_H
