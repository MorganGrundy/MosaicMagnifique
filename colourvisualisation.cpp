#include "colourvisualisation.h"
#include "ui_colourvisualisation.h"

#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

ColourVisualisation::ColourVisualisation(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::ColourVisualisation)
{
    ui->setupUi(this);
}

ColourVisualisation::ColourVisualisation(QWidget *parent, const cv::Mat &t_image,
                                         cv::Mat *t_libImages, const size_t t_noLib) :
    QMainWindow(parent),
    ui(new Ui::ColourVisualisation)
{
    ui->setupUi(this);

    //Create main histogram
    cv::calcHist(&t_image, 1, channels, cv::Mat(), mainHistogram, 3, histogramSize, ranges,
                 true, false);

    //Create library histogram
    cv::calcHist(t_libImages, static_cast<int>(t_noLib), channels, cv::Mat(),
                 libraryHistogram, 3, histogramSize, ranges, true, false);


    createColourList();
}

ColourVisualisation::~ColourVisualisation()
{
    delete ui;
}

void ColourVisualisation::createColourList()
{
    //Stores needed colour bins and priorities
    std::vector<std::pair<std::tuple<int, int, int>, float>> colourPriority;

    //Iterates over all bins in histogram
    for (int b = 0; b < histogramSize[2]; ++b)
    {
        for (int g = 0; g < histogramSize[1]; ++g)
        {
            for (int r = 0; r < histogramSize[0]; ++r)
            {
                float mainBin = mainHistogram.at<float>(b, g, r);
                //Bin is not empty
                if (mainBin > 0)
                {
                    float libraryBin = libraryHistogram.at<float>(b, g, r);

                    //Calculate priority
                    float priority = 0;
                    //Library bin empty instead treat as 0.5
                    if (libraryBin == 0)
                        priority = mainBin * 2;
                    else
                        priority = mainBin / libraryBin;

                    colourPriority.push_back({{r, g, b}, priority});
                }
            }
        }
    }

    //Sort colour bins by descending priority
    std::sort(colourPriority.begin(), colourPriority.end(),
              [](const auto &a, const auto &b)
    {
        return a.second > b.second;
    });

    for (auto data: colourPriority)
    {
        //Create square image of bin colour (using bin median colour)
        QPixmap colour(iconSize, iconSize);
        colour.fill(QColor((std::get<0>(data.first) + 0.5) * ((RGBRanges[1] - 1) / noOfBins),
                           (std::get<1>(data.first) + 0.5) * ((RGBRanges[1] - 1) / noOfBins),
                           (std::get<2>(data.first) + 0.5) * ((RGBRanges[1] - 1) / noOfBins)));

        QListWidgetItem *listItem = new QListWidgetItem(QIcon(colour), QString());
        ui->listWidget->addItem(listItem);
    }
}
