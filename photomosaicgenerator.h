#ifndef PHOTOMOSAICGENERATOR_H
#define PHOTOMOSAICGENERATOR_H

#include <QProgressDialog>
#include <opencv2/core/mat.hpp>

class PhotomosaicGenerator : public QProgressDialog
{
    Q_OBJECT
public:
    enum class Mode {RGB_EUCLIDEAN, CIE76, CIEDE2000};

    PhotomosaicGenerator(QWidget *t_parent = nullptr);
    ~PhotomosaicGenerator();

    cv::Mat generate(cv::Mat &mainImage, const std::vector<cv::Mat> &library);

    int findBestFitEuclidean(const cv::Mat &cell, const std::vector<cv::Mat> &library,
                             const std::map<size_t, int> &repeats) const;
    int findBestFitCIEDE2000(const cv::Mat &cell, const std::vector<cv::Mat> &library,
                             const std::map<size_t, int> &repeats)const ;

    std::map<size_t, int> calculateRepeats(const std::vector<std::vector<size_t>> &grid,
                                           const cv::Point &gridSize, const int x, const int y) const;

    double degToRad(const double deg) const;

    Mode m_mode;
    int m_repeatRange, m_repeatAddition;
};

#endif // PHOTOMOSAICGENERATOR_H
