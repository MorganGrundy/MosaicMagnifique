#ifndef PHOTOMOSAICGENERATOR_H
#define PHOTOMOSAICGENERATOR_H

#include <QProgressDialog>
#include <opencv2/core/mat.hpp>
#include <optional>

#include "cellshape.h"
#include "utilityfuncs.h"

class PhotomosaicGenerator : private QProgressDialog
{
    Q_OBJECT
public:
    enum class Mode {RGB_EUCLIDEAN, CIE76, CIEDE2000};

    PhotomosaicGenerator(QWidget *t_parent = nullptr);
    ~PhotomosaicGenerator();

    void setMainImage(const cv::Mat &t_img);
    void setLibrary(const std::vector<cv::Mat> &t_lib);
    void setDetail(const int t_detail = 100);
    void setMode(const Mode t_mode = Mode::RGB_EUCLIDEAN);

    void setCellShape(const CellShape &t_cellShape);
    void setSizeSteps(const size_t t_steps);
    void setRepeat(int t_repeatRange = 0, int t_repeatAddition = 0);

    cv::Mat generate();
#ifdef CUDA
    cv::Mat cudaGenerate();
#endif

private:
    cv::Mat m_img;
    std::vector<cv::Mat> m_lib;

    double m_detail;
    Mode m_mode;

    CellShape m_cellShape;
    size_t sizeSteps;

    int m_repeatRange, m_repeatAddition;

    std::pair<cv::Mat, std::vector<cv::Mat>> resizeAndCvtColor();

    std::optional<size_t> findCellBestFit(const CellShape &t_cellShape, const int x, const int y,
                                          const bool t_pad,
                                          const cv::Mat &t_image,
                                          const std::vector<cv::Mat> &t_lib,
                                          const std::vector<std::vector<size_t>> &t_grid) const;

    int findBestFitEuclidean(const cv::Mat &cell, const cv::Mat &mask,
                             const std::vector<cv::Mat> &library,
                             const std::map<size_t, int> &repeats,
                             const int yStart, const int yEnd,
                             const int xStart, const int xEnd) const;
    int findBestFitCIEDE2000(const cv::Mat &cell, const cv::Mat &mask,
                             const std::vector<cv::Mat> &library,
                             const std::map<size_t, int> &repeats,
                             const int yStart, const int yEnd,
                             const int xStart, const int xEnd) const;

    std::map<size_t, int> calculateRepeats(const std::vector<std::vector<size_t>> &grid,
                                           const int x, const int y) const;

    double degToRad(const double deg) const;
    cv::Mat combineResults(const cv::Point gridSize,
                           const std::vector<std::vector<size_t>> &result);
};

#endif // PHOTOMOSAICGENERATOR_H
