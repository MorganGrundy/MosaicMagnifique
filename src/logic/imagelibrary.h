#pragma once

#include <opencv2/core.hpp>
#include <QString>

#include "ImageUtility.h"

class ImageLibrary
{
public:
    static const quint32 MIL_VERSION = 4;
    static const quint32 MIL_MAGIC = 0xADBE2480;

    ImageLibrary(const size_t t_imageSize);

    //Return if ImageLibrary is equal to other
    bool operator==(const ImageLibrary &t_other) const;

    //Set image size
    void setImageSize(const size_t t_size);
    //Returns image size
    size_t getImageSize() const;

    //Add image to library with given name
    void addImage(const cv::Mat &t_im, const QString &t_name = QString());

    //Returns const reference to library image names
    const std::vector<QString> &getNames() const;
    //Returns const reference to library images
    const std::vector<cv::Mat> &getImages() const;

    //Removes the image at given index
    void removeAtIndex(const size_t t_index);
    //Clear image library
    void clear();

    //Saves the image library to the given file
    void saveToFile(const QString t_filename) const;

    //Loads image library from given file
    void loadFromFile(const QString t_filename);

private:
    //Size of images in library
    size_t m_imageSize;

    std::vector<QString> m_names; //Image names
    std::vector<cv::Mat> m_originalImages; //Original images
    std::vector<cv::Mat> m_resizedImages; //Library images resized

};