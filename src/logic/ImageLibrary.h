#pragma once

#include <opencv2/core.hpp>
#include <QString>

#include "ImageUtility.h"

class ImageLibrary
{
public:
    //Current version number
    static const quint32 MIL_VERSION = 6;
    static const quint32 MIL_MAGIC = 0xADBE2480;

    //Version that random sorting was added
    static const quint32 MIL_VERSION_RANDOMSORT = 5;
    //Version that image encoding was added
    static const quint32 MIL_VERSION_ENCODED = 6;

    ImageLibrary(const size_t t_imageSize);

    //Return if ImageLibrary is equal to other
    bool operator==(const ImageLibrary &t_other) const;

    //Set image size
    void setImageSize(const size_t t_size);
    //Returns image size
    size_t getImageSize() const;

    //Add image to library with given name at random index
    //Returns the index
    size_t addImage(const cv::Mat &t_im, const QString &t_name = QString());

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

protected:
    //Internals of addImage, adds image at the given index in relevant containers
    //Allows this to be overriden for CUDA functionality while using the same addImage
    virtual void addImageInternal(const size_t index, const cv::Mat &t_im);

private:
    //Size of images in library
    size_t m_imageSize;

    std::vector<QString> m_names; //Image names
    std::vector<cv::Mat> m_originalImages; //Original images
    std::vector<cv::Mat> m_resizedImages; //Library images resized
};