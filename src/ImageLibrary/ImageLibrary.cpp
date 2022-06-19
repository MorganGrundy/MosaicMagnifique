#include "ImageLibrary.h"

#include <QDebug>
#include <QFile>
#include <stdexcept>
#include <random>

#include <opencv2/imgcodecs.hpp>

#include "..\Other\CustomQDataStream.h"

ImageLibrary::ImageLibrary(const size_t t_imageSize)
    : m_imageSize{t_imageSize}
{}

//Return if ImageLibrary is equal to other
bool ImageLibrary::operator==(const ImageLibrary &t_other) const
{
    //Compare image size
    if (m_imageSize != t_other.m_imageSize)
        return false;

    //Compare number of images in library
    if (getImages().size() != t_other.getImages().size())
        return false;

    //Compare library images
    for (auto [im, otherIm] = std::pair{getImages().cbegin(), t_other.getImages().cbegin()};
         im != getImages().cend(); ++im, ++otherIm)
    {
        if (im->size != otherIm->size)
            return false;

        if (cv::sum(*im != *otherIm)[0] != 0)
            return false;
    }

    return true;
}

//Set image size
void ImageLibrary::setImageSize(const size_t t_size)
{
    if (t_size == m_imageSize)
        return;

    m_imageSize = t_size;

    //Resize library images to new size
    ImageUtility::batchResizeMat(m_originalImages, m_resizedImages, static_cast<int>(t_size),
                                 static_cast<int>(t_size), ImageUtility::ResizeType::EXACT);
}

//Returns image size
size_t ImageLibrary::getImageSize() const
{
    return m_imageSize;
}

//Add image to library with given name at random index
//Returns the index
size_t ImageLibrary::addImage(const cv::Mat &t_im, const QString &t_name)
{
    //Empty image, do not add
    if (t_im.empty())
        throw std::invalid_argument("t_im was empty.");

    //Square image
    cv::Mat squaredIm = t_im;
    if (t_im.rows != t_im.cols)
    {
        ImageUtility::imageToSquare(squaredIm, ImageUtility::SquareMethod::CROP);
    }

    //Insert image in a random position
    std::random_device rand_dev;
    std::mt19937 generator(rand_dev());
    std::uniform_int_distribution<size_t> distr(0, m_originalImages.size());
    const size_t randomIndex = distr(generator);

    m_names.insert(m_names.begin() + randomIndex, t_name);

    addImageInternal(randomIndex, ImageUtility::resizeImage(squaredIm, static_cast<int>(m_imageSize), static_cast<int>(m_imageSize), ImageUtility::ResizeType::EXACT));

    return randomIndex;
}

//Returns const reference to library image names
const std::vector<QString> &ImageLibrary::getNames() const
{
    return m_names;
}

//Returns const reference to library images
const std::vector<cv::Mat> &ImageLibrary::getImages() const
{
    return m_resizedImages;
}

//Removes the image at given index
void ImageLibrary::removeAtIndex(const size_t t_index)
{
    m_names.erase(m_names.begin() + t_index);
    m_originalImages.erase(m_originalImages.begin() + t_index);
    m_resizedImages.erase(m_resizedImages.begin() + t_index);
}

//Clear image library
void ImageLibrary::clear()
{
    m_names.clear();
    m_originalImages.clear();
    m_resizedImages.clear();
}

//Saves the image library to the given file
void ImageLibrary::saveToFile(const QString t_filename) const
{
    if (t_filename.isNull())
        throw std::invalid_argument("No filename");
    else
    {
        QFile file(t_filename);
        file.open(QIODevice::WriteOnly);
        if (!file.isWritable())
            throw std::invalid_argument("File is not writable: " + t_filename.toStdString());
        else
        {
            CustomQDataStream out(&file, CustomQDataStream::CVMatMode::ENCODE_PNG);
            QDataStream *outParent = &out;

            //Write header with "magic number" and version
            *outParent << MIL_MAGIC;
            *outParent << MIL_VERSION;

            out.setVersion(QDataStream::Qt_5_0);

            //Write image size and library size
            *outParent << static_cast<quint32>(m_imageSize);
            *outParent << static_cast<quint32>(m_resizedImages.size());

            //Write images and names
            for (auto [image, name] = std::pair{m_resizedImages.cbegin(), m_names.cbegin()};
                 image != m_resizedImages.cend(); ++image, ++name)
            {
                out << *image;
                *outParent << *name;
            }

            file.close();
        }
    }
}

//Loads image library from given file
void ImageLibrary::loadFromFile(const QString t_filename)
{
    if (t_filename.isNull())
        throw std::invalid_argument("No filename");
    {
        //Check for valid file
        QFile file(t_filename);
        file.open(QIODevice::ReadOnly);
        if (!file.isReadable())
            throw std::invalid_argument("File is not readable: " + t_filename.toStdString());
        else
        {
            CustomQDataStream in(&file, CustomQDataStream::CVMatMode::ENCODE_PNG);
            QDataStream *inParent = &in;

            //Read and check magic number
            quint32 magic;
            *inParent >> magic;
            if (magic != MIL_MAGIC)
                throw std::invalid_argument("File is not a valid .mil: "
                                            + t_filename.toStdString());

            //Read the version
            quint32 version;
            *inParent >> version;
            if (version <= MIL_VERSION && version >= 4)
                in.setVersion(QDataStream::Qt_5_0);
            else
            {
                if (version < MIL_VERSION)
                    throw std::invalid_argument(".mil uses an outdated file version: "
                                                + t_filename.toStdString());
                else
                    throw std::invalid_argument(".mil uses a newer file version: "
                                                + t_filename.toStdString());
            }

            //Encoding of images in .mil was introduced in version 6
            //For any mil before then we need to load with no encoding
            if (version < MIL_VERSION_ENCODED)
                in.SetCVMatMode(CustomQDataStream::CVMatMode::RAW);

            //Read image size
            quint32 imageSize;
            *inParent >> imageSize;
            m_imageSize = static_cast<size_t>(imageSize);

            //Read library size
            quint32 numberOfImage;
            *inParent >> numberOfImage;

            //Read images and names
            while (numberOfImage > 0)
            {
                --numberOfImage;
                cv::Mat image;
                in >> image;

                QString name;
                in >> name;

                size_t index = m_originalImages.size();
                //Random sorting of library images was introduced in version 5
                //For any mil before then we need to randomly sort when loading the mil
                if (version < 5)
                {
                    //Insert image in a random position
                    std::random_device rand_dev;
                    std::mt19937 generator(rand_dev());
                    std::uniform_int_distribution<size_t> distr(0, m_originalImages.size());
                    index = distr(generator);
                }

                m_names.insert(m_names.begin() + index, name);
                addImageInternal(index, image);
            }

            file.close();
        }
    }
}

//Internals of addImage, adds image at the given index in relevant containers
//Allows this to be overriden for CUDA functionality while using the same addImage
void ImageLibrary::addImageInternal(const size_t index, const cv::Mat &t_im)
{
    m_originalImages.insert(m_originalImages.begin() + index, t_im);
    m_resizedImages.insert(m_resizedImages.begin() + index, t_im);
}
