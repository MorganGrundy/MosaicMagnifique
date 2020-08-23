/*
    Copyright © 2018-2020, Morgan Grundy

    This file is part of Mosaic Magnifique.

    Mosaic Magnifique is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    Mosaic Magnifique is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with Mosaic Magnifique.  If not, see <https://www.gnu.org/licenses/>.
*/

#ifndef IMAGELIBRARYEDITOR_H
#define IMAGELIBRARYEDITOR_H

#include <QWidget>
#include <QListWidgetItem>
#include <QProgressBar>
#include <opencv2/core/mat.hpp>

namespace Ui {
class ImageLibraryEditor;
}

class ImageLibraryEditor : public QWidget
{
    Q_OBJECT

public:
    explicit ImageLibraryEditor(QWidget *parent = nullptr);
    ~ImageLibraryEditor();

    //Sets pointer to progress bar
    void setProgressBar(QProgressBar *t_progressBar);

    //Returns size of image library
    size_t getImageLibrarySize() const;

    //Returns image library
    const std::vector<cv::Mat> getImageLibrary() const;

public slots:
    //Loads images
    void addImages();
    //Deletes selected images
    void deleteImages();

    //Resizes image library
    void updateCellSize();

    //Saves the image library to a file
    void saveLibrary();
    //Loads an image library from a file
    void loadLibrary();

signals:
    void imageLibraryChanged(int t_newSize);

private:
    Ui::ImageLibraryEditor *ui;
    QProgressBar *m_progressBar;

    int m_imageSize;
    QMap<QListWidgetItem, std::pair<cv::Mat, cv::Mat>> m_images;
};

#endif // IMAGELIBRARYEDITOR_H
