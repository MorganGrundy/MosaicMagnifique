#ifndef CUDAPHOTOMOSAICGENERATORBASE_H
#define CUDAPHOTOMOSAICGENERATORBASE_H

#include "photomosaicgeneratorbase.h"

//Generates a Photomosaic on GPU using CUDA
class CUDAPhotomosaicGenerator : public PhotomosaicGeneratorBase
{
public:
    CUDAPhotomosaicGenerator(QWidget *t_parent = nullptr);

    //Returns a Photomosaic of the main image made of the library images
    cv::Mat generate();
};

#endif // CUDAPHOTOMOSAICGENERATORBASE_H
