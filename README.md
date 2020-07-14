# Photomosaic Generator

A GUI based application for generating Photomosaics.

## Dependencies

| Name | Version | Modules |
| - | - | - |
| [GCC](https://gcc.gnu.org/)/[MinGW](http://www.mingw.org/) | >= 8.2.0 | |
| [MSVC](https://visualstudio.microsoft.com/visual-cpp-build-tools/) | >= 16.2 | |
| | | |
| [Qt](https://www.qt.io/) | >= 5.13.1 | core, gui, widgets |
| [OpenCV](https://opencv.org/) | >= 4.1.1 | core, imgcodecs, imgproc |

## Optional dependencies

If you have a CUDA-capable device, then you can use the following to generate Photomosaics faster.

CUDA usage controlled by "CONFIG += CUDA" in MosaicMagnifique.pro

OpenCV Contrib usage controlled by "CONFIG += OPENCV_W_CUDA" in MosaicMagnifique.pro

| Name | Version | Modules |
| - | - | - |
| [CUDA](https://developer.nvidia.com/cuda-zone) | >= 10.1 | |
| [OpenCV Contrib](https://github.com/opencv/opencv_contrib) | >= 4.1.1 | cudaarithm, cudawarping, cudaimgproc |