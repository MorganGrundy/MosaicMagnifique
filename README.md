# Photomosaic Generator
A GUI based application for generating Photomosaics.

## Pre-built
Download the pre-built Windows app from: https://github.com/MorganGrundy/MosaicMagnifique/releases  
Only use the CUDA version if you have a [CUDA-capable GPU](https://developer.nvidia.com/cuda-gpus) or the app will just crash.  
You may need to run the included vc_redist executable first.


## Dependencies
| Name | Version | Modules |
| - | - | - |
| [GCC](https://gcc.gnu.org/)/[MinGW](http://www.mingw.org/) <br> or <br> [MSVC](https://visualstudio.microsoft.com/visual-cpp-build-tools/) | >= 5.3.1 <br> <br> >= 2017 | |
| [Qt](https://www.qt.io/) | >= 5.9.5 | core, gui, svg, widgets |
| [OpenCV](https://opencv.org/) | >= 4.1.1 | core, highgui, imgcodecs, imgproc |

## Optional dependencies
If you have a [CUDA-capable GPU](https://developer.nvidia.com/cuda-gpus), then you can use the following to generate Photomosaics faster.  
Currently only Windows .pro file has CUDA linking setup.

| Name | Version | Modules |
| - | - | - |
| [CUDA](https://developer.nvidia.com/cuda-zone) | >= 10.1 | |
| [OpenCV Contrib](https://github.com/opencv/opencv_contrib) | >= 4.1.1 | cudaarithm, cudawarping, cudaimgproc |

CUDA usage controlled by "CONFIG += CUDA" in .pro file.  
OpenCV Contrib usage controlled by "CONFIG += OPENCV_W_CUDA" in .pro file.  
*Note: OpenCV Contrib requires CUDA.*

## Linux
Linux .pro file requires [pkg-config](https://www.freedesktop.org/wiki/Software/pkg-config/) for linking OpenCV.

### Ubuntu
The provided [install-ubuntu.mk](https://github.com/MorganGrundy/MosaicMagnifique/blob/master/install-ubuntu.mk) makefile can be used to easily install dependencies and build Mosaic Magnifique. Tested on Ubuntu 20.04.  
`make -f install-ubuntu.mk all`  
or instead can install dependencies separately:  
`make -f install-ubuntu.mk gcc`  
`make -f install-ubuntu.mk pkg-config`  
`make -f install-ubuntu.mk qmake`  
`make -f install-ubuntu.mk qt`  
`make -f install-ubuntu.mk opencv`  
`make -f install-ubuntu.mk build`  

### Other
#### Qt
Use installer or build from source: https://doc.qt.io/qt-5/gettingstarted.html

#### OpenCV
Build from source: https://docs.opencv.org/master/d7/d9f/tutorial_linux_install.html  
In configuring step give cmake: -DOPENCV_GENERATE_PKGCONFIG=ON -DCMAKE_BUILD_TYPE=Release  
And for minimal build give cmake module list: -DBUILD_LIST=core,highgui,imgcodecs,imgproc

#### Mosaic Magnifique
Download source from: https://github.com/MorganGrundy/MosaicMagnifique/releases  
Create sub-directory "build"  
From build run:  
`qmake ../src/MosaicMagnifique-Linux.pro`  
`make`

## Windows
### Batch script
The provided [install-windows.cmd](https://github.com/MorganGrundy/MosaicMagnifique/blob/master/install-windows.cmd) batch script can be used to help install OpenCV and build Mosaic Magnifique, but not MSVC/CUDA/Qt.  
It has an additional dependency: [wget](https://www.gnu.org/software/wget/). Set environment variable %wgetdir% to the directory containing wget.exe.  
After installing other dependencies, run the script with admin (Setting OpenCV environment variables requires admin) from command line:  
`set mode=all`  
`install-windows.bat`  
If you have installed OpenCV manually then instead:  
`set mode=build`  
`install-windows.bat`

### Manual
#### MSVC
Download MSVC installer from: https://visualstudio.microsoft.com/downloads/  
Run installer and select Workload "Desktop development with C++", the minimum needed is MSVC C++ x64/x86 build tools and Windows SDK.

#### Qt
Use installer or build from source: https://doc.qt.io/qt-5/gettingstarted.html  
Add Qt bin to %PATH% environment variable.

#### CUDA (Optional)
Download CUDA installer from: https://developer.nvidia.com/cuda-downloads  

#### OpenCV
Build from source: https://docs.opencv.org/master/d3/d52/tutorial_windows_install.html  
In configuring step, give cmake: -DCMAKE_BUILD_TYPE=Release  
And for minimal build give cmake module list: -DBUILD_LIST=core,highgui,imgcodecs,imgproc  
  
If you are using CUDA you can give cmake: -DWITH_CUDA:BOOL=ON -DOPENCV_EXTRA_MODULES_PATH="C:/Path to/OpenCV Contrib/modules"  
And add the relevant contrib modules to module list: -DBUILDLIST=core,highgui,imgcodecs,imgproc,cudaarithm,cudawarping,cudaimgproc,cudafilters

#### Mosaic Magnifique
Download source from: https://github.com/MorganGrundy/MosaicMagnifique/releases  
Create sub-directory "build"  
From build run:  
`qmake ../src/MosaicMagnifique-Windows.pro -spec win32-msvc`  
`jom qmake_all`  
`jom`