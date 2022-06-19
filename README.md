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
| [OpenCV](https://opencv.org/) | >= 4.1.1 | calib3d, core, features2d, flann, highgui, imgcodecs, imgproc, objdetect |

## Optional CUDA dependencies
If you have a [CUDA-capable GPU](https://developer.nvidia.com/cuda-gpus), then you can use the following to generate Photomosaics faster.  
Currently only Windows supports CUDA.

| Name | Version | Modules |
| - | - | - |
| [CUDA](https://developer.nvidia.com/cuda-zone) | >= 10.1 | |


## Linux
With the move to Visual Studio it currently only builds for Windows. At some point I might try ti get it building for Linux again, but feel free to try yourself.

Linux requires [pkg-config](https://www.freedesktop.org/wiki/Software/pkg-config/) for linking OpenCV.

### Ubuntu
The provided [install-requirements-ubuntu.mk](https://github.com/MorganGrundy/MosaicMagnifique/blob/master/install-requirements-ubuntu.mk) makefile can be used to easily install dependencies. Tested on Ubuntu 20.04 + 18.04.  
`make -f install-ubuntu.mk all`  
or instead can install dependencies separately:  
`make -f install-ubuntu.mk gcc`  
`make -f install-ubuntu.mk pkg-config`  
`make -f install-ubuntu.mk qmake`  
`make -f install-ubuntu.mk qt`  
`make -f install-ubuntu.mk opencv`  

### Other
#### Qt
Use installer or build from source: https://doc.qt.io/qt-5/gettingstarted.html

#### OpenCV
Build from source: https://docs.opencv.org/master/d7/d9f/tutorial_linux_install.html  
In configuring step give cmake: -DOPENCV_GENERATE_PKGCONFIG=ON -DCMAKE_BUILD_TYPE=Release  
And for minimal build give cmake module list: -DBUILD_LIST=calib3d,core,features2d,flann,highgui,imgcodecs,imgproc,objdetect

#### Mosaic Magnifique
Download source from: https://github.com/MorganGrundy/MosaicMagnifique/releases  
Open project in Visual Studio
Build

## Windows
### Batch script
The provided [install-requirements-windows.cmd](https://github.com/MorganGrundy/MosaicMagnifique/blob/master/install-requirements-windows.cmd) batch script can be used to help install OpenCV, but not MSVC/CUDA/Qt.  
It has an additional dependency: [wget](https://www.gnu.org/software/wget/). Set environment variable %wgetdir% to the directory containing wget.exe.  
After installing other dependencies, run the script with admin (Setting OpenCV environment variables requires admin) from command line:  
`set mode=all`  
`install-windows.bat`  

### Manual
#### Qt
Use installer or build from source: https://doc.qt.io/qt-5/gettingstarted.html  
Add Qt bin to %PATH% environment variable.

#### CUDA (Optional)
Download CUDA installer from: https://developer.nvidia.com/cuda-downloads  

#### OpenCV
Build from source: https://docs.opencv.org/master/d3/d52/tutorial_windows_install.html  
In configuring step, give cmake: -DCMAKE_BUILD_TYPE=Release  
And for minimal build give cmake module list: -DBUILD_LIST=calib3d,core,features2d,flann,highgui,imgcodecs,imgproc,objdetect 

#### Mosaic Magnifique
Download source from: https://github.com/MorganGrundy/MosaicMagnifique/releases  
Open project in Visual Studio
Build