all: dependencies build

dependencies: gcc pkg-config qmake qt opencv

gcc:
	# install gcc
	sudo apt-get install build-essential

pkg-config:
	# install pkg-config
	sudo apt-get install pkg-config

qmake:
	# install qmake
	sudo apt install qtchooser

qt:
	# install qt
	sudo apt-get install qt5-default

opencv:
	# install opencv required packages
	sudo apt-get install cmake git libgtk2.0-dev libavcodec-dev libavformat-dev libswscale-dev
	# download opencv source
	sudo apt-get install wget
	sudo apt-get install unzip
	wget github.com/opencv/opencv/archive/4.3.0.zip -O opencv-4.3.0.zip
	unzip opencv-4.3.0.zip
	# build opencv source
	mkdir opencv-4.3.0/build
	cd opencv-4.3.0/build && cmake -DBUILD_LIST=core,imgcodecs,imgproc,highgui -DOPENCV_GENERATE_PKGCONFIG=ON -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/usr/local .. && make -j7 && sudo make install
	# ensures opencv shared libraries link
	sudo ldconfig

build:
	# build Mosaic Magnifique
	mkdir build
	cd build && qmake ../src/MosaicMagnifique-Linux.pro && make

clean:
ifneq (,$(wildcard ./opencv-4.3.0.zip))
	rm opencv-4.3.0.zip
endif
ifneq (,$(wildcard ./opencv-4.3.0))
	rm -r opencv-4.3.0
endif
ifneq (,$(wildcard ./build))
	rm -r build
endif
