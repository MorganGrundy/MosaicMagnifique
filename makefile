libs = -lopencv_core -lopencv_imgcodecs -lopencv_highgui -lopencv_imgproc -lboost_system -lboost_filesystem

.PHONY: all PhotoMosaic Preprocess clean

all: PhotoMosaic Preprocess

PhotoMosaic:
	g++ -Werror -Wall PhotoMosaic.cpp shared.cpp ImageComparison.cpp -o PhotoMosaic $(libs)

Preprocess:
	g++ -Werror -Wall Preprocess.cpp shared.cpp -o Preprocess $(libs)

clean:
	rm PhotoMosaic Preprocess

mosaic1:
	./PhotoMosaic ~/Pictures/42-north-788021-unsplash.jpg Images cat.jpg

fast-mosaic1:
	./PhotoMosaic ~/Pictures/42-north-788021-unsplash.jpg Images cat.jpg -f
