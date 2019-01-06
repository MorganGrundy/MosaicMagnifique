libs = -lopencv_core -lopencv_imgcodecs -lopencv_highgui -lopencv_imgproc -lboost_system -lboost_filesystem

.PHONY: all PhotoMosaic Preprocess clean

all: PhotoMosaic Preprocess

PhotoMosaic:
	g++ -std=c++11 -Werror -Wall PhotoMosaic.cpp shared.cpp cells.cpp ImageComparison.cpp -o PhotoMosaic $(libs)

Preprocess:
	g++ -std=c++11 -Werror -Wall Preprocess.cpp shared.cpp -o Preprocess $(libs)

clean:
	rm PhotoMosaic Preprocess

cat:
	./PhotoMosaic ~/Pictures/42-north-788021-unsplash.jpg Images cat.jpg

cat-fast:
	./PhotoMosaic ~/Pictures/42-north-788021-unsplash.jpg Images cat.jpg -f
