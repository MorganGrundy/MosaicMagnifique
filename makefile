all: PhotoMosaic Preprocess

PhotoMosaic:
	g++ -Werror -Wall PhotoMosaic.cpp shared.cpp -o PhotoMosaic -lopencv_core -lopencv_imgcodecs -lopencv_highgui -lopencv_imgproc

Preprocess:
	g++ -Werror -Wall Preprocess.cpp shared.cpp -o Preprocess -lopencv_core -lopencv_imgcodecs -lopencv_imgproc

clean:
	rm PhotoMosaic Preprocess
