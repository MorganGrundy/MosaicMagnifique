libs = -lopencv_core -lopencv_imgcodecs -lopencv_highgui -lopencv_imgproc -lboost_system -lboost_filesystem
img = abbs
ext = .jpg

.PHONY: BUILD BUILD-PhotoMosaic BUILD-Preprocess BUILD-clean PrePics def 2000 hex circ star star2

BUILD: BUILD-PhotoMosaic BUILD-Preprocess

BUILD-PhotoMosaic:
	g++ -std=c++11 -Werror -Wall PhotoMosaic.cpp shared.cpp cells.cpp ImageComparison.cpp -o PhotoMosaic $(libs)

BUILD-Preprocess:
	g++ -std=c++11 -Werror -Wall Preprocess.cpp shared.cpp -o Preprocess $(libs)

BUILD-clean:
	rm PhotoMosaic Preprocess

PrePics:
	./Preprocess ~/Pictures Images

def:
	./PhotoMosaic Input/$(img)$(ext) Images Results/$(img)$(ext)

2000:
	./PhotoMosaic Input/$(img)$(ext) Images Results/$(img)2000$(ext) -c

hex:
	./PhotoMosaic Input/$(img)$(ext) Images Results/$(img)Hex$(ext) -cs "Hexagon" -s 10

circ:
	./PhotoMosaic Input/$(img)$(ext) Images Results/$(img)Circ$(ext) -cs "Circle" -s 10

star:
	./PhotoMosaic Input/$(img)$(ext) Images Results/$(img)Star$(ext) -cs "6PointStar1" -s 10

star2:
	./PhotoMosaic Input/$(img)$(ext) Images Results/$(img)Star2$(ext) -cs "6PointStar2" -s 10
