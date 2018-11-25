libs = -lopencv_core -lopencv_imgcodecs -lopencv_highgui -lopencv_imgproc -lopencv_features2d -lopencv_xfeatures2d

.PHONY: all FeatureDetection clean

all: FeatureDetection

FeatureDetection:
	g++ FeatureDetection.cpp -o FeatureDetection $(libs)

clean:
	rm FeatureDetection
