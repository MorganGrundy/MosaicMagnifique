#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <iostream>

using namespace std;
using namespace cv;
using namespace cv::xfeatures2d;

int main(int argc, char** argv)
{
    //Reads args
    if(argc < 3)
    {
        cout << argv[0] << " path/to/image.1 path/to/image.2" << endl << endl;
        return -1;
    }

    //Loads and checks images
    String imageName1 = argv[1];
    Mat img1 = imread(imageName1, IMREAD_COLOR);
    if (img1.empty()) // Check for invalid input
    {
        cout <<  "Could not open or find the image" << endl;
        return -1;
    }
    String imageName2 = argv[2];
    Mat img2 = imread(imageName2, IMREAD_COLOR);
    if (img2.empty()) // Check for invalid input
    {
        cout <<  "Could not open or find the image" << endl;
        return -1;
    }

    int minHessian = 400;
    SURF* detector = SURF::create(minHessian);

    vector<KeyPoint> keypoints_1, keypoints_2;

    detector->detect(img1, keypoints_1);
    detector->detect(img2, keypoints_2);

    Mat img_keypoints_1, img_keypoints_2;

    drawKeypoints(img1, keypoints_1, img_keypoints_1, Scalar::all(-1), DrawMatchesFlags::DEFAULT);
    drawKeypoints(img2, keypoints_2, img_keypoints_2, Scalar::all(-1), DrawMatchesFlags::DEFAULT);

    imshow("Keypoints 1", img_keypoints_1);
    imshow("Keypoints 2", img_keypoints_2);

    waitKey(0); // Wait for a keystroke in the window
    destroyAllWindows();
    return 0;
}
