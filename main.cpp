#include <iostream>
#include <fstream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "include/Panorama0473.h"
#include "include/Panorama0473.cpp"

using namespace std;
using namespace cv;

int main() {
    // read images
    String dir = "../data/panorama-data1";
    vector<Mat> img_vec;
    cout << "Starting load images" << endl;
    for (int i = 1538; i < 1550; ++i) {
        String path = dir + "/DSC0" + to_string(i) + ".JPG";
        cout << "Load image path: " + path << endl;
        Mat image = imread(path, CV_LOAD_IMAGE_COLOR);
        img_vec.push_back(image);
    }
    cout << "Loading images done." << endl;

//    String dir = "../data/panorama-data2";
//    vector<Mat> img_vec;
//    cout << "Starting load images" << endl;
//    for (int i = 1599; i < 1619; ++i) {
//        String path = dir + "/DSC0" + to_string(i) + ".JPG";
//        cout << "Load image path: " + path << endl;
//        Mat image = imread(path, CV_LOAD_IMAGE_COLOR);
//        img_vec.push_back(image);
//    }
//    cout << "Loading images done." << endl;

    // read camera parameter : f
    cout << "Starting load camera parameter" << endl;
    double f;
    ifstream K;
    K.open(dir + "/K.txt");
    K >> f;
    cout << "Load camera parameter done, f: " << f << endl;

    // makePanorama
    cout << "Starting makePanorama" << endl;
    Mat img_out;
    auto * panorama = new Panorama0473();
    panorama->makePanorama(img_vec, img_out, f);

    //


//    Mat image1 = imread("../data/panorama-data1/DSC01538.JPG", CV_LOAD_IMAGE_COLOR);
//    Mat image2 = imread("../data/panorama-data1/DSC01539.JPG", CV_LOAD_IMAGE_COLOR);
//    imshow("image1", image1);
//    imshow("image2", image2);
//    waitKey(0);




}
