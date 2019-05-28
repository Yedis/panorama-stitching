//
// Created by yedis on 19-5-27.
//

#ifndef PANORAMA_STITCHING_PANORAMA_H
#define PANORAMA_STITCHING_PANORAMA_H

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "HW6_PA_H.h"

using namespace std;
using namespace cv;

class Panorama0473 : public CylindricalPanorama {

public:
    Panorama0473()= default;

    bool makePanorama(std::vector<cv::Mat>& img_vec,cv::Mat& img_out,double f) override;

private:
    Mat cylindrical(cv::Mat& img, double f);
};


#endif //PANORAMA_STITCHING_PANORAMA_H
