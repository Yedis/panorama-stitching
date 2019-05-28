//
// Created by yedis on 19-5-27.
//

#ifndef PANORAMA_STITCHING_HW6_PA_H_H
#define PANORAMA_STITCHING_HW6_PA_H_H

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace std;
using namespace cv;

class CylindricalPanorama
{
public:
    virtual bool makePanorama(
            std::vector<cv::Mat>& img_vec,cv::Mat& img_out,double f
    ) = 0;
};

#endif //PANORAMA_STITCHING_HW6_PA_H_H
