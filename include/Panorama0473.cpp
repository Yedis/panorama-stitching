//
// Created by yedis on 19-5-27.
//
#include "opencv2/opencv.hpp"
#include <opencv2/nonfree/features2d.hpp>

#include "Panorama0473.h"

using namespace std;

bool Panorama0473::makePanorama(std::vector<cv::Mat> &img_vec, cv::Mat &img_out, double f) {
    /*
     * implement of panorama stitching
     * img_vec:         input image vector
     * img_out:         output image Mat
     * f:               camera parameter f
     */

    Mat image1 = img_vec[0];
    Mat image2 = img_vec[1];

    // cylindrical
    image1 = cylindrical(image1, f);
    image2 = cylindrical(image2, f);

    // sift extractor create
    Mat g1(image1, Rect(0, 0, image1.cols, image1.rows));
    Mat g2(image2, Rect(0, 0, image2.cols, image2.rows));
    cvtColor(g1, g1, CV_BGR2GRAY);
    cvtColor(g2, g2, CV_BGR2GRAY);
    SiftFeatureDetector siftdet;
    vector<KeyPoint>kp1, kp2;
    SiftDescriptorExtractor extractor;
    Mat descriptor1, descriptor2;
    FlannBasedMatcher matcher;
    vector<DMatch> matches, good_matches;

    // sift interest point
    siftdet.detect(g1, kp1);
    siftdet.detect(g2, kp2);

    // descriptor
    extractor.compute(g1, kp1, descriptor1);
    extractor.compute(g2, kp2, descriptor2);

    // correspondence
    matcher.match(descriptor1, descriptor2, matches);

    // vis
    Mat  firstmatches;
    drawMatches(image1, kp1, image2, kp2,
                matches, firstmatches, Scalar::all(-1), Scalar::all(-1),
                vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
    imshow("first_matches", firstmatches);
    waitKey(0);

    /* 下面计算向量距离的最大值与最小值 */
    double max_dist = 0; double min_dist = 1000;
    for (int i = 0; i < descriptor1.rows; i++) {
        if (matches[i].distance > max_dist) {
            max_dist = matches[i].distance;
        }
        if (matches[i].distance < min_dist) {
            min_dist = matches[i].distance;
        }
    }
    cout << "The max distance is: " << max_dist << endl;
    cout << "The min distance is: " << min_dist << endl;
    for (int i = 0; i < descriptor1.rows; i++) {
        if (matches[i].distance < 0.1 * max_dist) {
            good_matches.push_back(matches[i]);
        }
    }  // todo
    Mat img_matches;
    /*第二次筛选后的结果*/
    drawMatches(image1, kp1, image2, kp2,
                good_matches, img_matches, Scalar::all(-1), Scalar::all(-1),
                vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
    imshow("good_matches", img_matches);
    waitKey(0);

    vector<Point2f> keypoints1, keypoints2;
    for (int i = 0; i < good_matches.size(); i++) {
        keypoints1.push_back(kp1[good_matches[i].queryIdx].pt);
        keypoints2.push_back(kp2[good_matches[i].trainIdx].pt);
    }

    /*计算单应矩阵*/
    Mat rev_H = findHomography(keypoints2, keypoints1, RANSAC);
    Mat H = findHomography(keypoints1, keypoints2, RANSAC);

    // stitchedImage
    Mat stitchedImage;
    int mRows = image2.rows;
    if (image1.rows> image2.rows)
    {
        mRows = image1.rows;
    }
    /*判断图像在左边还是在右边*/
    int propimg1 = 0, propimg2 = 0;
    for (int i = 0; i < good_matches.size(); i++) {
        if (kp1[good_matches[i].queryIdx].pt.x > image1.cols / 2) {
            propimg1++;
        }
        if (kp2[good_matches[i].trainIdx].pt.x > image2.cols / 2) {
            propimg2++;
        }
    }
    bool flag = false;
    Mat imgright;
    Mat imgleft;
    if ((propimg1 / (good_matches.size() + 0.0)) > (propimg2 / (good_matches.size() + 0.0))) {
        imgleft = image1.clone();
        flag = true;
    }
    else {
        imgleft = image2.clone();
        flag = false;
    }
    if (flag) {
        imgright = image2.clone();
        flag = false;
    }
    else {
        imgright = image1.clone();
    }

    imshow("Left", imgleft);
    waitKey(0);
    imshow("Right", imgright);
    waitKey(0);

//    /*把上边求得的右边的图像经过矩阵H转换到stitchedImage中对应的位置*/
//    warpPerspective(imgright, stitchedImage, H, Size(image2.cols + image1.cols, mRows));
//    /*把左边的图像放进来*/
//    Mat half(stitchedImage, Rect(0, 0, imgleft.cols, imgleft.rows));
//    imgleft.copyTo(half);

    vector<Point2f> corners_1(4);
    vector<Point2f> corners_2(4);
    corners_1[0] = Point2f(0, 0);
    corners_1[1] = Point2f((float)image1.cols, 0);
    corners_1[2] = Point2f((float)image1.cols, (float)image1.rows);
    corners_1[3] = Point2f(0, (float)image1.rows);

    perspectiveTransform(corners_1, corners_2, H);
    int down_rows = (int)min(corners_2[0].y, corners_2[1].y);
    down_rows = min(0, down_rows) * -1;
    int right_cols = (int)min(corners_2[0].x, corners_2[3].x);
    right_cols = min(0, right_cols) * -1;

    Mat stitch_img = Mat::zeros(image2.rows+down_rows, image2.cols+right_cols, CV_8UC3);
    image2.copyTo(Mat(stitch_img, Rect(right_cols, down_rows, image2.cols, image2.rows)));
    for (int i = 0; i < stitch_img.rows; ++i) {
        for (int j = 0; j < stitch_img.cols; ++j) {
            if (stitch_img.at<Vec3b>(i, j) != Vec3b(0, 0, 0)) {
                continue;
            }
            int x0 = j - right_cols;
            int y0 = i - down_rows;
            vector<Point2f> pix, dst;
            pix.emplace_back(x0, y0);
            perspectiveTransform(pix, dst, rev_H);
            Point2f pos = dst[0];
            //cout << pos << endl;
            int x = (int)floor(pos.x);
            int y = (int)floor(pos.y);
            if (0 < y && y < image1.rows && 0 < x && x < image1.cols && image1.at<Vec3b>(y,x) != Vec3b(0,0,0) ) {
                Vec3b c = image1.at<Vec3b>(y, x);
                //if (stitch_img.at<Vec3b>(i,j) != Vec3b(0, 0, 0)) { c += (stitch_img.at<Vec3b>(i,j)-c)/2; }
                stitch_img.at<Vec3b>(i, j) = c;
            }
        }
    }

    imshow("ResultImage.jpg", stitch_img);
    waitKey(0);

}

Mat Panorama0473::cylindrical(cv::Mat &srcImage, double f) {
    int height = srcImage.rows;  // height of origin image
    int width = srcImage.cols;  // width of origin image
    int centerX = width / 2;
    int centerY = height / 2;

    // set size of output image
    int colNum = int(2 * f*atan(0.5*srcImage.cols / f));
    int rowNum = int(0.5*srcImage.rows*f / sqrt(pow(f, 2)) + 0.5*srcImage.rows);
    Mat dstImage = Mat::zeros(rowNum, colNum, CV_8UC3);
//    Mat dstImage = Mat::zeros(srcImage.size(), CV_8UC3);


    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            // calculate value
            int pointX = int(f * atan((j - centerX) / f) + f * atan(width / (2 * f)));
            int pointY = int(f * (i - centerY) / sqrt((j - centerX) * (j - centerX) + f * f) + centerY);

            // set value
            if (pointX >= 0 && pointX < colNum && pointY >= 0 && pointY < rowNum){
                dstImage.at<Vec3b>(pointY, pointX) = srcImage.at<Vec3b>(i, j);
            }
//            if (pointX >= 0 && pointX < width && pointY >= 0 && pointY < height){
//                dstImage.at<Vec3b>(pointY, pointX) = srcImage.at<Vec3b>(i, j);
//            }
        }
    }

    // vis image
//     imshow("srcImage", srcImage);
//     imshow("dstImage", dstImage);
//     waitKey(0);
    return dstImage;
}