//  main.cpp
//  OpenCV XCode
//
//  Created by Don Toazza on 19/07/20.
//  Copyright © 2020 Toazza Labs. All rights reserved.
//

#include <iostream>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/xfeatures2d.hpp>

#include "Veinprint.h"
#include "Utils.h"

using std::cout;
using cv::Mat;

Veinprint vp = Veinprint();

int main(int argc, const char* argv[])
{
    //create a gui window:
    cv::namedWindow("Output", 1);
    Mat thresh;
    //initialize a 120X350 matrix of black pixels:
    Mat ddl1 = cv::imread("/Users/don/Pictures/ddl1.jpg");
    Mat ddl2 = cv::imread("/Users/don/Pictures/ddl2.jpg");
    
    vector<KeyPoint> keypoints1 = vp.extractKeypoints(ddl1);
    vector<KeyPoint> keypoints2 = vp.extractKeypoints(ddl2);
    
    vp.computeDescriptors(ddl1, keypoints1);
    vp.computeDescriptors(ddl2, keypoints2);

    cv::drawKeypoints(ddl2, keypoints1, ddl2);
    
    //display the image:
    cv::imshow("Output", ddl2);
    
    //wait for the user to press any key:
    cv::waitKey(0);
    
    return 0;
}
