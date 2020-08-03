//  main.cpp
//  OpenCV XCode
//
//  Created by William di Domenico
//  Copyright © 2020 Toazza Labs. All rights reserved.
//  Original from https://gist.github.com/EyalAr/3940636

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

using cv::Mat;
using cv::Scalar;
using cv::Point;

int main(int argc, char* argv[])
{
    // Create a gui window:
    cv::namedWindow("Output",1);
    
    // Initialize a 400x900 matrix of black pixels:
    Mat output = Mat::zeros( 400, 900, CV_8UC3 );
    
    // Writes text on the matrix:
    cv::putText(output, "Hello World!", Point(210, 190),
                cv::QT_FONT_NORMAL, 3, Scalar(0,255,0), 4);
    
    // Display the image:
    cv::imshow("Output", output);
    
    // Wait for the user to press any key:
    cv::waitKey(0);
    
    return 0;
}
