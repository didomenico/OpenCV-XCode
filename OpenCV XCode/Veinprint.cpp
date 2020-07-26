#include "Veinprint.h"

#include <opencv2/core.hpp>
#include <opencv2/photo.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>

using cv::Mat;
using cv::Feature2D;
using cv::KeyPoint;
using cv::Ptr;
using cv::Size;
using cv::KeyPoint;
using cv::MSER;
using cv::FeatureDetector;
using cv::DescriptorExtractor;

Mat Veinprint::extractVeinprint(Mat inROI, Mat& outThresholded,
								double pThreshold, int pRatio,
								int pKernelSize, bool pL2gradient)
{
	// Thresholding
	Mat thresholded;

	cv::blur(inROI, thresholded, cv::Size(3, 3));

	if (thresholded.channels() == 3)
	{
		cv::cvtColor(thresholded, thresholded, cv::COLOR_BGR2GRAY);
	}

	cv::threshold(thresholded, thresholded, 0, 255, cv::THRESH_BINARY + cv::THRESH_OTSU);
	//cv::adaptiveThreshold(thresholded, thresholded, 255, CV_ADAPTIVE_THRESH_GAUSSIAN_C, CV_THRESH_BINARY, 3, 1.2);

	outThresholded = thresholded.clone();

	Mat morphed;
	cv::morphologyEx( thresholded, morphed,
					  cv::MORPH_OPEN,
					  cv::getStructuringElement(cv::MORPH_ELLIPSE, Size(9, 9)));

	// Opening (morphological operation) to remove noise		
	cv::morphologyEx( morphed, morphed,
					  cv::MORPH_CLOSE,
					  cv::getStructuringElement(cv::MORPH_ELLIPSE, Size(3, 3)) );
	
	//return morphed.clone();
	return outThresholded;
}

vector<KeyPoint> Veinprint::extractKeypoints(const Mat& inVeinprint)
{	
	vector<KeyPoint> outKeypoints;
	
	// Keypoint extraction		
	keypointDetector->detect(inVeinprint, outKeypoints);

	return outKeypoints;
}

Mat Veinprint::computeDescriptors(Mat inVeinprint, vector<KeyPoint> inKeyPoints)
{
	Mat descriptors;

	descriptorExtractor->compute(inVeinprint, inKeyPoints, descriptors);

	return descriptors;	
}

//Legacy
Mat Veinprint::extract(Mat inImage, vector<KeyPoint>& outKeypoint,
	Mat& outThresholded, Mat& outMorphed,
	double pThreshold, int pRatio,
	int pKernelSize, bool pL2gradient) const
{
	// Thresholding
	Mat thresholded;

	cv::blur(inImage, thresholded, cv::Size(3, 3));

	cv::threshold(thresholded, thresholded, 0, 255, cv::THRESH_BINARY + cv::THRESH_OTSU);

	outThresholded = thresholded.clone();

	Mat morphed;
	cv::morphologyEx(thresholded, morphed,
		cv::MORPH_OPEN,
		cv::getStructuringElement(cv::MORPH_ELLIPSE, Size(9, 9)));

	// Opening (morphological operation) to remove noise		
	cv::morphologyEx(morphed, morphed,
		cv::MORPH_CLOSE,
		cv::getStructuringElement(cv::MORPH_ELLIPSE, Size(3, 3))
	);

	//outMorphed = morphed.clone();
	outMorphed = thresholded.clone();

	// Keypoint extraction		
	keypointDetector->detect(morphed, outKeypoint);

	// Descriptors computing
	Mat descriptors;
	descriptorExtractor->compute(morphed, outKeypoint, descriptors);

	return descriptors;
}