#pragma once

#include <opencv2/core.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>

using std::vector;

using cv::Mat;
using cv::Feature2D;
using cv::KeyPoint;
using cv::Ptr;
using cv::Size;
using cv::KeyPoint;
using cv::MSER;
using cv::FeatureDetector;
using cv::DescriptorExtractor;

class Veinprint
{
	protected:

	Ptr<FeatureDetector> keypointDetector = cv::BRISK::create();;

	Ptr<DescriptorExtractor> descriptorExtractor = cv::SIFT::create();

	Mat mRawBinarized;

	Mat mBinarized;	

	public:

	Mat extractVeinprint(Mat inROI, Mat& outThresholded,
						 double pThreshold, int pRatio,
						 int pKernelSize, bool pL2gradient);

	vector<KeyPoint> extractKeypoints(const Mat& inVeinprint);

	Mat computeDescriptors(Mat inVeinprint, vector<KeyPoint> inKeyPoints);

	// Legacy
	Mat extract(Mat inImage, vector<KeyPoint>& outKeypoint,
				Mat& outThresholded, Mat& outMorphed,
				double pThreshold, int pRatio,
				int pKernelSize, bool pL2gradient) const;
};
