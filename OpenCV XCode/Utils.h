#pragma once

#include <iostream>
#include <vector>

//#include <bsoncxx/types/value.hpp>
#include <opencv2/core/types.hpp>
/*
namespace cv 
{
	class VideoCapture;
	class Point;
	class Point2f;
	class Mat;
	class Scalar;
	class KeyPoint;
	class DMatch;
}
*/

class QMutex;
class QPixmap;

namespace cv 
{
	class Mat;
	class VideoCapture;
}

using std::cout;
using std::cin;
using std::endl;
using std::string;
using std::vector;

using cv::Point;
using cv::Point2f;
using cv::Mat;
using cv::Scalar;
using cv::KeyPoint;
using cv::DMatch;

class Utils
{
public:

	enum Key { ENTER = 13, SPACE = 32, ESC = 27, MSFP = 1000 / 30 };

	static vector<Scalar> COLOR;

	static vector< vector<Point> > combinations(vector<Point> pPoints, int pRange = 3);

	static double distance(const Point& pFirstPoint, const Point& pSecondPoint);

	static Mat copy(const Mat& pInput, int pCode = -1, int pChannels = 3);

	static Mat drawMinima(Mat pImage, vector<Point> pMinima, Scalar pColor = Scalar(0, 0, 255), bool pIsShowingNumbers = true);

	static Mat drawCircles(Mat pImage, vector<Point> pPoints, Scalar pColor = Scalar(0, 122, 200), int pRadius = 5, int pThickness = -1);

	static float xSlope(Point pFirstPoint, Point pSecondPoint);

	static float ySlope(Point pFirstPoint, Point pSecondPoint);

	static Point2f delta(Point pFirstPoint, Point pSecondPoint);

	static bool equals(Mat pFirstMatrix, const Mat pSecondMatrix);

	static vector<Point> remove(const vector<Point>& pVector, const vector<Point>& pSubVector);

	static vector<Point> getMinima(vector<Point> pPoints, vector<double> pDistances, int pRange);

	static cv::Size scale(const cv::Size& pOriginalSize, const float& pScaledLargestSide);

	template <typename T>
	static vector<T> concatenate(vector<T> startVector, vector<T> endVector);

	static string addTrailingZeroes(int pNumber, int pDigits);

	static void drawMatches(Mat image1, vector<KeyPoint> keypoints1,
							vector<Mat> image2, vector<vector<KeyPoint>> keypoints2,
							vector<DMatch> matches, Mat& outImage);

	static Mat concat(const Mat& top, const Mat& bottom);

	// Frames per second to milliseconds per frame
	static double FPStoMSPF(double fps);

	static Mat rotate(const Mat& in, int angle);
	
	// Convert OpenCV's image to QT's 
	static QPixmap toQPixmap(const cv::Mat& cvImage);

	// Convert OpenCV's image to MongoDB's 
	//static bsoncxx::types::b_binary toMongo(cv::Mat cvImage);

	//static cv::Mat toOpenCV(const bsoncxx::types::b_binary& mongoImage);	

	static void flush(cv::VideoCapture& vc);
	
	void flush(cv::VideoCapture& vc, QMutex cameraSemaphore);
};
