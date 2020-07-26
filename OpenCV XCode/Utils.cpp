#include "Utils.h"

#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/videoio.hpp>

//#include <QPixmap>
//#include <QElapsedTimer>
//#include <QMutex>

using cv::VideoCapture;

// Natural color system shades
vector<Scalar> Utils::COLOR = { Scalar(189, 135, 0),	// Blue
Scalar(107, 159, 0),	// Green
Scalar(51, 2, 196),		// Red
Scalar(0, 211, 255) }; 	// Yellow


vector< vector<Point> > Utils::combinations(vector<Point> pPoints, int pRange)
{
	vector< vector<Point> > combinations;

	int n = pPoints.size();

	vector<bool> v(n);

	std::fill(v.begin() + pRange, v.end(), true);

	do
	{
		combinations.push_back(vector<Point>());

		for (int i = 0; i < n; ++i)
		{
			if (!v[i])
			{
				combinations.back().push_back(pPoints[i]);
			}
		}
	}
	while (std::next_permutation(v.begin(), v.end()));

	return combinations;
}

double Utils::distance(const Point& pFirstPoint, const Point& pSecondPoint)
{
	return cv::norm(Mat(pFirstPoint), Mat(pSecondPoint));
}

Mat Utils::copy(const Mat& pInput, int pCode, int pChannels)
{
	Mat output = pInput.clone();

	if (pCode != -1)
	{
		if (pInput.channels() < 3)
		{
			//TODO: fix this 
			//cv::cvtColor(output, output, pCode, pChannels);
			cv::cvtColor(output, output, cv::COLOR_GRAY2BGR, pChannels);
		}
	}

	return output;
}

Mat Utils::drawMinima(Mat pImage, vector<Point> pMinima, Scalar pColor, bool pIsShowingNumbers)
{
	for (unsigned int i = 0; i < pMinima.size(); i++)
	{
		Point minimum = pMinima[i];

		cv::circle(pImage, minimum, 3,
			pColor, 2);

		if (pIsShowingNumbers)
		{
			cv::putText(pImage, std::to_string(i + 1), minimum,
				cv::FONT_HERSHEY_SIMPLEX, 0.75,
				Scalar(255, 255, 255));
		}
	}
	return pImage;
}

Mat Utils::drawCircles(Mat pImage, vector<Point> pPoints, Scalar pColor, int pRadius, int pThickness)
{
	for (Point point : pPoints)
	{
		cv::circle(pImage, point, pRadius, pColor, pThickness);
	}

	return pImage;
}

float Utils::xSlope(Point pFirstPoint, Point pSecondPoint)
{
	float xSlope = 1 / ySlope(pFirstPoint, pSecondPoint);

	return xSlope;
}

float Utils::ySlope(Point pFirstPoint, Point pSecondPoint)
{
	Point2f localDelta = delta(pFirstPoint, pSecondPoint);

	float ySlope = localDelta.y / localDelta.x;

	return ySlope;
}

Point2f Utils::delta(Point pFirstPoint, Point pSecondPoint)
{
	float deltaX = static_cast<float>(pSecondPoint.x - pFirstPoint.x);

	float deltaY = static_cast<float>(pSecondPoint.y - pFirstPoint.y);

	return Point2f(deltaX, deltaY);
}

bool Utils::equals(Mat pFirstMatrix, const Mat pSecondMatrix)
{
	// treat two empty mat as identical as well
	if (pFirstMatrix.empty() && pSecondMatrix.empty())
	{
		return true;
	}

	// if dimensionality of two mat is not identical, these two mat is not identical
	if (pFirstMatrix.cols != pSecondMatrix.cols || pFirstMatrix.rows != pSecondMatrix.rows || pFirstMatrix.dims != pSecondMatrix.dims)
	{
		return false;
	}

	if (pFirstMatrix.channels() > 1)
	{
		cv::cvtColor(pFirstMatrix, pFirstMatrix, cv::COLOR_BGR2GRAY, 1);
	}

	if (pSecondMatrix.channels() > 1)
	{
		cv::cvtColor(pSecondMatrix, pSecondMatrix, cv::COLOR_BGR2GRAY, 1);
	}

	Mat diff;

	cv::compare(pFirstMatrix, pSecondMatrix, diff, cv::CMP_NE);

	int nz = cv::countNonZero(diff);

	return nz == 0;
}

vector<Point> Utils::remove(const vector<Point>& pVector, const vector<Point>& pSubVector)
{
	vector<Point> superVectorCopy(pVector);

	for (Point p : pSubVector)
	{
		superVectorCopy.erase(std::remove(superVectorCopy.begin(), superVectorCopy.end(), p), superVectorCopy.end());
	}

	return superVectorCopy;
}

vector<Point> Utils::getMinima(vector<Point> pPoints, vector<double> pDistances, int pRange)
{
	// If range is larger than half the amount of point distances
	if (pRange > 0.5 * pDistances.size())
	{	// Set range to half the distance size
		pRange = 0.5 * pDistances.size();
	}

	vector<Point> minima;

	Mat plot(700, pDistances.size(), CV_8UC4, Scalar(0, 0, 0));

	//TODO: Sometimes the points and distances vector arguments are send with same size but 
	//TODO: for some reason the size of pDistances get changed to a larger amount.
	//TODO: This is a temp fix
	//for (int i = pRange; i < pDistances.size() - pRange; i++)
	for (int i = pRange; i < pPoints.size() - pRange; i++)
	{
		bool isMinimum = true;

		for (int j = 1; j <= pRange; j++)
		{
			if (pDistances[i] > pDistances[i - j] ||
				pDistances[i] > pDistances[i + j])
			{
				isMinimum = false;

				break;
			}
		}

		if (isMinimum)
		{
			minima.push_back(pPoints[i]);

			cv::circle(plot,
				Point(i, pDistances[i]),
				5,
				Scalar(0, 0, 255));

			cv::putText(plot, std::to_string(minima.size()), Point(i, pDistances[i]), cv::FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255, 255, 255));
		}
		else
		{
			cv::circle( plot,
						Point(i, pDistances[i]),
						1,
						Scalar(255, 255, 255) );
		}
	}

	//show(plot);

	return minima;
}

cv::Size Utils::scale(const cv::Size& pOriginalSize, const float& pScaledLargestSide)
{
	int originalLargestSide;

	if (pOriginalSize.width >= pOriginalSize.height)
	{
		originalLargestSide = pOriginalSize.width;
	}
	else
	{
		originalLargestSide = pOriginalSize.height;
	}

	float scaleFactor = pScaledLargestSide / originalLargestSide;

	float scaledWidth = pOriginalSize.width * scaleFactor;

	float scaledHeight = pOriginalSize.height * scaleFactor;

	cv::Size scaledSize(scaledWidth, scaledHeight);

	return scaledSize;
}

template <typename T>
vector<T> Utils::concatenate(vector<T> startVector, vector<T> endVector)
{
	startVector.insert(startVector.end(), endVector.begin(), endVector.end());

	return vector<T>();
}

// Uses a number and returs an string with fixed size,
// adding trailing zeroes as necessary
string Utils::addTrailingZeroes(int pNumber, int pDigits)
{
	string numberString = std::to_string(pNumber);

	std::string trailingZero = std::string(pDigits - numberString.length(), '0');

	return trailingZero + numberString;
}

// Draw matches from one image to a set of images
void Utils::drawMatches( Mat image1, vector<KeyPoint> keypoints1,
						vector<Mat> image2, vector<vector<KeyPoint>> keypoints2,
						vector<DMatch> matches, Mat& outImage)
{
	// Sort matches vector by which image it belongs
	vector <vector<DMatch>> orderedMatches(keypoints2.size());

	for (DMatch match : matches)
	{
		orderedMatches[match.imgIdx].push_back(match);
	}

	// Find largest width and height from image set	
	int largestWidth = 0;
	
	int largestHeight = image1.size().height;	

	for (Mat currentImage : image2)
	{
		if (currentImage.size().width > largestWidth)
		{
			largestWidth = currentImage.size().width;
		}
		
		if (currentImage.size().height > largestHeight)
		{
			largestHeight = currentImage.size().height;
		}
	}	

	// Builds an output image with the largest height and width
	// so there is no chance a keypoint will be out of bounds
	// and throw an error
	outImage = Mat(largestHeight, largestWidth + image1.size().width, CV_8UC3);	

	// Draw first matches outside loop, so it initializes background
	cv::drawMatches(image1, keypoints1, image2[0], keypoints2[0], orderedMatches[0],
					outImage, Utils::COLOR[0], Utils::COLOR[0]);
    
	// From second match onwards draw only 
	// the lines, otherwise the background 
	// would cover previous matches draws
	for (int i = 1; i < orderedMatches.size(); i++)
	{
		Scalar currentColor = Utils::COLOR[i % 4];

		cv::drawMatches(image1, keypoints1,
			image2[0], keypoints2[i],
			orderedMatches[i], outImage,
			currentColor, currentColor,
			std::vector<char>(),
			cv::DrawMatchesFlags::DRAW_OVER_OUTIMG);
        
        cv::drawMatches(image1, keypoints1,
                        image2[0], keypoints2[i],
                        orderedMatches[i], outImage);
	}	
}

Mat Utils::concat(const Mat& top, const Mat& bottom)
{
	Mat returnValue;

	// Left padding
	int leftPaddingCols = (top.cols - bottom.cols) / 2;

	Mat leftPadding(bottom.rows,
		leftPaddingCols,
		top.type(),
		Scalar(0, 0, 0));

	// Concatenate bottom image with left padding
	cv::hconcat(leftPadding, bottom, returnValue);

	// Right padding
	int rightPaddingCols = top.cols - bottom.cols - leftPaddingCols;

	Mat rightPadding( returnValue.rows,
					  rightPaddingCols,
					  top.type(),
					  Scalar(0, 0, 0) );

	// Concatenate bottom image with right padding
	cv::hconcat(returnValue, rightPadding, returnValue);

	// Concatenate top with padded bottom image
	cv::vconcat(top, returnValue, returnValue);

	return returnValue;
}

// Frames per second to milliseconds per frame
double Utils::FPStoMSPF(double fps)
{
	// ms/f = [1 / (f/s)] * 1000
	return 1000.f / fps;
}
/*
bsoncxx::types::b_binary Utils::toMongo(Mat cvImage)
{
	vector<uchar>* byteVector = new vector<uchar>;

	cv::imencode(".png", cvImage, *byteVector);

	bsoncxx::types::b_binary outMongoBinary{ bsoncxx::binary_sub_type::k_binary,
		static_cast<uint32_t>(byteVector->size()),
		byteVector->data() };

	return outMongoBinary;
}

QPixmap Utils::toQPixmap(const Mat& inMat)
{
	Mat matClone = inMat.clone();

	QImage outQImage;

	switch (matClone.type())
	{
	// 8-bit, 4 channel
	case CV_8UC4:
	{
		outQImage = QImage(matClone.data,
			matClone.cols, matClone.rows,
			static_cast<int>(matClone.step),
			QImage::Format_ARGB32);

		// QT is RGB and OpenCV BGR
		// This swaps red and blue channels
		outQImage = outQImage.rgbSwapped();

		break;
	}
	case CV_8UC3:
	{
		outQImage = QImage(matClone.data,
			matClone.cols, matClone.rows,
			static_cast<int>(matClone.step),
			QImage::Format_RGB888);

		outQImage = outQImage.rgbSwapped();

		break;
	}
	case CV_8UC1:
	{
		outQImage = QImage(inMat.data,
			inMat.cols, inMat.rows,
			inMat.step,
			QImage::Format_Grayscale8);

		break;
	}
	}

	return QPixmap::fromImage(outQImage);
}

cv::Mat Utils::toOpenCV(const bsoncxx::types::b_binary& mongoImage)
{
	Mat outMat = cv::imdecode(std::vector<uint8_t>(mongoImage.bytes, mongoImage.bytes + mongoImage.size),
		cv::IMREAD_GRAYSCALE);

	return outMat;
}
*/
Mat Utils::rotate(const Mat& in, int angle)
{
	Mat out = in;

	for (int i = 0; i < angle; i += 90)
	{
		cv::transpose(out, out);

		cv::flip(out, out, 1);
	}

	cv::flip(out, out, 1);

	return out;
}
/*
void Utils::flush(VideoCapture& vc)
{
	int delay = 0;
	QElapsedTimer timer;

	int framesWithDelayCount = 0;
	while (framesWithDelayCount <= 1)
	{
		timer.start();

		vc.grab();				 // Call VideoCapture::grab() and after that call the slower method VideoCapture::retrieve() 										
								 // to decode and get frame from each camera. This way the overhead on demosaicing or motion jpeg decompression etc. 
		delay = timer.elapsed(); // is eliminated and the retrieved frames from different cameras will be closer in time.

		if (delay > 0)
		{
			framesWithDelayCount++;
		}

		cout << "delay: " << delay << endl;
	}
}

void Utils::flush(VideoCapture& vc, QMutex cameraSemaphore)
{
	int delay = 0;
	QElapsedTimer timer;

	int framesWithDelayCount = 0;
	while (framesWithDelayCount <= 1)
	{
		
		timer.start();
		
		cameraSemaphore.lock();
		vc.grab();				 // Call VideoCapture::grab() and after that call the slower method VideoCapture::retrieve() 										
		cameraSemaphore.unlock();  // to decode and get frame from each camera. This way the overhead on demosaicing or motion jpeg decompression etc. 
								 // is eliminated and the retrieved frames from different cameras will be closer in time.
		delay = timer.elapsed(); 

		if (delay > 0)
		{
			framesWithDelayCount++;
		}

		cout << "delay: " << delay << endl;
	}
}
*/
