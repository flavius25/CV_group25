	#include <opencv2/highgui.hpp>

	#include <iostream>
	
	using namespace cv;

	int main( int argc, char** argv ) {

	cv::Mat image;

	image = cv::imread("Sample.png" , cv::IMREAD_COLOR);

	if(! image.data ) {

	std::cout << "Could not open or find the image" << std::endl ;

	return -1;

	}

	cv::namedWindow("Display window", cv::WINDOW_AUTOSIZE );

	cv::imshow("Display window", image );

	cv::waitKey(0);

	return 0;

	}

