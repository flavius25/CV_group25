// Test.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

//#include "stdafx.h"
#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;

int main(int argc, char** argv)
{
	// Read the image file
	Mat image = imread("C:/Utrecht_Stuff/P3/Computer Vision/Projects/Assignment1_Camera_Calibration/brain_1.jpg");

	int x = image.size().width;
	int y = image.size().height;

	Rect r = Rect((x-100)/2, (y-100)/2, 100, 100);
	//create a Rect with top-left vertex at (10,20), of width 100 and height 100 pixels.

	// Check for failure
	if (image.empty())
	{
		cout << "Could not open or find the image" << endl;
		cin.get(); //wait for any key press
		return -1;
	}

	rectangle(image, r, Scalar(255, 255, 255), 4, 8, 0);
	//draw the rect defined by r with line thickness 1 and Blue color

	imwrite("myImageWithRect.jpg", image);

	String windowName = "The Skull"; //Name of the window

	namedWindow(windowName); // Create a window

	imshow(windowName, image); // Show our image inside the created window.

	waitKey(0); // Wait for any keystroke in the window

	destroyAllWindows(); //destroy the created window //destroyWindow(windowName) bugged - exception not handled

	

	return 0;
}
