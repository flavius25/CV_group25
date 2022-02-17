#include <opencv2/opencv.hpp>
//#include <opencv2/calib3d/calib3d.hpp>
//#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
//#include <opencv2/core/core_c.h>
//#include <opencv2/core/core.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/highgui.hpp>
#include <stdio.h>
#include <iostream>

//#define CV_CALIB_CB_ADAPTIVE_THRESH 1
//#define CV_CALIB_CB_FAST_CHECK 8
//#define CV_CALIB_CB_NORMALIZE_IMAGE 2

// Defining the dimensions of checkerboard
int CHECKERBOARD[2]{6,9}; 

using namespace cv;
using namespace std;

int main()
{
	// Creating vector to store vectors of 3D points for each checkerboard image
	std::vector<std::vector<cv::Point3f> > objpoints;

	// Creating vector to store vectors of 2D points for each checkerboard image
	std::vector<std::vector<cv::Point2f> > imgpoints;

	// Defining the world coordinates for 3D points
	std::vector<cv::Point3f> objp;
	for (int i{ 0 }; i < CHECKERBOARD[1]; i++)
	{
		for (int j{ 0 }; j < CHECKERBOARD[0]; j++)
			objp.push_back(cv::Point3f(2.2 * j, 2.2 * i, 0));
	}


	// Extracting path of individual image stored in a given directory
	std::vector<cv::String> images;
	// Path of the folder containing checkerboard images
	std::string path = "./images/*.jpg";

	cv::glob(path, images);

	cv::Mat frame, gray;
	// vector to store the pixel coordinates of detected checker board corners 
	std::vector<cv::Point2f> corner_pts;
	bool success;

	// Looping over all the images in the directory
	for (int i{ 0 }; i < images.size(); i++)
	{
		frame = cv::imread(images[i]);

		int x = frame.size().width;
		int y = frame.size().height;

		//int down_width = 1280;
		//int down_height = 720;
		//Mat resized_up;

		//resize(frame, resized_up, Size(down_width, down_height), INTER_LINEAR);

		cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);

		// Finding checker board corners
		// If desired number of corners are found in the image then success = true  
		success = cv::findChessboardCorners(gray, cv::Size(CHECKERBOARD[0], CHECKERBOARD[1]), corner_pts, CALIB_CB_ADAPTIVE_THRESH | CALIB_CB_FAST_CHECK | CALIB_CB_NORMALIZE_IMAGE);

		/*
		 * If desired number of corner are detected,
		 * we refine the pixel coordinates and display
		 * them on the images of checker board
		*/
		if (success)
		{
			cv::TermCriteria criteria(TermCriteria::EPS | TermCriteria::MAX_ITER, 30, 0.001);

			// refining pixel coordinates for given 2d points.
			cv::cornerSubPix(gray, corner_pts, cv::Size(11, 11), cv::Size(-1, -1), criteria);

			// Displaying the detected corner points on the checker board
			cv::drawChessboardCorners(frame, cv::Size(CHECKERBOARD[0], CHECKERBOARD[1]), corner_pts, success);

			objpoints.push_back(objp);
			imgpoints.push_back(corner_pts);
		}

		//int down_width = 1280;
		//int down_height = 720;
		//Mat resized_up;

		//resize(frame, resized_up, Size(down_width, down_height), INTER_LINEAR);


		//cv::imshow("Image", frame);
		cv::waitKey(0);
	}

	cv::destroyAllWindows();

	cv::Mat cameraMatrix, distCoeffs, R, T;

	/*
	 * Performing camera calibration by
	 * passing the value of known 3D points (objpoints)
	 * and corresponding pixel coordinates of the
	 * detected corners (imgpoints)
	*/
	cv::calibrateCamera(objpoints, imgpoints, cv::Size(gray.rows, gray.cols), cameraMatrix, distCoeffs, R, T);

	std::cout << "cameraMatrix : " << cameraMatrix << std::endl;
	std::cout << "distCoeffs : " << distCoeffs << std::endl;
	std::cout << "Rotation vector : " << R << std::endl;
	std::cout << "Translation vector : " << T << std::endl;


	//VideoCapture capture(0); for when camera is on
	Mat view;
	Mat Img;

	vector<Point3f>  objectPoints_cam;

	const string x = "X";
	const string y = "Y";
	const string z = "Z";
	bool found = false;
	vector<Point3d> point3D;
	vector<Point2d> point2D;

	std::vector<cv::Point2f> corners1;


	int ChessboardPatternWidth = 9;//Horrizonal Number of internal corners of pattern //change it accrounding to your pattern 
	int ChessboardPatternHight = 6;//vertical Number of internal corners of pattern //change it accrounding to your pattern 
	Size patternSize(ChessboardPatternWidth, ChessboardPatternHight);
	float BoardBoxSize = 2.2;//distance between 2 correns //change it accrounding to your pattern . megger it in cm or mm.
						 //your unit of meggerment will consider as object point units.
	for (int j = 0; j < patternSize.height;j++)
	{
		for (int i = 0; i < patternSize.width;i++)
		{
			objectPoints_cam.push_back(Point3f(i * BoardBoxSize, j * BoardBoxSize, 0));
		}
	}

	//below are the 3d object point(world point) to drow x , y z axis.
	point3D.push_back(Point3d(0, 0, -6.6)); //-z this point represents 10( cm or mm accrounding to BoardBoxSize unit  ) 
	point3D.push_back(Point3d(6.6, 0, 0));  //x
	point3D.push_back(Point3d(0, 6.6, 0));  //y

	////below are the 3d object point(world point) to drow Box.
	point3D.push_back(Point3d(4.4, 0, -4.4));//(x,y,z)
	point3D.push_back(Point3d(0, 4.4, -4.4));
	point3D.push_back(Point3d(4.4, 4.4, -4.4));
	point3D.push_back(Point3d(4.4, 0, -4.4));
	point3D.push_back(Point3d(0, 0, -4.4));
	
	
	for (int i{ 0 }; i < images.size(); i++)
	{
		view = cv::imread(images[i]);

		if (view.empty() != 1)
		{
			found = findChessboardCorners(view, patternSize, corners1);//This will detect pattern
		}

		if (found)

		{
			cv::cvtColor(view, Img, COLOR_BGR2GRAY);

			cornerSubPix(Img, corners1, Size(11, 11), Size(-1, -1),
				TermCriteria(TermCriteria::EPS + TermCriteria::MAX_ITER, 30, 0.001));

			drawChessboardCorners(view, patternSize, corners1, found);

			// The following are two important funtion
			cv::solvePnP(objectPoints_cam, corners1, cameraMatrix, distCoeffs, R, T);//Gives you rotation_vector, translation_vector

			//following funtion gives you point2d from point3D world point to drow them on 2d image.
			cv::projectPoints(point3D, R, T, cameraMatrix, distCoeffs, point2D);


			// following are just drowing funtion to drow object on output image.

			//To draw x,y z axis on image.
			cv::line(view, corners1[0], point2D[0], cv::Scalar(0, 0, 255), 6);//z
			cv::line(view, corners1[0], point2D[1], cv::Scalar(255, 0, 0), 6);//x
			cv::line(view, corners1[0], point2D[2], cv::Scalar(0, 255, 0), 6);//y


			putText(view, x, Point(point2D[1].x - 10, point2D[1].y - 10), FONT_HERSHEY_COMPLEX_SMALL, 1, cv::Scalar(255, 0, 0), 2);
			putText(view, y, Point(point2D[2].x - 10, point2D[2].y - 10), FONT_HERSHEY_COMPLEX_SMALL, 1, cv::Scalar(0, 255, 0), 2);
			putText(view, z, Point(point2D[0].x - 10, point2D[0].y - 10), FONT_HERSHEY_COMPLEX_SMALL, 1, cv::Scalar(0, 0, 255), 2);
			circle(view, point2D[0], 3, cv::Scalar(0, 0, 255), 4, 8, 0);
			circle(view, point2D[1], 3, cv::Scalar(255, 0, 0), 4, 8, 0);
			circle(view, point2D[2], 3, cv::Scalar(0, 255, 0), 4, 8, 0);

			// To drow box on image. It will writen for pattern size 9,6.
		   //If you are using diffrent change corners1 point and point2D point. 

			//creating margin lines of the cube
			cv::line(view, corners1[0], point2D[7], cv::Scalar(255, 255, 0), 3); //top-left
			cv::line(view, corners1[18], point2D[4], cv::Scalar(255, 255, 0), 3); //bottom-left
			cv::line(view, corners1[20], point2D[5], cv::Scalar(255, 255, 0), 3); //bottom-right
			cv::line(view, corners1[2], point2D[6], cv::Scalar(255, 255, 0), 3);  //top-right

			//creating bottom square
			cv::line(view, corners1[0], corners1[2], cv::Scalar(255, 255, 0), 3); // top-green
			cv::line(view, corners1[2], corners1[20], cv::Scalar(255, 255, 0), 3); //right-green
			cv::line(view, corners1[20], corners1[18], cv::Scalar(255, 255, 0), 3); //bottom-green
			cv::line(view, corners1[18], corners1[0], cv::Scalar(255, 255, 0), 3);  //left-green

			cv::line(view, point2D[3], point2D[7], cv::Scalar(255, 255, 0), 3);
			cv::line(view, point2D[4], point2D[5], cv::Scalar(255, 255, 0), 3);
			cv::line(view, point2D[5], point2D[6], cv::Scalar(255, 255, 0), 3);
			cv::line(view, point2D[4], point2D[7], cv::Scalar(255, 255, 0), 3);

		}

		// Display image.
		cv::imshow("Output", view);
		cv::waitKey(0);

	}

	cv::destroyAllWindows();

	//while (1)
	//{
	//	capture >> view;


	//	if (view.empty() != 1)
	//	{
	//		found = findChessboardCorners(view, patternSize, corners1);//This will detect pattern
	//	}

	//	if (found)

	//	{
	//		cv::cvtColor(view, Img, COLOR_BGR2GRAY);

	//		cornerSubPix(Img, corners1, Size(11, 11), Size(-1, -1),
	//			TermCriteria(TermCriteria::EPS + TermCriteria::MAX_ITER, 30, 0.001));

	//		//	drawChessboardCorners(view, patternSize, corners1, found);

	//		// The following are two important funtion
	//		cv::solvePnP(objectPoints_cam, corners1, cameraMatrix, distCoeffs, R, T);//Gives you rotation_vector, translation_vector

	//		//following funtion gives you point2d from point3D world point to drow them on 2d image.
	//		cv::projectPoints(point3D, R, T, cameraMatrix, distCoeffs, point2D);


	//		// following are just drowing funtion to drow object on output image.

	//		//To draw x,y z axis on image.
	//		cv::line(view, corners1[0], point2D[0], cv::Scalar(0, 0, 255), 6);//z
	//		cv::line(view, corners1[0], point2D[1], cv::Scalar(255, 0, 0), 6);//x
	//		cv::line(view, corners1[0], point2D[2], cv::Scalar(0, 255, 0), 6);//y


	//		putText(view, x, Point(point2D[1].x - 10, point2D[1].y - 10), FONT_HERSHEY_COMPLEX_SMALL, 1, cv::Scalar(255, 0, 0), 2);
	//		putText(view, y, Point(point2D[2].x - 10, point2D[2].y - 10), FONT_HERSHEY_COMPLEX_SMALL, 1, cv::Scalar(0, 255, 0), 2);
	//		putText(view, z, Point(point2D[0].x - 10, point2D[0].y - 10), FONT_HERSHEY_COMPLEX_SMALL, 1, cv::Scalar(0, 0, 255), 2);
	//		circle(view, point2D[0], 3, cv::Scalar(0, 0, 255), 4, 8, 0);
	//		circle(view, point2D[1], 3, cv::Scalar(255, 0, 0), 4, 8, 0);
	//		circle(view, point2D[2], 3, cv::Scalar(0, 255, 0), 4, 8, 0);

	//		// To drow box on image. It will writen for pattern size 9,6.
	//	   //If you are using diffrent change corners1 point and point2D point. 

	//		//creating margin lines of the cube
	//		cv::line(view, corners1[0], point2D[7], cv::Scalar(255, 255, 0), 3); //top-left
	//		cv::line(view, corners1[18], point2D[4], cv::Scalar(255, 255, 0), 3); //bottom-left
	//		cv::line(view, corners1[20], point2D[5], cv::Scalar(255, 255, 0), 3); //bottom-right
	//		cv::line(view, corners1[2], point2D[6], cv::Scalar(255, 255, 0), 3);  //top-right

	//		//creating bottom square
	//		cv::line(view, corners1[0], corners1[2], cv::Scalar(255, 255, 0), 3); // top-green
	//		cv::line(view, corners1[2], corners1[20], cv::Scalar(255, 255, 0), 3); //right-green
	//		cv::line(view, corners1[20], corners1[18], cv::Scalar(255, 255, 0), 3); //bottom-green
	//		cv::line(view, corners1[18], corners1[0], cv::Scalar(255, 255, 0), 3);  //left-green

	//		cv::line(view, point2D[3], point2D[7], cv::Scalar(255, 255, 0), 3);
	//		cv::line(view, point2D[4], point2D[5], cv::Scalar(255, 255, 0), 3);
	//		cv::line(view, point2D[5], point2D[6], cv::Scalar(255, 255, 0), 3);
	//		cv::line(view, point2D[4], point2D[7], cv::Scalar(255, 255, 0), 3);

	//	}

	//	// Display image.
	//	cv::imshow("Output", view);
	//	cv::waitKey(1);
	//}



  return 0;
}
