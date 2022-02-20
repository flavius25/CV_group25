
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/highgui.hpp>
#include <stdio.h>

using namespace cv;
using namespace std;

// Defining the dimensions of checkerboard
int CHECKERBOARD[2]{6,9};
float BoardBoxSize = 2.2; //in cm

//Declaring function iterativeCalibration()
double iterativeCalibration();

//Declaration of vector with images
std::vector<cv::String> images;

//Global parameters for final calculation
std::vector<std::vector<cv::Point3f>> objectPointsGlobal;
std::vector<std::vector<cv::Point2f>> imagePointsGlobal;
cv::Mat finalCameraMatrix;
int intRows;
int intCols;
int width_image;
int height_image;

//Global parameters for online stage
cv::Mat frame, gray; //use for offline stage
bool success;
const string x = "X"; //label x for line 
const string y = "Y"; //label y for line
const string z = "Z"; //label z for line
std::vector<cv::Point2f> corner_pts; // vector to store the pixel coordinates of detected checker board corners

//Global parameters used for calibration optimisation
int noImagesUsed;
int minImages = 10;
double epsilon = 0.21; 
bool maxIterationsReached = false;
bool iterationsDone = false;

//For choosing functionality
enum class func
{
    online_images,
    camera_performance
};

func choose_function = func::online_images;    //switch to online_images or camera_performance here 
bool edge_enhancing = false;                   //swith to true to activate edge enhancing.

int main() {
  
  // Path of the folder containing checkerboard images
  std::string path = "./images/*.jpg";

  cv::glob(path, images);
  noImagesUsed = images.size();

  //getting the first overall rms reprojection error
  double rmsRP_Error = iterativeCalibration();

  /*while the error is higher than a certain threshold value or maximum iterations as 
  *specified by the minimum amount of images to consider has not been reached
  *continue looping through and removing images with highest perViewError */
  while (rmsRP_Error >= epsilon && !maxIterationsReached){
    rmsRP_Error = iterativeCalibration();
  }

  //set iterationsDone to true, go into iterativeCalibration again to print the images and get the final matrix
  iterationsDone = true;
  iterativeCalibration();

  //initialise variables here and call function calibrateCamera again with the matrix and images obtained by optimisation
  //done in order to be able to save the values in XML file
  cv::Mat distCoeffs, R, T, stdDeviationsIntrinsics, stdDeviationsExtrinsics, perViewErrors;
  rmsRP_Error = cv::calibrateCamera(objectPointsGlobal, imagePointsGlobal, cv::Size(intRows, intCols), finalCameraMatrix, distCoeffs, R, T, stdDeviationsIntrinsics, stdDeviationsExtrinsics, perViewErrors, CALIB_USE_INTRINSIC_GUESS);
  
  if (choose_function == func::online_images) {
      
      cv::Mat view, Img;
      Size patternSize(CHECKERBOARD[1], CHECKERBOARD[0]);
	  vector<Point3f>  objectPoints_imag;
      vector<Point3d> point3D;
      vector<Point2d> point2D;
      bool found = false;
      std::vector<cv::Point2f> corners1; //used for storing projectedPoints


	  for (int j = 0; j < patternSize.height;j++)
	  {
		  for (int i = 0; i < patternSize.width;i++)
		  {
			  objectPoints_imag.push_back(Point3f(i * BoardBoxSize, j * BoardBoxSize, 0));
		  }
	  }

	  //below are the 3d object point(world point) to drow x , y z axis.
	  point3D.push_back(Point3d(0, 0, -6.6)); //-z this point represents -6.6 cm 
	  point3D.push_back(Point3d(6.6, 0, 0));  //x
	  point3D.push_back(Point3d(0, 6.6, 0));  //y

	  //below are the 3d object point(world point) to drow Box.
	  point3D.push_back(Point3d(2 * BoardBoxSize, 0, -2 * BoardBoxSize));//(x,y,z)
	  point3D.push_back(Point3d(0, 2 * BoardBoxSize, -2 * BoardBoxSize));
	  point3D.push_back(Point3d(2 * BoardBoxSize, 2 * BoardBoxSize, -2 * BoardBoxSize));
	  point3D.push_back(Point3d(2 * BoardBoxSize, 0, -2 * BoardBoxSize));
	  point3D.push_back(Point3d(0, 0, -2 * BoardBoxSize));


	  for (int i{ 0 }; i < images.size(); i++)
	  {
		  view = cv::imread(images[i]);

		  if (view.empty() != 1)
		  {
			  found = findChessboardCorners(view, patternSize, corners1); //This will detect pattern
		  }

		  if (found)

		  {
			  cv::cvtColor(view, Img, COLOR_BGR2GRAY);

			  cornerSubPix(Img, corners1, Size(11, 11), Size(-1, -1),
				  TermCriteria(TermCriteria::EPS + TermCriteria::MAX_ITER, 30, 0.001));

			  drawChessboardCorners(view, patternSize, corners1, found);

			  // The following are two important funtion
			  cv::solvePnP(objectPoints_imag, corners1, finalCameraMatrix, distCoeffs, R, T);//Gives you rotation_vector, translation_vector

			  //following funtion gives you point2d from point3D world point to drow them on 2d image.
			  cv::projectPoints(point3D, R, T, finalCameraMatrix, distCoeffs, point2D);


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
  }

  else if(choose_function == func::camera_performance){
  
    VideoCapture capture(0); //for when camera is on
    cv::Mat view, Img; 
    Size patternSize(CHECKERBOARD[1], CHECKERBOARD[0]);
    vector<Point3f>  objectPoints_cam;
    vector<Point3d> point3D;
    vector<Point2d> point2D;
    std::vector<cv::Point2f> corners1; //used for storing projectedPoints for online stage
    bool found = false;

    for (int j = 0; j < patternSize.height;j++)
    {
        for (int i = 0; i < patternSize.width;i++)
        {
            objectPoints_cam.push_back(Point3f(i * BoardBoxSize, j * BoardBoxSize, 0));
        }
    }

    //below are the 3d object point(world point) to drow x , y z axis.
    point3D.push_back(Point3d(0, 0, -6.6)); //-z this point represents -6.6 cm 
    point3D.push_back(Point3d(6.6, 0, 0));  //x
    point3D.push_back(Point3d(0, 6.6, 0));  //y

    //below are the 3d object point(world point) to drow Box.
    point3D.push_back(Point3d(2 * BoardBoxSize, 0, -2 * BoardBoxSize));//(x,y,z)
    point3D.push_back(Point3d(0, 2 * BoardBoxSize, -2 * BoardBoxSize));
    point3D.push_back(Point3d(2 * BoardBoxSize, 2 * BoardBoxSize, -2 * BoardBoxSize));
    point3D.push_back(Point3d(2 * BoardBoxSize, 0, -2 * BoardBoxSize));
    point3D.push_back(Point3d(0, 0, -2 * BoardBoxSize));

    while (1)
    {

  	  capture >> view;

  	  if (view.empty() != 1)
  	  {
  		found = findChessboardCorners(view, patternSize, corners1);//This will detect pattern
  	  }

  	  if (found)

  	  {
  		cv::cvtColor(view, Img, COLOR_BGR2GRAY);

  		cornerSubPix(Img, corners1, Size(11, 11), Size(-1, -1),
  			TermCriteria(TermCriteria::EPS + TermCriteria::MAX_ITER, 30, 0.001));

  		//drawChessboardCorners(view, patternSize, corners1, found); //for drawing

  		// The following are two important funtion
  		cv::solvePnP(objectPoints_cam, corners1, finalCameraMatrix, distCoeffs, R, T);//Gives you rotation_vector, translation_vector

  		//following funtion gives you point2d from point3D world point to drow them on 2d image.
  		cv::projectPoints(point3D, R, T, finalCameraMatrix, distCoeffs, point2D);

  		// following are just drawing funtions -> to draw object on output image.
  		//To draw x,y,z axis on image.
  		cv::line(view, corners1[0], point2D[0], cv::Scalar(0, 0, 255), 6);//z
  		cv::line(view, corners1[0], point2D[1], cv::Scalar(255, 0, 0), 6);//x
  		cv::line(view, corners1[0], point2D[2], cv::Scalar(0, 255, 0), 6);//y

  		putText(view, x, Point(point2D[1].x - 10, point2D[1].y - 10), FONT_HERSHEY_COMPLEX_SMALL, 1, cv::Scalar(255, 0, 0), 2);
  		putText(view, y, Point(point2D[2].x - 10, point2D[2].y - 10), FONT_HERSHEY_COMPLEX_SMALL, 1, cv::Scalar(0, 255, 0), 2);
  		putText(view, z, Point(point2D[0].x - 10, point2D[0].y - 10), FONT_HERSHEY_COMPLEX_SMALL, 1, cv::Scalar(0, 0, 255), 2);
  		circle(view, point2D[0], 3, cv::Scalar(0, 0, 255), 4, 8, 0);
  		circle(view, point2D[1], 3, cv::Scalar(255, 0, 0), 4, 8, 0);
  		circle(view, point2D[2], 3, cv::Scalar(0, 255, 0), 4, 8, 0);

  		// To draw box on image. It will writen for pattern size 9,6
  		//creating margin lines of the cube
  		cv::line(view, corners1[0], point2D[7], cv::Scalar(255, 255, 0), 3); //top-left
  		cv::line(view, corners1[18], point2D[4], cv::Scalar(255, 255, 0), 3); //bottom-left
  		cv::line(view, corners1[20], point2D[5], cv::Scalar(255, 255, 0), 3); //bottom-right
  		cv::line(view, corners1[2], point2D[6], cv::Scalar(255, 255, 0), 3);  //top-right

  		//creating bottom square
  		cv::line(view, corners1[0], corners1[2], cv::Scalar(255, 255, 0), 3); 
  		cv::line(view, corners1[2], corners1[20], cv::Scalar(255, 255, 0), 3); 
  		cv::line(view, corners1[20], corners1[18], cv::Scalar(255, 255, 0), 3); 
  		cv::line(view, corners1[18], corners1[0], cv::Scalar(255, 255, 0), 3);  

		//creating top square
  		cv::line(view, point2D[3], point2D[7], cv::Scalar(255, 255, 0), 3);
  		cv::line(view, point2D[4], point2D[5], cv::Scalar(255, 255, 0), 3);
  		cv::line(view, point2D[5], point2D[6], cv::Scalar(255, 255, 0), 3);
  		cv::line(view, point2D[4], point2D[7], cv::Scalar(255, 255, 0), 3);

  	}

  	// Display image.
  	cv::imshow("CameraPerformance", view);
  	cv::waitKey(1);
  }
  cv::destroyAllWindows();

}

  //name of XML-file to store values
  std::string filename = "Params.xml";

  // Saving the parameters in an XML file
  cv::FileStorage fs (filename, FileStorage:: WRITE);
  fs << "width" << width_image;
  fs << "height" << height_image;
  fs << "cameraMatrix" << finalCameraMatrix;
  fs << "distCoeffs" << distCoeffs;
  fs << "Rotation_vector" << R;
  fs << "Translation_vector" << T;
  fs << "PerViewErrors" << perViewErrors;
  fs << "overallRMS_RPError" << rmsRP_Error; 
  fs.release();

  return 0;

}

//function to print elements of image vector
void print(std::vector <cv::String> const &a) {
  std::cout << "Images used for calibration : " << '\n';

   for(int i=0; i < a.size(); i++){
      std::cout << a.at(i) << ' ' << '\n';
   }
   std::cout << std::endl;
   
}

double iterativeCalibration() {

  std::vector<std::vector<cv::Point3f> > objpoints; // Creating vector to store vectors of 3D points for each image
  std::vector<std::vector<cv::Point2f> > imgpoints; // Creating vector to store vectors of 2D points for each image

  // Defining the world coordinates for 3D points, each point multiplied by size of each checker square in cm 
  std::vector<cv::Point3f> objp;
  for(int i{0}; i<CHECKERBOARD[1]; i++)
  {
    for(int j{0}; j<CHECKERBOARD[0]; j++)
      objp.push_back(cv::Point3f(BoardBoxSize * j, BoardBoxSize * i, 0));
  }

  // Looping over all the images in the image vector
  for (int i{ 0 }; i < images.size(); i++)
  {

      frame = cv::imread(images[i]);

      width_image = frame.size().width;  
      height_image = frame.size().height;

      if (edge_enhancing) {
          //edge enhacement ->applying a median filter to preserve edges
          //local variables needed for filtering
          Mat img_blur, gray_enhanced;
          std::vector<cv::Point2f> corner_pts_edge_enh;

          //apply median filter
          medianBlur(frame, img_blur, 5);

          cv::cvtColor(img_blur, gray_enhanced, cv::COLOR_BGR2GRAY);
          cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);

          success = cv::findChessboardCorners(gray_enhanced, cv::Size(CHECKERBOARD[0], CHECKERBOARD[1]), corner_pts_edge_enh, CALIB_CB_ADAPTIVE_THRESH | CALIB_CB_FAST_CHECK | CALIB_CB_NORMALIZE_IMAGE);
          cv::findChessboardCorners(gray, cv::Size(CHECKERBOARD[0], CHECKERBOARD[1]), corner_pts, CALIB_CB_ADAPTIVE_THRESH | CALIB_CB_FAST_CHECK | CALIB_CB_NORMALIZE_IMAGE);

          cv::Mat copyf = frame.clone();

          for (size_t i = 0; i < corner_pts.size(); i++)
          {
              //std::cout << " -- No filtering applied [" << i << "]  (" << corner_pts[i].x << "," << corner_pts[i].y << ")\n";
              cv::circle(copyf, corner_pts[i], 4, cv::Scalar(0, 0, 255), cv::FILLED);
          }

          for (size_t i = 0; i < corner_pts_edge_enh.size(); i++)
          {
              //std::cout << " -- Filtering applied [" << i << "]  (" << corner_pts_edge_enh[i].x << "," << corner_pts_edge_enh[i].y << ")\n";
              cv::circle(copyf, corner_pts_edge_enh[i], 4, cv::Scalar(0, 255, 0), cv::FILLED);
          }

          if (success)
          {
              cv::TermCriteria criteria(TermCriteria::EPS | TermCriteria::MAX_ITER, 30, 0.001);

              // refining pixel coordinates for given 2d points.
              cv::cornerSubPix(gray, corner_pts_edge_enh, cv::Size(11, 11), cv::Size(-1, -1), criteria);

              if (iterationsDone) {
                  // Displaying the detected corner points on the checker board
                  cv::drawChessboardCorners(frame, cv::Size(CHECKERBOARD[0], CHECKERBOARD[1]), corner_pts_edge_enh, success);
              }

              objpoints.push_back(objp);
              imgpoints.push_back(corner_pts_edge_enh);
          }

          //cv::imshow("Image", copyf);  //uncomment this if you want to plot the circles
          //cv::waitKey(1);

          cv::destroyAllWindows();
      }
      else {

          cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY); //converting to grayscale

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

              if (iterationsDone) {
                  // Displaying the detected corner points on the checker board
                  cv::drawChessboardCorners(frame, cv::Size(CHECKERBOARD[0], CHECKERBOARD[1]), corner_pts, success);
                  //cv::imshow("Image", frame);
                  //cv::waitKey(0);
              }

              objpoints.push_back(objp);
              imgpoints.push_back(corner_pts);
          }

          cv::destroyAllWindows();
      }
  }

  //intialise camera matrix, if iterations are not done, initialise matrix with help of initCameraMatrix2D function
  //else use the precalculated matrix from previous iterations
  cv::Mat cameraMatrix;
  if (!iterationsDone){
  cameraMatrix = cv::initCameraMatrix2D(objpoints,imgpoints,cv::Size(gray.rows,gray.cols));
  }
  else {
    cameraMatrix = finalCameraMatrix;
  }

  //initialise matrices for storing values obtained from calibrateCamera function
  cv::Mat distCoeffs, R, T, stdDeviationsIntrinsics, stdDeviationsExtrinsics, perViewErrors;

  /*
  * Performing camera calibration by 
  * passing the value of known 3D points (objpoints)
  * and corresponding pixel coordinates of the 
  * detected corners (imgpoints).
  * Use flag CALIB_USE_INTRINSIC_GUESS in order to estimate image center not from resolution
  */

  double rmsRP_Error;
  rmsRP_Error = cv::calibrateCamera(objpoints, imgpoints, cv::Size(gray.rows,gray.cols), cameraMatrix, distCoeffs, R, T, stdDeviationsIntrinsics, stdDeviationsExtrinsics, perViewErrors, CALIB_USE_INTRINSIC_GUESS);
  
 
  //if overall reprojection error is still above certain threshold and minimum images to consider has not yet been reached:
  //consider which image has the highest perViewError and remove it from images list
  if (rmsRP_Error > epsilon){
    if (!maxIterationsReached){

    //converting perViewError to vector of floats in order to obtain indices easier
    std::vector<float> perViewErrorsVector = perViewErrors;
    
    //Getting the index of the element with the most error, removing the image of index from image list
    int maxElementIndex = std::max_element(perViewErrorsVector.begin(), perViewErrorsVector.end()) - perViewErrorsVector.begin();
    images.erase(images.begin() + maxElementIndex);
    noImagesUsed = images.size();

    }
  }
  //assign intrinsic camera matrix K to global variable 
  else{

    finalCameraMatrix = cameraMatrix;
  }

  //check if number of images in image list is equal to minimum number to consider, if so maximum iterations have been reached
  if(noImagesUsed == minImages) {
    maxIterationsReached = true;
    iterationsDone = true;
  }

  if (iterationsDone){
  //Assigning values to global variables so that cameraCalibration can be called outside scope of function
  objectPointsGlobal = objpoints;
  imagePointsGlobal = imgpoints; 
  intRows = gray.rows;
  intCols = gray.cols;
  finalCameraMatrix = cameraMatrix;

  //print which images that are used for the final calibration 
  print(images);
  }
  
  //print the rms reprojection error for each iteration
  std::cout << "Overall RMS Reprojection Error : " << rmsRP_Error << std::endl;

  return rmsRP_Error;
}
