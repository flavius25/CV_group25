
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/highgui.hpp>
#include <stdio.h>
#include <iostream>
#include<typeinfo>
#include<iterator>
#include <algorithm>
#include<vector>

using namespace cv;

// Defining the dimensions of checkerboard
int CHECKERBOARD[2]{6,9}; 

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

//Global parameters used for calibration optimisation
int noImagesUsed;
int minImages = 10;
double epsilon = 0.284; 
bool maxIterationsReached = false;
bool iterationsDone = false;


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

  //name of XML-file to store values
  std::string filename = "Params.xml";

  // Saving the parameters in an XML file
  cv::FileStorage fs (filename, FileStorage:: WRITE); 
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
  std::cout << "Images used for calibration : ";

   for(int i=0; i < a.size(); i++){
      std::cout << a.at(i) << ' ';
   }
   std::cout << std::endl;
   
}

double iterativeCalibration() {

  // Creating vector to store vectors of 3D points for each image
  std::vector<std::vector<cv::Point3f> > objpoints;

  // Creating vector to store vectors of 2D points for each image
  std::vector<std::vector<cv::Point2f> > imgpoints;

  // Defining the world coordinates for 3D points, each point multiplied by size of each checker square in cm 
  std::vector<cv::Point3f> objp;
  for(int i{0}; i<CHECKERBOARD[1]; i++)
  {
    for(int j{0}; j<CHECKERBOARD[0]; j++)
      objp.push_back(cv::Point3f(2.2 * j, 2.2 * i, 0));
  }

  
  // vector to store the pixel coordinates of detected checker board corners
  std::vector<cv::Point2f> corner_pts;

  //boolean and matrix initialised for finding the chessboard corners
  bool success;
  cv::Mat frame, gray;

  // Looping over all the images in the image vector
  for(int i{0}; i<images.size(); i++)
  {

    frame = cv::imread(images[i]);

    int x = frame.size().width;
    int y = frame.size().height;

    cv::cvtColor(frame,gray,cv::COLOR_BGR2GRAY);

    // Finding checker board corners
    // If desired number of corners are found in the image then success = true  
    success = cv::findChessboardCorners(gray, cv::Size(CHECKERBOARD[0], CHECKERBOARD[1]), corner_pts, CALIB_CB_ADAPTIVE_THRESH | CALIB_CB_FAST_CHECK | CALIB_CB_NORMALIZE_IMAGE);
    
    /* 
    * If desired number of corner are detected,
    * we refine the pixel coordinates and display 
    * them on the images of checker board
    */
    if(success)
    {
      cv::TermCriteria criteria(TermCriteria::EPS | TermCriteria::MAX_ITER, 30, 0.001);
      
      // refining pixel coordinates for given 2d points.
      cv::cornerSubPix(gray,corner_pts,cv::Size(11,11), cv::Size(-1,-1),criteria);
      
      if (iterationsDone) {
      // Displaying the detected corner points on the checker board
      cv::drawChessboardCorners(frame, cv::Size(CHECKERBOARD[0], CHECKERBOARD[1]), corner_pts, success);
      cv::imshow("Image", frame);
      cv::waitKey(0);
    }
      
      objpoints.push_back(objp);
      imgpoints.push_back(corner_pts);
    }
  }
  
  cv::destroyAllWindows();

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
    std::vector<float> perViewErrorsVector;
    perViewErrorsVector.assign(perViewErrors.begin<float>(), perViewErrors.end<float>());
    
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