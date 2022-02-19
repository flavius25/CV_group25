
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
#include<typeinfo>

//#define CV_CALIB_CB_ADAPTIVE_THRESH 1
//#define CV_CALIB_CB_FAST_CHECK 8
//#define CV_CALIB_CB_NORMALIZE_IMAGE 2

// Defining the dimensions of checkerboard
int CHECKERBOARD[2]{6,9}; 

using namespace cv;

double iterationBody();

//Global parameters for final calculation
std::vector<std::vector<cv::Point3f>> objectPointsGlobal;
std::vector<std::vector<cv::Point2f>> imagePointsGlobal;
int intRows;
int intCols;
int noImagesUsed = 15;
int minImages = 5;
float epsilon = 2; 
bool maxIterationsReached = false;
bool drawImages = false;

//Declaration of vector with indices of images to ignore
std::vector <int> imagesToIgnore;


int main() {


  double rmsRP_Error = iterationBody();

  while (rmsRP_Error > epsilon || maxIterationsReached){
    rmsRP_Error = iterationBody();
  }
  drawImages = true;
  iterationBody();

  cv::Mat cameraMatrix, distCoeffs, R, T, stdDeviationsIntrinsics, stdDeviationsExtrinsics, perViewErrors;
  rmsRP_Error = cv::calibrateCamera(objectPointsGlobal, imagePointsGlobal, cv::Size(intRows, intCols), cameraMatrix, distCoeffs, R, T, stdDeviationsIntrinsics, stdDeviationsExtrinsics, perViewErrors, CALIB_USE_INTRINSIC_GUESS);
  //, CALIB_USE_INTRINSIC_GUESS|CALIB_FIX_PRINCIPAL_POINT
   

  std::string filename = "Params.xml";

  // Saving the parameters in an XML file
  cv::FileStorage fs (filename, FileStorage:: WRITE); 
  fs << "cameraMatrix" << cameraMatrix;
  fs << "distCoeffs" << distCoeffs;
  fs << "Rotation_vector" << R;
  fs << "Translation_vector" << T;
  fs << "PerViewErrors" << perViewErrors;
  fs << "overallRMS_RPError" << rmsRP_Error; 
  fs.release();

  std::cout << "cameraMatrix : " << cameraMatrix << std::endl;
  std::cout << "distCoeffs : " << distCoeffs << std::endl;
  std::cout << "Rotation vector : " << R << std::endl;
  std::cout << "Translation vector : " << T << std::endl;
  std::cout << "Overall RMS projection errors : " << rmsRP_Error << std::endl;
  std::cout << "Per view errors : " << perViewErrors << std::endl; 

  return 0;

}

void print(std::vector <int> const &a) {
   std::cout << "The vector elements are : ";

   for(int i=0; i < a.size(); i++){
      std::cout << a.at(i) << ' ';
   }
   std::cout << std::endl;
   
}

double iterationBody() {

  // Creating vector to store vectors of 3D points for each checkerboard image
  std::vector<std::vector<cv::Point3f> > objpoints;

  // Creating vector to store vectors of 2D points for each checkerboard image
  std::vector<std::vector<cv::Point2f> > imgpoints;

  // Defining the world coordinates for 3D points
  std::vector<cv::Point3f> objp;
  for(int i{0}; i<CHECKERBOARD[1]; i++)
  {
    for(int j{0}; j<CHECKERBOARD[0]; j++)
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

  for(int i{0}; i<images.size(); i++)
  {
    //check if image should be ignore, if so, continue
    if (std::find(imagesToIgnore.begin(), imagesToIgnore.end(), i) != imagesToIgnore.end()){
      continue;
    }
    else {
      frame = cv::imread(images[i]);

      int x = frame.size().width;
      int y = frame.size().height;

      //int down_width = 1280;
      //int down_height = 720;
      //Mat resized_up;

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
        
        // Displaying the detected corner points on the checker board
        cv::drawChessboardCorners(frame, cv::Size(CHECKERBOARD[0], CHECKERBOARD[1]), corner_pts, success);
        
        objpoints.push_back(objp);
        imgpoints.push_back(corner_pts);
      }

      //int down_width = 1280;
      //int down_height = 720;
      //Mat resized_up;

      //resize(frame, resized_up, Size(down_width, down_height), INperViewErrorsTER_LINEAR);

      if (drawImages) {
      cv::imshow("Image", frame);
      cv::waitKey(0);
      }

      
    }
    
  }
  
  //Debug purposes 
  std::cout << "No.images used : " << imgpoints.size() << std::endl;
  cv::destroyAllWindows();

  cv::Mat cameraMatrix;
  cameraMatrix = cv::initCameraMatrix2D(objpoints,imgpoints,cv::Size(gray.rows,gray.cols));
  

  cv::Mat distCoeffs, R, T, stdDeviationsIntrinsics, stdDeviationsExtrinsics, perViewErrors;

  /*
  * Performing camera calibration by 
  * passing the value of known 3D points (objpoints)
  * and corresponding pixel coordinates of the 
  * detected corners (imgpoints)
  */

  double rmsRP_Error;
  rmsRP_Error = cv::calibrateCamera(objpoints, imgpoints, cv::Size(gray.rows,gray.cols), cameraMatrix, distCoeffs, R, T, stdDeviationsIntrinsics, stdDeviationsExtrinsics, perViewErrors, CALIB_USE_INTRINSIC_GUESS);
  //CALIB_USE_INTRINSIC_GUESS

  //Transforming perViewErrors to vector, in order to get indices in next step
  cv::Mat flat = perViewErrors.reshape(1, perViewErrors.total()*perViewErrors.channels());
  std::vector<uchar> perViewErrorsVector = perViewErrors.isContinuous()? flat : flat.clone();

  //Getting the index of the element with the most error, adding it to vector of image-indices to ignore
  int maxElementIndex = std::max_element(perViewErrorsVector.begin(), perViewErrorsVector.end()) - perViewErrorsVector.begin();
  imagesToIgnore.push_back(maxElementIndex);

  if(noImagesUsed - imagesToIgnore.size() == minImages) {
    maxIterationsReached = true;
  }
  std::cout << maxIterationsReached << std::endl;

  //Assigning values to global variables so that cameraCalibration can be called outside scope of function
  objectPointsGlobal = objpoints;
  imagePointsGlobal = imgpoints; 
  intRows = gray.rows;
  intCols = gray.cols;
  
  std::cout << rmsRP_Error << std::endl;
  std::cout << imagesToIgnore.size() << std::endl;
  print(imagesToIgnore);
  return rmsRP_Error;
}