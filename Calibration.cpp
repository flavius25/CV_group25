
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
#include<iterator>
#include <algorithm>
#include<vector>

//#define CV_CALIB_CB_ADAPTIVE_THRESH 1
//#define CV_CALIB_CB_FAST_CHECK 8
//#define CV_CALIB_CB_NORMALIZE_IMAGE 2

// Defining the dimensions of checkerboard
int CHECKERBOARD[2]{6,9}; 

using namespace cv;


//Declaring function iterationBody()
double iterationBody();

//Declaration of vector with images
std::vector<cv::String> images;

//Global parameters for final calculation
std::vector<std::vector<cv::Point3f>> objectPointsGlobal;
std::vector<std::vector<cv::Point2f>> imagePointsGlobal;
int intRows;
int intCols;
int noImagesUsed;
cv::Mat finalCameraMatrix; 
int minImages = 10;
double epsilon = 0.284; 
bool maxIterationsReached = false;
bool iterationsDone = false;

int main() {
  
  // Path of the folder containing checkerboard images
  std::string path = "./images/*.jpg";

  cv::glob(path, images);
  noImagesUsed = images.size();

  double rmsRP_Error = iterationBody();

  while (rmsRP_Error >= epsilon && !maxIterationsReached){
    rmsRP_Error = iterationBody();
  }
  iterationsDone = true;
  iterationBody();

  cv::Mat cameraMatrix, distCoeffs, R, T, stdDeviationsIntrinsics, stdDeviationsExtrinsics, perViewErrors;
  rmsRP_Error = cv::calibrateCamera(objectPointsGlobal, imagePointsGlobal, cv::Size(intRows, intCols), finalCameraMatrix, distCoeffs, R, T, stdDeviationsIntrinsics, stdDeviationsExtrinsics, perViewErrors, CALIB_USE_INTRINSIC_GUESS);

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

  std::cout << "cameraMatrix : "<< cameraMatrix << std::endl;
  std::cout << "distCoeffs : " << distCoeffs << std::endl;
  std::cout << "Rotation vector : " << R << std::endl;
  std::cout << "Translation vector : " << T << std::endl;
  std::cout << "Overall RMS projection errors : " << rmsRP_Error << std::endl;

  return 0;

}

//Debugging purposes functions to print elements in vector 
void print(std::vector <int> const &a) {
   std::cout << "The vector elements are : ";

   for(int i=0; i < a.size(); i++){
      std::cout << a.at(i) << ' ';
   }
   std::cout << std::endl;
   
}

void print(std::vector <float> const &a) {
   std::cout << "The vector elements are : ";

   for(int i=0; i < a.size(); i++){
      std::cout << a.at(i) << ' ';
   }
   std::cout << std::endl;
   
}

void print(std::vector <cv::String> const &a) {
   std::cout << "The vector elements are : ";

   for(int i=0; i < a.size(); i++){
      std::cout << a.at(i) << ' ';
   }
   std::cout << std::endl;
   
}

/*Function to add zeros to index position of images to be ignored
*This makes sure that the perViewErrorsVector is always equal to the amount of images used
*which will make it possible to ignore the images that gives the highest perViewError even 
*when the vector of perViewErrors is getting smaller
print(images);
*/
//  std::vector<float> insertZeros(std::vector<float> vec){
  
//   std::vector<float> copyVector = vec;
//   for(int i = 0; i<imagesToIgnore.size(); i++){
    
//       auto itPos = copyVector.begin() + (imagesToIgnore[i]+1);
//       float zero = 0.0;
//       copyVector.insert(itPos, zero);
//   }
//   return copyVector;
// }

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

  cv::Mat frame, gray;
  // vector to store the pixel coordinates of detected checker board corners 
  std::vector<cv::Point2f> corner_pts;
  bool success;

  // Looping over all the images in the image list
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
  
  //Debug purposes 
  std::cout << "No.images used : " << imgpoints.size() << std::endl;
  cv::destroyAllWindows();

  cv::Mat cameraMatrix;
  if (!iterationsDone){
  //Finding initial values for cameraMatrix to pass to calibrateCamera function
  
  cameraMatrix = cv::initCameraMatrix2D(objpoints,imgpoints,cv::Size(gray.rows,gray.cols));
  }
  else {
    cameraMatrix = finalCameraMatrix;
  }
  cv::Mat distCoeffs, R, T, stdDeviationsIntrinsics, stdDeviationsExtrinsics, perViewErrors;

  /*
  * Performing camera calibration by 
  * passing the value of known 3D points (objpoints)
  * and corresponding pixel coordinates of the 
  * detected corners (imgpoints)
  */

  double rmsRP_Error;
  rmsRP_Error = cv::calibrateCamera(objpoints, imgpoints, cv::Size(gray.rows,gray.cols), cameraMatrix, distCoeffs, R, T, stdDeviationsIntrinsics, stdDeviationsExtrinsics, perViewErrors, CALIB_USE_INTRINSIC_GUESS);
  
  std::cout << "rmsPError 1 : " << rmsRP_Error << std::endl;
 
  //if maximum iterations are not reached (number of images to consider is still above the minimum), consider which image has the highest perViewError and remove it from images to consider
  if (rmsRP_Error > epsilon){
    if (!maxIterationsReached){
    //converting to vector of floats
    std::vector<float> perViewErrorsVector;
    perViewErrorsVector.assign(perViewErrors.begin<float>(), perViewErrors.end<float>());
    
    //Debugging purposes
    print(perViewErrorsVector);

    //Getting the index of the element with the most error, adding it to vector of image-indices to ignore
    int maxElementIndex = std::max_element(perViewErrorsVector.begin(), perViewErrorsVector.end()) - perViewErrorsVector.begin();
    // Deletes the second element (vec[1])
    std::cout << "Erasing image at position " << maxElementIndex << std::endl;
    images.erase(images.begin() + maxElementIndex);
    std::cout << "Images size after removing 1 image: " << images.size() << std::endl;
    noImagesUsed = images.size();
    }
  }
  else{
    finalCameraMatrix = cameraMatrix;
  }


  if(noImagesUsed == minImages) {
    maxIterationsReached = true;
  }
  std::cout << maxIterationsReached << std::endl;

  if (iterationsDone){
  //Assigning values to global variables so that cameraCalibration can be called outside scope of function
  objectPointsGlobal = objpoints;
  imagePointsGlobal = imgpoints; 
  intRows = gray.rows;
  intCols = gray.cols;
  finalCameraMatrix = cameraMatrix;
  }
  
  std::cout << "rmsPError 2  : " << rmsRP_Error << std::endl;
  return rmsRP_Error;
}