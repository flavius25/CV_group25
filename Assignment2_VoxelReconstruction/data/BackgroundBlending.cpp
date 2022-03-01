
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include <iostream>

using namespace cv;

// File for blending 10 separate background images taken from background.avi of each camera


int main( void )
{

  // For loop over all camera directories
  for (int i{1}; i<5; i++){

    // Path of the folder containing the background images
    String number = std::to_string(i);
    std::string path = "./cam" + number + "/background_images/*.png";
    
    //Declaration of vector with images
    std::vector<cv::String> images;

    //initialising image vector
    cv::glob(path, images);

    //empty vector to store the cv::Mats of the images
    std::vector<cv::Mat> image_matrices;

    //for loop over all images to make them into cv::Mats, store in image_matrices vector
    for (int i{0}; i < images.size(); i++){
      cv::Mat this_image = imread(images[i]);
      image_matrices.push_back(this_image);
    }

    //Matrix for the last image
    cv::Mat dst;
    //Set first  image in image_matrices to be the first average image to then be updated in the for loop
    cv::Mat mean_img = image_matrices[0];

    //for loop over all images in image_matrices updating the mean_img iteratively
    for (int i{0}; i < image_matrices.size(); i++){
      if (i == 0){
        continue;
      }
      else {
        //alpha and beta values
        double alpha = 0.5;
        double beta = (1.0 - alpha);

        //the actual blending
        addWeighted(image_matrices[i], alpha, mean_img, beta, 0.0, dst);
      }
    }

    //save file as background.png in each camera folder
    imwrite("./cam" + number + "/background.png", mean_img);
    imshow("Background camera " + number, dst);
    waitKey(0);
  }
    return 0;

}

