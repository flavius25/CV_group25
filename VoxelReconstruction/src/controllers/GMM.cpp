#include "GMM.h"

#include <opencv2/core/mat.hpp>
#include <opencv2/core/operations.hpp>
#include <opencv2/core/types_c.h>
#include <opencv2/opencv.hpp>
#include <math.h>
#include <cassert>
#include <iostream>

#include "../utilities/General.h"
#include "Reconstructor.h"
#include "Scene3DRenderer.h"
#include "Reconstructor.h"



using namespace  std;
using namespace cv;
using namespace cv::ml;

//Idea for a function that prepares the input for the GMM model. In the offline phase I need an input Matrix of all pixel-rgb values corresponding to the torso of each person.
//for the online phase each persons torso should be fed to the predict separately, see code below where there is a vector of matrices that are separately fed to the GMM-predict function
//and then assigned labels
void findPixelsFromLabels(Mat centers, std::vector <int> labels){
    double lower_y_threshold = 
    double upper_y_threshold = 
    
 
}


void gaussianMixtureModel(std::vector<int> labels){
    //This code taken from: https://programming.vip/docs/gmm-gaussian-mixture-model-method-for-opencv-image-segmentation.html


    Mat img; 

    int width = img.cols;
    int height = img.rows;
    int dims = img.channels();

    int no_samples = width*height;
    Mat points(no_samples, dims, CV_64FC1);
    Mat labels;
    Mat result = Mat::zeros(img.size(), CV_8UC3);

    //Define classification, that is, how many classification points of function K value
    int no_clusters = 4;

    // Find RGB pixel values from image coordinates and assign to points
   int index = 0;
   for (int row = 0; row < height; row++) {
       for (int col = 0; col < width; col++) {
           index = row*width + col;
           Vec3b rgb = img.at<Vec3b>(row, col);
           points.at<double>(index, 0) = static_cast<int>(rgb[0]);
           points.at<double>(index, 1) = static_cast<int>(rgb[1]);
           points.at<double>(index, 2) = static_cast<int>(rgb[2]);
       }
   }

   //Create model  
   Ptr<EM> GMM_model = EM::create();
   //Initialise number of clusters to look for 
   GMM_model->setClustersNumber(no_clusters);
   //Set covariance matrix type
   GMM_model->setCovarianceMatrixType(EM::COV_MAT_SPHERICAL);
   //Set convergence conditions
   GMM_model->setTermCriteria(TermCriteria(TermCriteria::EPS + TermCriteria::COUNT, 100, 0.1));
   //Store the probability partition to labs EM according to the sample training
   GMM_model->trainEM(points, noArray(), labels, noArray());

    //Save model in xml-file
    GMM_model->save("GMM_model.xml"); 

//This will be implemented to see that the GMM model works
//    // Mark color and display for each pixel
//    Mat sample(1, dims, CV_64FC1);//
//    int r = 0, g = 0, b = 0;
//    //Put each pixel in the sample
//    for (int row = 0; row < height; row++) {
//        for (int col = 0; col < width; col++) {
//            index = row*width + col;

//            //Get the color of each channel
//            b = src.at<Vec3b>(row, col)[0];
//            g = src.at<Vec3b>(row, col)[1];
//            r = src.at<Vec3b>(row, col)[2];
           
//            //Put pixels in sample data
//            sample.at<double>(0, 0) = static_cast<double>(b);
//            sample.at<double>(0, 1) = static_cast<double>(g);
//            sample.at<double>(0, 2) = static_cast<double>(r);
           
//            //Rounding
//            int prediction = cvRound(GMM_model->predict2(sample, noArray())[1]);
//            Scalar c = color_tab[prediction];
//            result.at<Vec3b>(row, col)[0] = c[0];
//            result.at<Vec3b>(row, col)[1] = c[1];
//            result.at<Vec3b>(row, col)[2] = c[2];
//        }
//    }
//    imshow("EM-Segmentation", result);

//    waitKey(0);
//    destroyAllWindows();
}



    //Below here is essentially the online phase, won't be implemented here

    Ptr<EM> GMM_model = EM::load("GMM_model.xml"); 

    //Define 4 colors, these are already in my Reconstructor.h
    // Scalar color_tab[]={
    //     Scalar(0,0,255),
    //     Scalar(0,255,0),
    //     Scalar(255,0,0),
    //     Scalar(255,0,255)
    // };

    //vector of cluster matrices
    std::vector <Mat> cluster_matrices; 

    std::vector <int> predictions; 

    //for loop where prediction of color_label happens, important that all voxels have assigned label for which cluster they belong to (0,1,2,3) and that the cluster matrices are order the same
    for (int cl = 0; cl < cluster_matrices.size(); cl++){

        int prediction = cvRound(GMM_model->predict2(cl, noArray())[1]);
        predictions.push_back(prediction); 
    }

    for (int i = 0; i < (int)m_visible_voxels.size(); i++){
        int lb = m_visible_voxels[i]->label;
        int color_index = predictions[lb]; 
        m_visible_voxels[i]->color = color_tab[color_index];
 

}


 //Alternative loading and saving: 
    // FileStorage fsLoad("GMM_model.xml", FileStorage::READ);
    // GMM_model.read(fsLoad.root());
    // fsLoad.release();
    //  FileStorage fsOut("GMM_model.xml", FileStorage::WRITE);
    // GMM_model.write(fsOut);
    // fsOut.release();












}