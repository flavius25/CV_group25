// #include "Clustering.h"

// #include <opencv2/core/mat.hpp>
// #include <opencv2/core/operations.hpp>
// #include <opencv2/core/types_c.h>
// #include <opencv2/opencv.hpp>
// #include <math.h>
// #include <cassert>
// #include <iostream>

// #include "../utilities/General.h"
// #include "Reconstructor.h"
// #include "Scene3DRenderer.h"
// #include "Reconstructor.h"



// using namespace  std;
// using namespace cv;
// using namespace cv::ml;


// using namespace cv;
// using namespace std;

// std::vector<int> clustering(){ 



//     vector<Reconstructor::Voxel*> voxels = getVisibleVoxels();

//     //find a frame
//     //frame 0-50 camera2
//     //frame 514 camera4
//     //frame 1244/2235 camera3
//     Scene3DRenderer& scene3d = m_Glut->getScene3d();



//     scene3d.setCurrentFrame(514);
//     for (size_t c = 0; c < scene3d.getCameras().size(); ++c)
//     if (c == 3) {
//     scene3d.getCameras()[c]->setVideoFrame(scene3d.getCurrentFrame());
//     }
//     else {
//     cout << "nope";
//     }



//     scene3d.processFrame();
//     scene3d.getReconstructor().update();



//     vector<Reconstructor::Voxel*> voxels = m_Glut->getScene3d().getReconstructor().getVisibleVoxels();



//     std::vector<cv::Point2f> points(voxels.size());
//     for (size_t v = 0; v < voxels.size(); v++)
//     {
//     points[v] = Point2f(voxels[v]->x, voxels[v]->y);
//     cout << points[v];
//     }



//     Mat centers;
//     vector<int> labels;
//     double performance_measure = kmeans(points, current_k, labels, TermCriteria(TermCriteria::EPS+TermCriteria::COUNT, 10, acceptance_threshold), 3, KMEANS_PP_CENTERS, centers);
//     if (performance_measure >= switch_threshold){
//             current_k = min_clusters;
//             Mat centers_mink;
//             vector<int> labels_mink;
//             double min_clusters_measure = kmeans(points, current_k, labels_mink, TermCriteria(TermCriteria::EPS+TermCriteria::COUNT, 10, acceptance_threshold), 3, KMEANS_PP_CENTERS, centers_mink);
//             if (min_clusters_measure < performance_measure){
//                 centers = centers_mink;
//                 labels = labels_mink;
//                 //only label 3 clusters 
//             }
//         } 

//     for (size_t l = 0; l < labels.size(); l++) {
//     cout << labels[l];
//     }
//     return labels;
// }

// void findPixelsFromLabels(Mat centers, std::vector <int> labels){
    
//    voxels = getVisibleVoxels();





// }


// void gaussianMixtureModel(std::vector<int> labels){
//     //This code taken from: https://programming.vip/docs/gmm-gaussian-mixture-model-method-for-opencv-image-segmentation.html

//     scene3d.setCurrentFrame(514);
//     scene3d.getCameras()[3]->setVideoFrame(scene3d.getCurrentFrame());
//     Mat img = scene3d.getCamera().getFrame(); 

//     // So here, we need to take the labels that we got from the clustering, find out what pixels they correspond to, 
//     //get the pixel values that these correspond to (only considering the pixels at a certain height since the trousers are not very descriptive) 
//     //train the EM model on these pixels, save the model, then we can just load the model and whenever we get a new frame, we find the cluster centers, take the same pixels at that height(y), 
//     //feed them cluster-wise to the Gaussian mixture model predict2 function, which will output what class it belongs to, then use this to properly color the voxel model
    

//     //display picture
//     namedWindow("src pic",WINDOW_AUTOSIZE);
//     imshow("src pic",img);

//     //Define 4 colors 
//     Scalar color_tab[]={
//         Scalar(0,0,255),
//         Scalar(0,255,0),
//         Scalar(255,0,0),
//         Scalar(255,0,255)
//     };

//     int width = img.cols;
//     int height = img.rows;
//     int dims = img.channels();

//     int nsamples = width*height;
//     Mat points(nsamples, dims, CV_64FC1);
//     Mat labels;
//     Mat result = Mat::zeros(src.size(), CV_8UC3);

//     //Define classification, that is, how many classification points of function K value
//     int no_clusters = 4;

//     // Image RGB pixel data to sample data
//    int index = 0;
//    for (int row = 0; row < height; row++) {
//        for (int col = 0; col < width; col++) {
//            index = row*width + col;
//            Vec3b rgb = src.at<Vec3b>(row, col);
//            points.at<double>(index, 0) = static_cast<int>(rgb[0]);
//            points.at<double>(index, 1) = static_cast<int>(rgb[1]);
//            points.at<double>(index, 2) = static_cast<int>(rgb[2]);
//        }
//    }


//    // EM Cluster Train
//    Ptr<EM> GMM_model = EM::create();
//    //Partition number
//    GMM_model->setClustersNumber(no_clusters);
//    //Set covariance matrix type
//    GMM_model->setCovarianceMatrixType(EM::COV_MAT_SPHERICAL);
//    //Set convergence conditions
//    GMM_model->setTermCriteria(TermCriteria(TermCriteria::EPS + TermCriteria::COUNT, 100, 0.1));
//    //Store the probability partition to labs EM according to the sample training
//    GMM_model->trainEM(points, noArray(), labels, noArray());



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
//            int response = cvRound(GMM_model->predict2(sample, noArray())[1]);
//            Scalar c = color_tab[response];
//            result.at<Vec3b>(row, col)[0] = c[0];
//            result.at<Vec3b>(row, col)[1] = c[1];
//            result.at<Vec3b>(row, col)[2] = c[2];
//        }
//    }
//    imshow("EM-Segmentation", result);

//    waitKey(0);
//    destroyAllWindows();

// }





















// }