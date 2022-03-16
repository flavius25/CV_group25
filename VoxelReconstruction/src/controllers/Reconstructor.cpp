/*
 * Reconstructor.cpp
 *
 *  Created on: Nov 15, 2013
 *      Author: coert
 */

#include "Reconstructor.h"
#include "Hungarian.h"

#include <opencv2/opencv.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/operations.hpp>
#include <opencv2/core/types_c.h>
#include <math.h>
#include <cassert>
#include <iostream>
#include <iterator>
#include <opencv2/imgproc/imgproc.hpp>

#include "../utilities/General.h"

using namespace std;
using namespace cv;
using namespace cv::ml;

namespace nl_uu_science_gmt
{

/**
 * Constructor
 * Voxel reconstruction class
 */
Reconstructor::Reconstructor(
		const vector<Camera*> &cs) :
				m_cameras(cs),
				m_height(2048), //2048 for debug //2560 optimal
				m_step(32)
{
	for (size_t c = 0; c < m_cameras.size(); ++c)
	{
		if (m_plane_size.area() > 0)
			assert(m_plane_size.width == m_cameras[c]->getSize().width && m_plane_size.height == m_cameras[c]->getSize().height);
		else
			m_plane_size = m_cameras[c]->getSize();
	}

	const size_t edge = 2 * m_height;
	m_voxels_amount = (edge / m_step) * (edge / m_step) * (m_height / m_step);

	initialize();
	offlineOcclusionPrep();
}

/**
 * Deconstructor
 * Free the memory of the pointer vectors
 */
Reconstructor::~Reconstructor()
{
	for (size_t c = 0; c < m_corners.size(); ++c)
		delete m_corners.at(c);
	for (size_t v = 0; v < m_voxels.size(); ++v)
		delete m_voxels.at(v);
}

/**
 * Create some Look Up Tables
 * 	- LUT for the scene's box corners
 * 	- LUT with a map of the entire voxelspace: point-on-cam to voxels
 * 	- LUT with a map of the entire voxelspace: voxel to cam points-on-cam
 */
void Reconstructor::initialize()
{
	// Cube dimensions from [(-m_height, m_height), (-m_height, m_height), (0, m_height)]
	const int xL = -m_height; // - 1000
	const int xR = m_height; //  - 1000
	const int yL = -m_height; //  + 500
	const int yR = m_height; //  + 500
	const int zL = 0;
	const int zR = m_height;
	const int plane_y = (yR - yL) / m_step;
	const int plane_x = (xR - xL) / m_step;
	const int plane = plane_y * plane_x;

	// Save the 8 volume corners
	// bottom
	m_corners.push_back(new Point3f((float) xL, (float) yL, (float) zL));
	m_corners.push_back(new Point3f((float) xL, (float) yR, (float) zL));
	m_corners.push_back(new Point3f((float) xR, (float) yR, (float) zL));
	m_corners.push_back(new Point3f((float) xR, (float) yL, (float) zL));

	// top
	m_corners.push_back(new Point3f((float) xL, (float) yL, (float) zR));
	m_corners.push_back(new Point3f((float) xL, (float) yR, (float) zR));
	m_corners.push_back(new Point3f((float) xR, (float) yR, (float) zR));
	m_corners.push_back(new Point3f((float) xR, (float) yL, (float) zR));

	// Acquire some memory for efficiency
	cout << "Initializing " << m_voxels_amount << " voxels ";
	m_voxels.resize(m_voxels_amount);

	int z;
	int pdone = 0;
#pragma omp parallel for schedule(static) private(z) shared(pdone)
	for (z = zL; z < zR; z += m_step)
	{
		const int zp = (z - zL) / m_step;
		int done = cvRound((zp * plane / (double) m_voxels_amount) * 100.0);

#pragma omp critical
		if (done > pdone)
		{
			pdone = done;
			cout << done << "%..." << flush;
		}

		int y, x;
		for (y = yL; y < yR; y += m_step)
		{
			const int yp = (y - yL) / m_step;

			for (x = xL; x < xR; x += m_step)
			{
				const int xp = (x - xL) / m_step;

				// Create all voxels
				Voxel* voxel = new Voxel;
				voxel->x = x;
				voxel->y = y;
				voxel->z = z;
				voxel->camera_projection = vector<Point>(m_cameras.size());
				voxel->valid_camera_projection = vector<int>(m_cameras.size(), 0);

				const int p = zp * plane + yp * plane_x + xp;  // The voxel's index

				for (size_t c = 0; c < m_cameras.size(); ++c)
				{
					Point point = m_cameras[c]->projectOnView(Point3f((float) x, (float) y, (float) z));

					// Save the pixel coordinates 'point' of the voxel projection on camera 'c'
					voxel->camera_projection[(int) c] = point;

					// If it's within the camera's FoV, flag the projection
					if (point.x >= 0 && point.x < m_plane_size.width && point.y >= 0 && point.y < m_plane_size.height)
						voxel->valid_camera_projection[(int) c] = 1;
				}

				//Writing voxel 'p' is not critical as it's unique (thread safe)
				m_voxels[p] = voxel;
			}
		}
	}

	cout << "done!" << endl;

}

void Reconstructor::offlineOcclusionPrep()
{
#pragma omp parallel
{
	vector <int> cameras_used2 = { 2,3 };
	std::vector<Voxel*> voxels_distance;

#pragma omp for{
		for (int i = 0; i < m_voxels.size(); i++) {
			/*visible_voxels[i]->label = labels[i];
			int i_label = visible_voxels[i]->label;*/

			for (int k = 0; k < m_voxels.size(); k++) {
				/*visible_voxels[k]->label = labels[k];
				int k_label = visible_voxels[k]->label;*/

				if (!(m_voxels[i] == m_voxels[k])) {

					if ((m_voxels[i]->z > (m_height * 2 / 5)) && (m_voxels[k]->z > (m_height * 2 / 5))) {

						for (int c = 0; c < cameras_used2.size(); c++) {

							const Point i_point = m_voxels[i]->camera_projection[c];
							const Point k_point = m_voxels[k]->camera_projection[c];

							//int radius_region_of_interest = 5;			//check if the k_point is in the region (radius of circle around i_point). Currently set to 5
							//((k_point.x - i_point.x) * (k_point.x - i_point.x) + (k_point.y - i_point.y) * (k_point.y - i_point.y) <= radius_region_of_interest * radius_region_of_interest) 


							if (i_point == k_point){ 
								Vec3f camWPoint = m_cameras[c]->getCameraLocation();
								Vec3f i_W_point = Point3f(i_point.x, i_point.y, (float)m_voxels[i]->z);
								Vec3f k_W_point = Point3f(k_point.x, k_point.y, (float)m_voxels[k]->z);
								float i_Cam_dist = norm(i_W_point, camWPoint, NORM_L2);
								float k_Cam_dist = norm(k_W_point, camWPoint, NORM_L2);

								if (i_Cam_dist > k_Cam_dist) {
									m_voxels[i]->voxels_occluded.push_back(m_voxels[k]);
								}
							
							}

						}

					}

				}

			}
		}
	}	
	
}

/**
 * Count the amount of camera's each voxel in the space appears on,
 * if that amount equals the amount of cameras, add that voxel to the
 * visible_voxels vector
 */
void Reconstructor::update()
{
	countFrames++;
	////cout << countFrames << "\n";
	//if (countFrames == 514) {
	//	//Mat frame = m_cameras[3]->getVideoFrame(514);
	//	Mat act_frame = m_cameras[3]->getFrame();
	//	imshow("frame2", act_frame);
	//}
	m_visible_voxels.clear();
	//m_visible_voxels_frame.clear();
	std::vector<Voxel*> visible_voxels;
	std::vector<Voxel*> visible_voxels_frame;

	int v;
#pragma omp parallel for schedule(static) private(v) shared(visible_voxels)
	for (v = 0; v < (int)m_voxels_amount; ++v)
	{
		int camera_counter = 0;
		Voxel* voxel = m_voxels[v];

		for (size_t c = 0; c < m_cameras.size(); ++c)
		{

			if (voxel->valid_camera_projection[c])
			{
				const Point point = voxel->camera_projection[c];

				//If there's a white pixel on the foreground image at the projection point, add the camera
				if (m_cameras[c]->getForegroundImage().at<uchar>(point) == 255) ++camera_counter;
			}

		}

		// If the voxel is present on all cameras
		if (camera_counter == m_cameras.size())
		{
#pragma omp critical //push_back is critical
			visible_voxels.push_back(voxel);			//for online
			if (countFrames == 4)
			{
#pragma omp critical //push_back is critical
				visible_voxels_frame.push_back(voxel);	//for offline
			}
		}

	}


	// *O	
	// *	F
	// *		F
	// *			L
	// *				I
	// *					N
	// *						E


	//OFFLINE PHASE    -- only run once
	//for frame 514 process labels and centers
	if (!visible_voxels_frame.empty()) {
		//m_visible_voxels_frame.insert(m_visible_voxels_frame.end(), visible_voxels_frame.begin(), visible_voxels_frame.end());
		vector<Point2f> groundCoordinates_frame(visible_voxels_frame.size());
		for (int i = 0; i < (int)visible_voxels_frame.size(); i++) {

			if (visible_voxels_frame[i]->z > (m_height * 2 / 5))		//we only take the points that we are interested in (upper-body)
			{
				groundCoordinates_frame[i] = Point2f(visible_voxels_frame[i]->x, visible_voxels_frame[i]->y);
			}
		}
		std::vector<int> labels_frame;
		kmeans(groundCoordinates_frame, 4, labels_frame, TermCriteria(CV_TERMCRIT_ITER, 10, 1.0), 3, KMEANS_PP_CENTERS, centers_frame);

		//get currentframe in img
		Mat img = m_cameras[1]->getFrame();
		//Mat img2 = m_cameras[2]->getFrame();

		Mat samples1;
		Mat samples2;
		Mat samples3;
		Mat samples4;

		//asign m_visible_voxels_frame to labels
		for (int i = 0; i < visible_voxels_frame.size(); i++) {
			visible_voxels_frame[i]->label = labels_frame[i];
			int label_no = labels_frame[i];
	
			const Point point_forrgb = visible_voxels_frame[i]->camera_projection[1];
			//const Point point_forrgb2 = visible_voxels_frame[i]->camera_projection[2];
			cv::Vec3b rgb = img.at<cv::Vec3b>(point_forrgb);		//get original RGB values for pixels of interest
			//cv::Vec3b rgb2 = img2.at<cv::Vec3b>(point_forrgb2);		//get original RGB values for pixels of interest

			Mat rgb_r(1, 3, CV_64FC1);
			rgb_r.at<double>(0, 0) = static_cast<int>(rgb[0]);
			rgb_r.at<double>(0, 1) = static_cast<int>(rgb[1]);
			rgb_r.at<double>(0, 2) = static_cast<int>(rgb[2]);

			//Mat rgb_r2(1, 3, CV_64FC1);
			//rgb_r2.at<double>(0, 0) = static_cast<int>(rgb[0]);
			//rgb_r2.at<double>(0, 1) = static_cast<int>(rgb[1]);
			//rgb_r2.at<double>(0, 2) = static_cast<int>(rgb[2]);

			switch (label_no) {
			case 0:
				samples1.push_back(rgb_r);
				//samples1.push_back(rgb_r2);
				break;
			case 1:
				samples2.push_back(rgb_r);
				//samples2.push_back(rgb_r2);
				break;
			case 2:
				samples3.push_back(rgb_r);
				//samples3.push_back(rgb_r2);
				break;
			case 3:
				samples4.push_back(rgb_r);
				//samples4.push_back(rgb_r2);
				break;
			}
		}


		//m_groundCoordinates_frame.assign(groundCoordinates_frame.begin(), groundCoordinates_frame.end());
		//m_labels_frame.assign(labels_frame.begin(), labels_frame.end());

		//Put number of clusters to 3 here, corresponding to number of color components, as each person has roughly 3 specific colors
		int no_clusters = 2; //good results with 2

		//Create model 1
		Ptr<EM> GMM_model1 = EM::create();
		//Initialise number of clusters to look for 
		GMM_model1->setClustersNumber(no_clusters);
		//Set covariance matrix type
		GMM_model1->setCovarianceMatrixType(EM::COV_MAT_SPHERICAL);
		//Set convergence conditions
		GMM_model1->setTermCriteria(TermCriteria(TermCriteria::EPS + TermCriteria::COUNT, 100, 0.1));
		//Store the probability partition to labs EM according to the sample training
		GMM_model1->trainEM(samples1);


		//Create model 2  
		Ptr<EM> GMM_model2 = EM::create();
		GMM_model2->setClustersNumber(no_clusters);
		GMM_model2->setCovarianceMatrixType(EM::COV_MAT_SPHERICAL);
		GMM_model2->setTermCriteria(TermCriteria(TermCriteria::EPS + TermCriteria::COUNT, 100, 0.1));
		GMM_model2->trainEM(samples2);


		//Create model 3 
		Ptr<EM> GMM_model3 = EM::create(); 
		GMM_model3->setClustersNumber(no_clusters);
		GMM_model3->setCovarianceMatrixType(EM::COV_MAT_SPHERICAL);
		GMM_model3->setTermCriteria(TermCriteria(TermCriteria::EPS + TermCriteria::COUNT, 100, 0.1));
		GMM_model3->trainEM(samples3);


		//Create model 4
		Ptr<EM> GMM_model4 = EM::create();
		GMM_model4->setClustersNumber(no_clusters);
		GMM_model4->setCovarianceMatrixType(EM::COV_MAT_SPHERICAL);
		GMM_model4->setTermCriteria(TermCriteria(TermCriteria::EPS + TermCriteria::COUNT, 100, 0.1));
		GMM_model4->trainEM(samples4);
		//cout << labels;

		//Save model in xml-file
		GMM_model1->save("GMM_model1.xml");
		GMM_model2->save("GMM_model2.xml");
		GMM_model3->save("GMM_model3.xml");
		GMM_model4->save("GMM_model4.xml");
	
	}






	// *O	
	// *	N
	// *		L
	// *			I
	// *				N
	// *					E





	//ONLINE PHASE


	vector<Point2f> groundCoordinates(visible_voxels.size());
	m_visible_voxels.insert(m_visible_voxels.end(), visible_voxels.begin(), visible_voxels.end()); //I needed to comment this and add it at line 388 in for, so we can see the actual upper-body voxels

	for (int i = 0; i < (int)visible_voxels.size(); i++) {
			groundCoordinates[i] = Point2f(visible_voxels[i]->x, visible_voxels[i]->y);
	}

	std::vector<int> labels;				//labels

	kmeans(groundCoordinates, 4, labels, TermCriteria(CV_TERMCRIT_ITER, 10, 1.0), 3, KMEANS_PP_CENTERS, centers);

	//load the GMM_model
	Ptr<EM> GMM_model1 = EM::load("GMM_model1.xml");
	Ptr<EM> GMM_model2 = EM::load("GMM_model2.xml");
	Ptr<EM> GMM_model3 = EM::load("GMM_model3.xml");
	Ptr<EM> GMM_model4 = EM::load("GMM_model4.xml");

	//cout << centers;
	//cout << centers.at<float>(0,1);

	//get currentframe in img
	//Mat img = m_cameras[1]->getFrame();
	Mat img2 = m_cameras[2]->getFrame();
	Mat img3 = m_cameras[3]->getFrame();
	vector <Mat> img_vec = {img2, img3};

	//cv::Scalar colorCircle1(0, 0, 255);
	//cv::Point centerCircle1(centers.at<float>(0, 0), centers.at<float>(0, 1));

	//cv::circle(img, centerCircle1, 30, colorCircle1, 2);

	//imshow("View", img);

	Mat samples1;
	Mat samples2;
	Mat samples3;
	Mat samples4;

	vector <int> cameras_used = {2,3};

 
	// //Looping to check for occlusions, only consider pixels from voxels that are closest to camera
	// for (int i = 0; i < visible_voxels.size(); i++) {
	// 	visible_voxels[i]->label = labels[i];
	// 	int i_label = visible_voxels[i]->label;

	// 	for (int k = 0; k < visible_voxels.size(); k++){
	// 		visible_voxels[i]->label = labels[i];
	// 		int k_label = visible_voxels[k]->label;

	// 		if (!(visible_voxels[i] == visible_voxels[k])){
	// 			if ((visible_voxels[i]->z > (m_height * 1.5 / 5)) && (visible_voxels[k]->z > (m_height * 1.5 / 5))){
	// 				for (int c = 0; c < cameras_used.size(); c++){
	// 					const Point i_point = visible_voxels[i]->camera_projection[c];
	// 					const Point k_point = visible_voxels[k]->camera_projection[c];

	// 					if ((i_point == k_point) && !(i_label == k_label)){
	// 						Vec3f camWPoint= m_cameras[c]->getCameraLocation();
	// 						Vec3f i_W_point = Point3f(i_point.x, i_point.y, (float) visible_voxels[i]->z);
	// 						Vec3f k_W_point = Point3f(k_point.x, k_point.y, (float) visible_voxels[k]->z);
	// 						float i_Cam_dist = norm(i_W_point,camWPoint, NORM_L2);
	// 						float k_Cam_dist = norm(k_W_point,camWPoint, NORM_L2);
	// 						if(i_Cam_dist > k_Cam_dist){
	// 							continue;
	// 						}
	// 					else{
	// 						cv::Vec3b rgb = img_vec[c].at<cv::Vec3b>(i_point);	//get original RGB values for pixels of interest

	// 						Mat rgb_r(1, 3, CV_64FC1);
	// 						rgb_r.at<double>(0, 0) = static_cast<int>(rgb[0]);  //Put into matrix
	// 						rgb_r.at<double>(0, 1) = static_cast<int>(rgb[1]);
	// 						rgb_r.at<double>(0, 2) = static_cast<int>(rgb[2]);

	// 						switch (i_label) {
	// 						case 0:
	// 							samples1.push_back(rgb_r);
	// 							break;
	// 						case 1:
	// 							samples2.push_back(rgb_r);
	// 							break;
	// 						case 2:
	// 							samples3.push_back(rgb_r);
	// 							break;
	// 						case 3:
	// 							samples4.push_back(rgb_r);
	// 							break;
	// 							}
	// 						}	
	// 					}
	// 				}	
	// 			}
	// 		}
	// 	}

	// }
#pragma omp for
{
	for (int i = 0; i < visible_voxels.size(); i++) {
		visible_voxels[i]->label = labels[i];
		int i_label = visible_voxels[i]->label;

		if (visible_voxels[i]->z > (m_height * 1.5 / 5))
		{
			for (int c = 0; c < cameras_used.size(); c++){
				std::vector <Voxel*> occluding_voxels = visible_voxels[i]->voxels_occluded[c-2];

				bool occluding_voxels_on = false;
				for (int v = 0; v < occluding_voxels.size(); v++) {
					while (!occluding_voxels_on){
   						if (std::find(visible_voxels.begin(), visible_voxels.end(), v) != visible_voxels.end()) {
							occluding_voxels_on = true;
							}

				if (!occluding_voxels_on){		
				
					const Point point_forrgb = visible_voxels[i]->camera_projection[c];

					cv::Vec3b rgb = img_vec[c].at<cv::Vec3b>(point_forrgb);		//get original RGB values for pixels of interest
			
					
					Mat rgb_r(1, 3, CV_64FC1);
					rgb_r.at<double>(0, 0) = static_cast<int>(rgb[0]);
					rgb_r.at<double>(0, 1) = static_cast<int>(rgb[1]);
					rgb_r.at<double>(0, 2) = static_cast<int>(rgb[2]);

			

					switch (i_label) {
					case 0:
						samples1.push_back(rgb_r);
						break;
					case 1:
						samples2.push_back(rgb_r);
						break;
					case 2:
						samples3.push_back(rgb_r);
						break;
					case 3:
						samples4.push_back(rgb_r);
						break;
							}
						}
					}
				}
			}
		}
	}
}



	vector <Mat> matVec = {samples1, samples2, samples3, samples4}; //Array of all matrices to be combined into one


	vector <int> final_labels; 
	vector <vector <double>> costMatrix;

	for (int m = 0; m < matVec.size(); m++){
		int nr_rows = matVec[m].rows;
		vector <double> sums = {0.0, 0.0, 0.0, 0.0};
		for (int r = 0; r < nr_rows; r++){
			Mat row(1, 3, CV_64FC1);
			row.at<double>(0,0) = matVec[m].at<double>(r, 0);
			row.at<double>(0,1) = matVec[m].at<double>(r, 1);
			row.at<double>(0,2) = matVec[m].at<double>(r, 2);
			sums[0] += abs(GMM_model1->predict2(row, noArray())[0]);
			sums[1] += abs(GMM_model2->predict2(row, noArray())[0]);
			sums[2] += abs(GMM_model3->predict2(row, noArray())[0]);
			sums[3] += abs(GMM_model4->predict2(row, noArray())[0]);
		}
		costMatrix.push_back(sums);
	}

	HungarianAlgorithm HungAlgo;

	//find label for each sample
	double cost = HungAlgo.Solve(costMatrix, final_labels);
	
	//Deciding which center belongs to which person and putting in vector to be plotted on grid
	for (int i = 0; i < final_labels.size(); i++){
		Vec2f center = centers.row(i);
		int color_index = final_labels[i];
		center_labels[color_index].push_back(center);
	}


	//Assigning color labels to each visible voxel
	for (int i = 0; i < (int)visible_voxels.size(); i++) {
		int lb = visible_voxels[i]->label;
		int color_index = final_labels[lb];
		visible_voxels[i]->color = color_tab[color_index];
	}
	
}
}

/* namespace nl_uu_science_gmt */