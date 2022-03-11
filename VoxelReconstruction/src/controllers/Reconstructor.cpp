/*
 * Reconstructor.cpp
 *
 *  Created on: Nov 15, 2013
 *      Author: coert
 */

#include "Reconstructor.h"

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
				m_height(2048),
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
	const int xL = -m_height;
	const int xR = m_height;
	const int yL = -m_height;
	const int yR = m_height;
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
	m_visible_voxels_frame.clear();
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
			groundCoordinates_frame[i] = Point2f(visible_voxels_frame[i]->x, visible_voxels_frame[i]->y);
			//cout << groundCoordinates_frame[i];
		}
		std::vector<int> labels_frame;
		kmeans(groundCoordinates_frame, 4, labels_frame, TermCriteria(CV_TERMCRIT_ITER, 10, 1.0), 3, KMEANS_PP_CENTERS, centers_frame);

		int cluster1_occur = count(labels_frame.begin(), labels_frame.end(), 0);
		int cluster2_occur = count(labels_frame.begin(), labels_frame.end(), 1);
		int cluster3_occur = count(labels_frame.begin(), labels_frame.end(), 2);
		int cluster4_occur = count(labels_frame.begin(), labels_frame.end(), 3);

		//get currentframe in img
		Mat img = m_cameras[1]->getFrame();
		//std::vector<cv::Vec3b> m_bgr; //vector for storing RGB values for voxel

		Mat cluster1(cluster1_occur, 3, CV_64FC1);
		Mat cluster2(cluster2_occur, 3, CV_64FC1);
		Mat cluster3(cluster3_occur, 3, CV_64FC1);
		Mat cluster4(cluster4_occur, 3, CV_64FC1);

		int index_cluster1 = 0;
		int index_cluster2 = 0;
		int index_cluster3 = 0;
		int index_cluster4 = 0;

		//asign m_visible_voxels_frame to labels
		for (int i = 0; i < visible_voxels_frame.size(); i++) {
			visible_voxels_frame[i]->label = labels_frame[i];
			int label_no = labels_frame[i];
			//Mat mat_name;
			
			if (m_visible_voxels_frame[i]->z > (m_height * 2/5))
					{
						const Point point_forrgb = m_visible_voxels_frame[i]->camera_projection[1];
						cv::Vec3b rgb = img.at<cv::Vec3b>(point_forrgb);		//get original RGB values for pixels of interest

						switch (label_no) {
						case 0:
							cluster1.at<double>(index_cluster1, 0) = static_cast<int>(rgb[0]);
							cluster1.at<double>(index_cluster1, 1) = static_cast<int>(rgb[1]);
							cluster1.at<double>(index_cluster1, 2) = static_cast<int>(rgb[2]);
							index_cluster1++;
							cluster1_occur--; 
							break;
						case 1:
							cluster2.at<double>(index_cluster2, 0) = static_cast<int>(rgb[0]);
							cluster2.at<double>(index_cluster2, 1) = static_cast<int>(rgb[1]);
							cluster2.at<double>(index_cluster2, 2) = static_cast<int>(rgb[2]);
							index_cluster2++;
							cluster2_occur--;
							break;
						case 2:
							cluster3.at<double>(index_cluster3, 0) = static_cast<int>(rgb[0]);
							cluster3.at<double>(index_cluster3, 1) = static_cast<int>(rgb[1]);
							cluster3.at<double>(index_cluster3, 2) = static_cast<int>(rgb[2]);
							index_cluster3++;
							cluster3_occur--;
							break;
						case 3:
							cluster4.at<double>(index_cluster4, 0) = static_cast<int>(rgb[0]);
							cluster4.at<double>(index_cluster4, 1) = static_cast<int>(rgb[1]);
							cluster4.at<double>(index_cluster4, 2) = static_cast<int>(rgb[2]);
							index_cluster4++;
							cluster4_occur--;
							break;
						}
					}

					//How to implement less voxels idea: so we need the Mat we are pushing these two to be the exact size 
					//(maybe double check first that it wouldnt work with only putting zeros at the empty places) but basically we can use the decreasing value of clusterX_occur 
					//to determine how many elements are missing from the matrix, n then deleting the last rows of the matrix based on this. 


			//cout << row;

			//cout << "first: " << static_cast<int>(rgb[0]);
			//cout << "second: " << static_cast<int>(rgb[1]);
			//cout << "third: " << static_cast<int>(rgb[2]);
			//mat_name.push_back(row);

		}

		//cout << "\nFirstlabel: \n" << cluster1;
		//cout << "\nSecondlabel: \n" << cluster2;
		//cout << "\nThirdlabel: \n" << cluster3;
		//cout << "\nFourthlabel: \n" << cluster4;

		//Mat img_2 = m_bgr;
		//cout << m_bgr[0] << " " << m_bgr[1];

		//m_groundCoordinates_frame.assign(groundCoordinates_frame.begin(), groundCoordinates_frame.end());
		//m_labels_frame.assign(labels_frame.begin(), labels_frame.end());

		//Put number of clusters to 3 here, corresponding to number of color components, as each person has roughly 3 specific colors
		int no_clusters = 4;

		//Create model 1
		Ptr<EM> GMM_model1 = EM::create();
		//Initialise number of clusters to look for 
		GMM_model1->setClustersNumber(no_clusters);
		//Set covariance matrix type
		GMM_model1->setCovarianceMatrixType(EM::COV_MAT_SPHERICAL);
		//Set convergence conditions
		GMM_model1->setTermCriteria(TermCriteria(TermCriteria::EPS + TermCriteria::COUNT, 100, 0.1));
		//Store the probability partition to labs EM according to the sample training
		GMM_model1->trainEM(cluster1);


		//Create model 2  
		Ptr<EM> GMM_model2 = EM::create();
		GMM_model2->setClustersNumber(no_clusters);
		GMM_model2->setCovarianceMatrixType(EM::COV_MAT_SPHERICAL);
		GMM_model2->setTermCriteria(TermCriteria(TermCriteria::EPS + TermCriteria::COUNT, 100, 0.1));
		GMM_model2->trainEM(cluster2);


		//Create model 3 
		Ptr<EM> GMM_model3 = EM::create(); 
		GMM_model3->setClustersNumber(no_clusters);
		GMM_model3->setCovarianceMatrixType(EM::COV_MAT_SPHERICAL);
		GMM_model3->setTermCriteria(TermCriteria(TermCriteria::EPS + TermCriteria::COUNT, 100, 0.1));
		GMM_model3->trainEM(cluster3);


		//Create model 4
		Ptr<EM> GMM_model4 = EM::create();
		GMM_model4->setClustersNumber(no_clusters);
		GMM_model4->setCovarianceMatrixType(EM::COV_MAT_SPHERICAL);
		GMM_model4->setTermCriteria(TermCriteria(TermCriteria::EPS + TermCriteria::COUNT, 100, 0.1));
		GMM_model4->trainEM(cluster4);
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
	//clustering for each frame
	m_visible_voxels.insert(m_visible_voxels.end(), visible_voxels.begin(), visible_voxels.end());

	vector<Point2f> groundCoordinates(visible_voxels.size());

	for (int i = 0; i < (int)visible_voxels.size(); i++) {
		groundCoordinates[i] = Point2f(visible_voxels[i]->x, visible_voxels[i]->y);
	}

	std::vector<int> labels;								//labels

	kmeans(groundCoordinates, 4, labels, TermCriteria(CV_TERMCRIT_ITER, 10, 1.0), 3, KMEANS_PP_CENTERS, centers);

	//load the GMM_model
	Ptr<EM> GMM_model1 = EM::load("GMM_model1.xml");
	Ptr<EM> GMM_model2 = EM::load("GMM_model2.xml");
	Ptr<EM> GMM_model3 = EM::load("GMM_model3.xml");
	Ptr<EM> GMM_model4 = EM::load("GMM_model4.xml");

	int cluster1_occur = count(labels.begin(), labels.end(), 0);
	int cluster2_occur = count(labels.begin(), labels.end(), 1);
	int cluster3_occur = count(labels.begin(), labels.end(), 2);
	int cluster4_occur = count(labels.begin(), labels.end(), 3);

	//get currentframe in img
	Mat img = m_cameras[1]->getFrame();

	Mat cluster1(cluster1_occur, 3, CV_64FC1);
	Mat cluster2(cluster2_occur, 3, CV_64FC1);
	Mat cluster3(cluster3_occur, 3, CV_64FC1);
	Mat cluster4(cluster4_occur, 3, CV_64FC1);

	int index_cluster1 = 0;
	int index_cluster2 = 0;
	int index_cluster3 = 0;
	int index_cluster4 = 0;


	//asign m_visible_voxels to labels
	for (int i = 0; i < visible_voxels.size(); i++) {
		visible_voxels[i]->label = labels[i];
		int label_no = labels[i];


		if (m_visible_voxels_frame[i]->z < (m_height * 2/5)){
			cout << "This is true";

			const Point point_forrgb = m_visible_voxels[i]->camera_projection[1];
			cv::Vec3b rgb = img.at<cv::Vec3b>(point_forrgb);		//get original RGB values for pixels of interest


			switch (label_no) {
			case 0:
				cluster1.at<double>(index_cluster1, 0) = static_cast<int>(rgb[0]);
				cluster1.at<double>(index_cluster1, 1) = static_cast<int>(rgb[1]);
				cluster1.at<double>(index_cluster1, 2) = static_cast<int>(rgb[2]);
				index_cluster1++;
				cluster1_occur--;
				break;
			case 1:
				cluster2.at<double>(index_cluster2, 0) = static_cast<int>(rgb[0]);
				cluster2.at<double>(index_cluster2, 1) = static_cast<int>(rgb[1]);
				cluster2.at<double>(index_cluster2, 2) = static_cast<int>(rgb[2]);
				index_cluster2++;
				cluster2_occur--;
				break;
			case 2:
				cluster3.at<double>(index_cluster3, 0) = static_cast<int>(rgb[0]);
				cluster3.at<double>(index_cluster3, 1) = static_cast<int>(rgb[1]);
				cluster3.at<double>(index_cluster3, 2) = static_cast<int>(rgb[2]);
				index_cluster3++;
				cluster3_occur--;
				break;
			case 3:
				cluster4.at<double>(index_cluster4, 0) = static_cast<int>(rgb[0]);
				cluster4.at<double>(index_cluster4, 1) = static_cast<int>(rgb[1]);
				cluster4.at<double>(index_cluster4, 2) = static_cast<int>(rgb[2]);
				index_cluster4++;
				cluster4_occur--;
				break;
			}

		}


		//cout << row;

		//cout << "first: " << static_cast<int>(rgb[0]);
		//cout << "second: " << static_cast<int>(rgb[1]);
		//cout << "third: " << static_cast<int>(rgb[2]);
		//mat_name.push_back(row);

	}

	//cout << "\nFirstlabel: \n" << cluster1;
	//cout << "\nSecondlabel: \n" << cluster2;
	//cout << "\nThirdlabel: \n" << cluster3;
	//cout << "\nFourthlabel: \n" << cluster4;

	vector <Mat> matVec = { cluster1, cluster2, cluster3, cluster4 }; //Array of all matrices to be combined into one

	//vector of cluster matrices
	//std::vector <Mat> cluster_matrices;
		//row= allClustersMat.at<float>(0);

	//cout << "CLusterMAT: \n" << allClustersMat;
	//cout << allClustersMat.at<Vec3f>(0);
	//Mat 
	vector <int> final_labels; 

	for (int m = 0; m < matVec.size(); m++){
		int nr_rows = matVec[m].rows;
		vector <float> sums = {0.0, 0.0, 0.0, 0.0};
		for (int r = 0; r < nr_rows; r++){
			Mat row(1, 3, CV_64FC1);
			row.at<double>(0,0) = matVec[m].at<double>(r, 0);
			row.at<double>(0,1) = matVec[m].at<double>(r, 1);
			row.at<double>(0,2) = matVec[m].at<double>(r, 2);
			sums[0] += GMM_model1->predict2(row, noArray())[0];
			sums[1] += GMM_model2->predict2(row, noArray())[0];
			sums[2] += GMM_model3->predict2(row, noArray())[0];
			sums[3] += GMM_model4->predict2(row, noArray())[0];
		}
		int maxElementIndex = std::max_element(sums.begin(), sums.end()) - sums.begin(); //calculated once
		final_labels.push_back(maxElementIndex);										 //we always put maxElementIndex
	}
	

	//Need check here for double assignation -> perhaps if double assignation remake/refine GMM models with only upper torso, otherwise try kmeans with 3 clusters?

	// ////for loop where prediction of color_label happens, important that all voxels have assigned label for which cluster they belong to (0,1,2,3) and that the cluster matrices are order the same
	// 	for (int r = 0; r < nr_rows; r++) {
	// 		row.at<double>(0,0) = allClustersMat.at<double>(r, 0);
	// 		row.at<double>(0,1) = allClustersMat.at<double>(r, 1);
	// 		row.at<double>(0,2) = allClustersMat.at<double>(r, 2);
	// 		int prediction = cvRound(GMM_model->predict2(row, noArray())[1]);			//fixed the error, no color on the ouput still :D
	// 		//cout << allClustersMat.at<int>(1);
	// 		predictions.push_back(prediction);
	// 	}

	for (int i = 0; i < (int)visible_voxels.size(); i++) {
		int lb = visible_voxels[i]->label;
		int color_index = final_labels[lb];
		//Vec3b c = color_tab[1];
		visible_voxels[i]->color = color_tab[color_index];
		//cout << color_tab[1];
	}

	//cout << centers.at<float>(3,1) << "\n";
	//for (size_t l = 0; l < labels.size(); l++) {
	//	cout << labels[l];
	//}
	//cout << centers;
}

} /* namespace nl_uu_science_gmt */