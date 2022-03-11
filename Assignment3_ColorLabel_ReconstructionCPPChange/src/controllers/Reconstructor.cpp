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
		int camera_counter_frame = 0;
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

		if (countFrames == 4) {

			for (size_t c = 0; c < m_cameras.size(); ++c)
			{

				if (voxel->valid_camera_projection[3])
				{
					const Point point_frame = voxel->camera_projection[3];

					//If there's a white pixel on the foreground image at the projection point, add the camera
					if (m_cameras[c]->getForegroundImage().at<uchar>(point_frame) == 255) ++camera_counter_frame;
				}
			}
		}

		// If the voxel is present on all cameras
		if (camera_counter == m_cameras.size())
		{
#pragma omp critical //push_back is critical
			visible_voxels.push_back(voxel);
		}

		// If the voxel is present on all cameras
		if (camera_counter_frame == m_cameras.size())
		{
#pragma omp critical //push_back is critical
			visible_voxels_frame.push_back(voxel);
		}

	}

	//OFFLINE PHASE    -- only run once
	//for frame 514 process labels and centers
	if (!visible_voxels_frame.empty()) {
		m_visible_voxels_frame.insert(m_visible_voxels_frame.end(), visible_voxels_frame.begin(), visible_voxels_frame.end());
		vector<Point2f> groundCoordinates_frame(visible_voxels_frame.size());
		for (int i = 0; i < (int)m_visible_voxels_frame.size(); i++) {
			groundCoordinates_frame[i] = Point2f(m_visible_voxels_frame[i]->x, m_visible_voxels_frame[i]->y);
			cout << groundCoordinates_frame[i];
		}
		std::vector<int> labels_frame;
		kmeans(groundCoordinates_frame, 4, labels_frame, TermCriteria(CV_TERMCRIT_ITER, 10, 1.0), 3, KMEANS_PP_CENTERS, centers_frame);

		int label1_occur = count(labels_frame.begin(), labels_frame.end(), 0);
		int label2_occur = count(labels_frame.begin(), labels_frame.end(), 1);
		int label3_occur = count(labels_frame.begin(), labels_frame.end(), 2);
		int label4_occur = count(labels_frame.begin(), labels_frame.end(), 3);

		//get currentframe in img
		Mat img = m_cameras[3]->getFrame();
		//std::vector<cv::Vec3b> m_bgr; //vector for storing RGB values for voxel

		Mat label1(label1_occur, 3, CV_64FC1);
		Mat label2(label2_occur, 3, CV_64FC1);
		Mat label3(label3_occur, 3, CV_64FC1);
		Mat label4(label4_occur, 3, CV_64FC1);

		int index_label1 = 0;
		int index_label2 = 0;
		int index_label3 = 0;
		int index_label4 = 0;

		//asign m_visible_voxels_frame to labels
		for (int i = 0; i < m_visible_voxels_frame.size(); i++) {
			m_visible_voxels_frame[i]->label = labels_frame[i];
			int label_no = labels_frame[i];
			//Mat mat_name;

			const Point point_forrgb = m_visible_voxels_frame[i]->camera_projection[3];
			cv::Vec3b rgb = img.at<cv::Vec3b>(point_forrgb);		//get original RGB values for pixels of interest
			//m_bgr.push_back(bgr);

			//Mat row(1, 3, CV_64FC1);
			//row.at<double>(0, 0) = static_cast<int>(rgb[0]);
			//row.at<double>(0, 1) = static_cast<int>(rgb[1]);
			//row.at<double>(0, 2) = static_cast<int>(rgb[2]);

			switch (label_no) {
			case 0:
				label1.at<double>(index_label1, 0) = static_cast<int>(rgb[0]);
				label1.at<double>(index_label1, 1) = static_cast<int>(rgb[1]);
				label1.at<double>(index_label1, 2) = static_cast<int>(rgb[2]);
				index_label1++;
				break;
			case 1:
				label2.at<double>(index_label2, 0) = static_cast<int>(rgb[0]);
				label2.at<double>(index_label2, 1) = static_cast<int>(rgb[1]);
				label2.at<double>(index_label2, 2) = static_cast<int>(rgb[2]);
				index_label2++;
				break;
			case 2:
				label3.at<double>(index_label3, 0) = static_cast<int>(rgb[0]);
				label3.at<double>(index_label3, 1) = static_cast<int>(rgb[1]);
				label3.at<double>(index_label3, 2) = static_cast<int>(rgb[2]);
				index_label3++;
				break;
			case 3:
				label4.at<double>(index_label4, 0) = static_cast<int>(rgb[0]);
				label4.at<double>(index_label4, 1) = static_cast<int>(rgb[1]);
				label4.at<double>(index_label4, 2) = static_cast<int>(rgb[2]);
				index_label4++;
				break;
			}



			//cout << row;

			cout << "first: " << static_cast<int>(rgb[0]);
			cout << "second: " << static_cast<int>(rgb[1]);
			cout << "third: " << static_cast<int>(rgb[2]);
			//mat_name.push_back(row);

		}

		cout << "\nFirstlabel: \n" << label1;
		cout << "\nSecondlabel: \n" << label2;
		cout << "\nThirdlabel: \n" << label3;
		cout << "\nFourthlabel: \n" << label4;

		//Concatenate all matrices
		Mat matArray[] = { label1, label2, label3, label4 }; //Array of all matrices to be combined into one
		cv::Mat allClustersMat; //new matrix of all values, feed this to GMM model
		cv::vconcat(matArray, 4, allClustersMat);


		//Mat img_2 = m_bgr;
		//cout << m_bgr[0] << " " << m_bgr[1];

		m_groundCoordinates_frame.assign(groundCoordinates_frame.begin(), groundCoordinates_frame.end());
		m_labels_frame.assign(labels_frame.begin(), labels_frame.end());


		//if we do it like this, with img, we don't use at all kmeans. I am still not sure we need to use with GMM, since for this implementation you only need the frame.


		//cv::Vec3f bgr = img.at<cv::Vec3f>(m_visible_voxels_frame[0]);
		//m_visible_voxels_frame[0]->color = bgr;

		// int width = img.cols;
		// int height = img.rows;
		// int dims = img.channels();

		// int no_samples = width * height;
		// Mat points(no_samples, dims, CV_64FC1);
		Mat labels;
		// Mat result = Mat::zeros(img.size(), CV_8UC3);

		// //Define classification, that is, how many classification points of function K value
		// int no_clusters = 4;

		// // Find RGB pixel values from image coordinates and assign to points
		// int index = 0;
		// for (int row = 0; row < height; row++) {
		// 	for (int col = 0; col < width; col++) {
		// 		index = row * width + col;
		// 		Vec3b rgb = img.at<Vec3b>(row, col);
		// 		points.at<double>(index, 0) = static_cast<int>(rgb[0]);
		// 		points.at<double>(index, 1) = static_cast<int>(rgb[1]);
		// 		points.at<double>(index, 2) = static_cast<int>(rgb[2]);
		// 	}
		// }

		//Define classification, that is, how many classification points of function K value
		int no_clusters = 4;
		//Create model  
		Ptr<EM> GMM_model = EM::create();
		//Initialise number of clusters to look for 
		GMM_model->setClustersNumber(no_clusters);
		//Set covariance matrix type
		GMM_model->setCovarianceMatrixType(EM::COV_MAT_SPHERICAL);
		//Set convergence conditions
		GMM_model->setTermCriteria(TermCriteria(TermCriteria::EPS + TermCriteria::COUNT, 100, 0.1));
		//Store the probability partition to labs EM according to the sample training
		GMM_model->trainEM(allClustersMat, noArray(), labels, noArray());

		cout << labels;

		//Save model in xml-file
		GMM_model->save("GMM_model.xml");
	}

	//ONLINE PHASE
	//clustering for each frame
	m_visible_voxels.insert(m_visible_voxels.end(), visible_voxels.begin(), visible_voxels.end());

	vector<Point2f> groundCoordinates(m_visible_voxels.size());

	for (int i = 0; i < (int)m_visible_voxels.size(); i++) {
		groundCoordinates[i] = Point2f(m_visible_voxels[i]->x, m_visible_voxels[i]->y);
	}

	m_groundCoordinates.assign(groundCoordinates.begin(), groundCoordinates.end());

	std::vector<int> labels;								//labels

	kmeans(groundCoordinates, 4, labels, TermCriteria(CV_TERMCRIT_ITER, 10, 1.0), 3, KMEANS_PP_CENTERS, centers);

	m_labels.assign(labels.begin(), labels.end());

	//load the GMM_model
	Ptr<EM> GMM_model = EM::load("GMM_model.xml");

	int label1_occur = count(labels.begin(), labels.end(), 0);
	int label2_occur = count(labels.begin(), labels.end(), 1);
	int label3_occur = count(labels.begin(), labels.end(), 2);
	int label4_occur = count(labels.begin(), labels.end(), 3);

	//get currentframe in img
	Mat img = m_cameras[3]->getFrame();
	//std::vector<cv::Vec3b> m_bgr; //vector for storing RGB values for voxel

	Mat label1(label1_occur, 3, CV_64FC1);
	Mat label2(label2_occur, 3, CV_64FC1);
	Mat label3(label3_occur, 3, CV_64FC1);
	Mat label4(label4_occur, 3, CV_64FC1);

	int index_label1 = 0;
	int index_label2 = 0;
	int index_label3 = 0;
	int index_label4 = 0;


	//asign m_visible_voxels_frame to labels
	for (int i = 0; i < m_visible_voxels.size(); i++) {
		m_visible_voxels[i]->label = labels[i];
		int label_no = labels[i];
		//Mat mat_name;

		const Point point_forrgb = m_visible_voxels[i]->camera_projection[3];
		cv::Vec3b rgb = img.at<cv::Vec3b>(point_forrgb);		//get original RGB values for pixels of interest
		//m_bgr.push_back(bgr);

		//Mat row(1, 3, CV_64FC1);
		//row.at<double>(0, 0) = static_cast<int>(rgb[0]);
		//row.at<double>(0, 1) = static_cast<int>(rgb[1]);
		//row.at<double>(0, 2) = static_cast<int>(rgb[2]);

		switch (label_no) {
		case 0:
			label1.at<double>(index_label1, 0) = static_cast<int>(rgb[0]);
			label1.at<double>(index_label1, 1) = static_cast<int>(rgb[1]);
			label1.at<double>(index_label1, 2) = static_cast<int>(rgb[2]);
			index_label1++;
			break;
		case 1:
			label2.at<double>(index_label2, 0) = static_cast<int>(rgb[0]);
			label2.at<double>(index_label2, 1) = static_cast<int>(rgb[1]);
			label2.at<double>(index_label2, 2) = static_cast<int>(rgb[2]);
			index_label2++;
			break;
		case 2:
			label3.at<double>(index_label3, 0) = static_cast<int>(rgb[0]);
			label3.at<double>(index_label3, 1) = static_cast<int>(rgb[1]);
			label3.at<double>(index_label3, 2) = static_cast<int>(rgb[2]);
			index_label3++;
			break;
		case 3:
			label4.at<double>(index_label4, 0) = static_cast<int>(rgb[0]);
			label4.at<double>(index_label4, 1) = static_cast<int>(rgb[1]);
			label4.at<double>(index_label4, 2) = static_cast<int>(rgb[2]);
			index_label4++;
			break;
		}



		//cout << row;

		//cout << "first: " << static_cast<int>(rgb[0]);
		//cout << "second: " << static_cast<int>(rgb[1]);
		//cout << "third: " << static_cast<int>(rgb[2]);
		//mat_name.push_back(row);

	}

	//cout << "\nFirstlabel: \n" << label1;
	//cout << "\nSecondlabel: \n" << label2;
	//cout << "\nThirdlabel: \n" << label3;
	//cout << "\nFourthlabel: \n" << label4;

	//Create cluster matrices
	//std::vector <Mat> cluster_matrices;
	//cluster_matrices.push_back(label1);
	//cluster_matrices.push_back(label2);
	//cluster_matrices.push_back(label3);
	//cluster_matrices.push_back(label4);
	Mat matArray[] = { label1, label2, label3, label4 }; //Array of all matrices to be combined into one
	cv::Mat allClustersMat; //new matrix of all values, feed this to GMM model
	cv::vconcat(matArray, 4, allClustersMat);
	//vector of cluster matrices
	//std::vector <Mat> cluster_matrices;
	int nr_rows = allClustersMat.rows;

	Mat row(1, 3, CV_64FC1);
	//row= allClustersMat.at<float>(0);

	//cout << "CLusterMAT: \n" << allClustersMat;
	//cout << allClustersMat.at<Vec3f>(0);
	//Mat 
	std::vector <int> predictions;

	////for loop where prediction of color_label happens, important that all voxels have assigned label for which cluster they belong to (0,1,2,3) and that the cluster matrices are order the same
		for (int r = 0; r < nr_rows; r++) {
			row.at<double>(0,0) = allClustersMat.at<double>(r, 0);
			row.at<double>(0,1) = allClustersMat.at<double>(r, 1);
			row.at<double>(0,2) = allClustersMat.at<double>(r, 2);
			int prediction = cvRound(GMM_model->predict2(row, noArray())[1]);			//fixed the error, no color on the ouput still :D
			//cout << allClustersMat.at<int>(1);
			predictions.push_back(prediction);
		}

	for (int i = 0; i < (int)m_visible_voxels.size(); i++) {
		int lb = m_visible_voxels[i]->label;
		int color_index = predictions[lb];
		m_visible_voxels[i]->color = color_tab[color_index];


	}

	//cout << centers.at<float>(3,1) << "\n";
	//for (size_t l = 0; l < labels.size(); l++) {
	//	cout << labels[l];
	//}
	//cout << centers;
}

} /* namespace nl_uu_science_gmt */
