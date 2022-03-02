/*
 * Scene3DRenderer.cpp
 *
 *  Created on: Nov 15, 2013
 *      Author: coert
 */

#include "Scene3DRenderer.h"

#include <opencv2/core/mat.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/imgproc/types_c.h>
#include <stddef.h>
#include <string>

#include "../utilities/General.h"
#include <chrono>
#include <iostream>
using namespace std::chrono;

using namespace std;
using namespace cv;

namespace nl_uu_science_gmt
{

/**
 * Constructor
 * Scene properties class (mostly called by Glut)
 */
Scene3DRenderer::Scene3DRenderer(
		Reconstructor &r, const vector<Camera*> &cs) :
				m_reconstructor(r),
				m_cameras(cs),
				m_num(4),
				m_sphere_radius(1850)
{
	m_width = 640;
	m_height = 480;
	m_quit = false;
	m_paused = false;
	m_rotate = false;
	m_camera_view = true;
	m_show_volume = true;
	m_show_grd_flr = true;
	m_show_cam = true;
	m_show_org = true;
	m_show_arcball = false;
	m_show_info = true;
	m_fullscreen = false;

	// Read the checkerboard properties (XML)
	FileStorage fs;
	fs.open(m_cameras.front()->getDataPath() + ".." + string(PATH_SEP) + General::CBConfigFile, FileStorage::READ);
	if (fs.isOpened())
	{
		fs["CheckerBoardWidth"] >> m_board_size.width;
		fs["CheckerBoardHeight"] >> m_board_size.height;
		fs["CheckerBoardSquareSize"] >> m_square_side_len;
	}
	fs.release();

	m_current_camera = 0;
	m_previous_camera = 0;

	m_number_of_frames = m_cameras.front()->getFramesAmount();
	m_current_frame = 0;
	m_previous_frame = -1;

	const int H = 0;
	const int S = 0;
	const int V = 0;
	m_h_threshold = H;
	m_ph_threshold = H;
	m_s_threshold = S;
	m_ps_threshold = S;
	m_v_threshold = V;
	m_pv_threshold = V;

	createTrackbar("Frame", VIDEO_WINDOW, &m_current_frame, m_number_of_frames - 2);
	createTrackbar("H", VIDEO_WINDOW, &m_h_threshold, 255);
	createTrackbar("S", VIDEO_WINDOW, &m_s_threshold, 255);
	createTrackbar("V", VIDEO_WINDOW, &m_v_threshold, 255);

	createFloorGrid();
	setTopView();
}

/**
 * Deconstructor
 * Free the memory of the floor_grid pointer vector
 */
Scene3DRenderer::~Scene3DRenderer()
{
	for (size_t f = 0; f < m_floor_grid.size(); ++f)
		for (size_t g = 0; g < m_floor_grid[f].size(); ++g)
			delete m_floor_grid[f][g];
}

/**
 * Process the current frame on each camera
 */
bool Scene3DRenderer::processFrame()
{
	for (size_t c = 0; c < m_cameras.size(); ++c)
	{
		if (m_current_frame == m_previous_frame + 1)
		{
			m_cameras[c]->advanceVideoFrame();
		}
		else if (m_current_frame != m_previous_frame)
		{
			m_cameras[c]->getVideoFrame(m_current_frame);
		}
		assert(m_cameras[c] != NULL);

		//for counting execution time. flag before functiontime
		auto start = high_resolution_clock::now();
		processForeground(m_cameras[c]);
		// After function call
		auto stop = high_resolution_clock::now();
		auto duration = duration_cast<milliseconds>(stop - start);

		// To get the value of duration use the count()
		// member function on the duration object
		//std::cout << duration.count() << endl;
	}
	return true;
}

/**
 * Separate the background from the foreground
 * ie.: Create an 8 bit image where only the foreground of the scene is white (255)
 */
void Scene3DRenderer::processForeground(
		Camera* camera)
{
    //Get the foregroundMask learnt in camera initialization
	Mat foregroundMask = camera->getSubtractorMask();

	assert(!camera->getFrame().empty());
	Mat video_frame;
	video_frame = camera->getFrame();						  //get the frame from the video

	/*--------------------XOR optimization frame-to-frame--------------not workingyet
	Mat video_frame_last;

	bool first_time = true;

	assert(!camera->getFrame().empty());
	Mat video_frame, video_frame_thres, video_frame_last_thres, result_xor;
	video_frame = camera->getFrame();

	//video_frame.copyTo(images);

	video_frame_last = camera->getFrame();
	//video = camera->getVideo();


	//if (first_time) {
	//	video_frame_last = video_frame.clone();
	//	first_time = false;
	//}

	//for vide_frame
	cvtColor(video_frame, video_frame_thres, CV_RGB2GRAY);
	threshold(video_frame_thres, video_frame_thres, 70, 255, CV_THRESH_BINARY);

	//for last video_frame
	cvtColor(video_frame_last, video_frame_last_thres, CV_RGB2GRAY);
	threshold(video_frame_last_thres, video_frame_last_thres, 70, 255, CV_THRESH_BINARY);

	bitwise_xor(video_frame_thres, video_frame_last_thres, result_xor);

	imshow("video_frame_thres", video_frame_thres);
	imshow("video_frame_last_thres", video_frame_last_thres);
	imshow("bitwise", result_xor);

    video_frame_last = video_frame.clone(); // to be set after if conditions

	*/

	if (camera->getId() == 2) {

		vector<vector<Point>> contours;  //define vector for contours

		//get frame from the video
		camera->pBackSub->apply(video_frame, foregroundMask, 0); //learningrate is 0, we don't want to "learn" at this stage

		//post_processing (perform opening with 3x3)
		int morph_size = 1;
		Mat element = getStructuringElement(MORPH_ELLIPSE, Size(2 * morph_size + 1, 2 * morph_size + 1), Point(morph_size, morph_size));
		morphologyEx(foregroundMask, foregroundMask, 2, element); //operation 2 for opening = erosion + dilation (smoothing the image)

		Mat inverted = foregroundMask.clone();
		Mat thresh;
		bitwise_not(foregroundMask, inverted);
		threshold(inverted, thresh, 100, 255, THRESH_BINARY);
		bitwise_not(thresh, thresh);

		findContours(thresh, contours, RETR_TREE, CHAIN_APPROX_SIMPLE); //find contours

		//find maximum blob on the 1st level
		double maxArea = 0;
		int maxAreaContourId = -1;
		for (int j = 0; j < contours.size(); j++) {
			double newArea = cv::contourArea(contours.at(j));
			if (newArea > maxArea) {
				maxArea = newArea;
				maxAreaContourId = j;
			} // End if
		} // End for

		//eliminate maximum area on the 1st level
		vector<vector<Point>> contours2;

		for (int i = maxAreaContourId+1; i < contours.size();i++) {
			contours2.push_back(contours.at(i));
		}

		//find maximum blob on the 2nd level
		maxArea = 0;
		maxAreaContourId = -1;
		int maxContourIdchair_legs = -1;
		for (int j = 0; j < contours2.size(); j++) {
			double newArea = cv::contourArea(contours2.at(j));
			if (newArea > maxArea) {
				maxArea = newArea;
				maxAreaContourId = j;
				maxContourIdchair_legs = j;
			} // End if
		} // End for


		//eliminate maximum area on the 2nd level
		vector<vector<Point>> contours3;

		for (int i = maxAreaContourId + 1; i < contours2.size();i++) {
			contours3.push_back(contours2.at(i));
		}

		//find maximum blob on the 3rd level
		maxArea = 0;
		maxAreaContourId = -1;
		for (int j = 0; j < contours3.size(); j++) {
			double newArea = cv::contourArea(contours3.at(j));
			if (newArea > maxArea) {
				maxArea = newArea;
				maxAreaContourId = j;
			} // End if
		} // End for

		drawContours(foregroundMask, contours3, maxAreaContourId, Scalar(255, 255, 255), -1); //identified hole in camera 3 and filled the gap
		drawContours(foregroundMask, contours2, maxContourIdchair_legs, Scalar(0, 0, 0), -1); //space between chair and legs for camera 3 is better contoured with this method.
	}
	else {

		camera->pBackSub->apply(video_frame, foregroundMask, 0); //learningrate is 0, we don't want to "learn" at this stage

		//post_processing (perform opening with 3x3)
		int morph_size = 1;
		Mat element = getStructuringElement(MORPH_ELLIPSE, Size(2 * morph_size + 1, 2 * morph_size + 1), Point(morph_size, morph_size));
		morphologyEx(foregroundMask, foregroundMask, 2, element); //operation 2 for opening = erosion + dilation (smoothing the image)
	}

	camera->setForegroundImage(foregroundMask);
}

/**
 * Set currently visible camera to the given camera id
 */
void Scene3DRenderer::setCamera(
		int camera)
{
	m_camera_view = true;

	if (m_current_camera != camera)
	{
		m_previous_camera = m_current_camera;
		m_current_camera = camera;
		m_arcball_eye.x = m_cameras[camera]->getCameraPlane()[0].x;
		m_arcball_eye.y = m_cameras[camera]->getCameraPlane()[0].y;
		m_arcball_eye.z = m_cameras[camera]->getCameraPlane()[0].z;
		m_arcball_up.x = 0.0f;
		m_arcball_up.y = 0.0f;
		m_arcball_up.z = 1.0f;
	}
}

/**
 * Set the 3D scene to bird's eye view
 */
void Scene3DRenderer::setTopView()
{
	m_camera_view = false;
	if (m_current_camera != -1)
		m_previous_camera = m_current_camera;
	m_current_camera = -1;

	m_arcball_eye = vec(0.0f, 0.0f, 10000.0f);
	m_arcball_centre = vec(0.0f, 0.0f, 0.0f);
	m_arcball_up = vec(0.0f, 1.0f, 0.0f);
}

/**
 * Create a LUT for the floor grid
 */
void Scene3DRenderer::createFloorGrid()
{
	const int size = m_reconstructor.getSize() / m_num;
	const int z_offset = 3;

	// edge 1
	vector<Point3i*> edge1;
	for (int y = -size * m_num; y <= size * m_num; y += size)
		edge1.push_back(new Point3i(-size * m_num, y, z_offset));

	// edge 2
	vector<Point3i*> edge2;
	for (int x = -size * m_num; x <= size * m_num; x += size)
		edge2.push_back(new Point3i(x, size * m_num, z_offset));

	// edge 3
	vector<Point3i*> edge3;
	for (int y = -size * m_num; y <= size * m_num; y += size)
		edge3.push_back(new Point3i(size * m_num, y, z_offset));

	// edge 4
	vector<Point3i*> edge4;
	for (int x = -size * m_num; x <= size * m_num; x += size)
		edge4.push_back(new Point3i(x, -size * m_num, z_offset));

	m_floor_grid.push_back(edge1);
	m_floor_grid.push_back(edge2);
	m_floor_grid.push_back(edge3);
	m_floor_grid.push_back(edge4);
}

} /* namespace nl_uu_science_gmt */
