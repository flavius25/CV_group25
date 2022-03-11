/*
 * Reconstructor.h
 *
 *  Created on: Nov 15, 2013
 *      Author: coert
 */

#ifndef RECONSTRUCTOR_H_
#define RECONSTRUCTOR_H_

#include <opencv2/core/core.hpp>
#include <stddef.h>
#include <vector>

#include "Camera.h"

namespace nl_uu_science_gmt
{

class Reconstructor
{
public:
	/*
	 * Voxel structure
	 * Represents a 3D pixel in the half space
	 */
	struct Voxel
	{
		int x, y, z;                               // Coordinates
		cv::Vec3f color;                          // Color
		std::vector<cv::Point> camera_projection;  // Projection location for camera[c]'s FoV (2D)
		std::vector<int> valid_camera_projection;  // Flag if camera projection is in camera[c]'s FoV
		int label;
	};

	//Define 4 colors 
	std::vector <cv::Vec3f> color_tab = {
		{0,0,255},  //RGB
		{0,255,0},
		{255,0,0},
		{255,0,255}
    };

private:
	const std::vector<Camera*> &m_cameras;  // vector of pointers to cameras
	const int m_height;                     // Cube half-space height from floor to ceiling
	const int m_step;                       // Step size (space between voxels)
	int countFrames;						// generalCounter for frames	

	std::vector<cv::Point3f*> m_corners;			// Cube half-space corner locations
	std::vector<int> m_labels;						// labels
	std::vector<int> m_labels_frame;						// labels for specific frame
	cv::Mat centers;								// to store centers
	cv::Mat centers_frame;								// to store centers
	std::vector<cv::Point2f> m_groundCoordinates;	// take groundcoordinates
	std::vector<cv::Point2f> m_groundCoordinates_frame;	// take groundcoordinates for specific frame

	size_t m_voxels_amount;                 // Voxel count
	cv::Size m_plane_size;                  // Camera FoV plane WxH

	std::vector<Voxel*> m_voxels;           // Pointer vector to all voxels in the half-space
	std::vector<Voxel*> m_visible_voxels;   // Pointer vector to all visible voxels
	std::vector<Voxel*> m_visible_voxels_frame;   // Pointer vector to all visible voxels from specified frame

	void initialize();

public:
	Reconstructor(
			const std::vector<Camera*> &);
	virtual ~Reconstructor();

	void update();

	const std::vector<Voxel*>& getVisibleVoxels() const
	{
		return m_visible_voxels;
	}

	const std::vector<Voxel*>& getVoxels() const
	{
		return m_voxels;
	}

	void setVisibleVoxels(
			const std::vector<Voxel*>& visibleVoxels)
	{
		m_visible_voxels = visibleVoxels;
	}

	void setVoxels(
			const std::vector<Voxel*>& voxels)
	{
		m_voxels = voxels;
	}

	const std::vector<cv::Point3f*>& getCorners() const
	{
		return m_corners;
	}

	const std::vector<cv::Point2f>& getGroundCoordinates() const
	{
		return m_groundCoordinates;
	}

	const std::vector<cv::Point2f>& getGroundCoordinatesFrame() const
	{
		return m_groundCoordinates_frame;
	}

	const std::vector<int>& getLabels() const
	{
		return m_labels;
	}

	const std::vector<int>& getLabelsFrame() const
	{
		return m_labels_frame;
	}

	const std::vector<cv::Mat>& getCenters() const
	{
		return centers;
	}

	const std::vector<cv::Mat>& getCentersFrame() const
	{
		return centers_frame;
	}

	int getSize() const
	{
		return m_height;
	}

	const cv::Size& getPlaneSize() const
	{
		return m_plane_size;
	}
};

} /* namespace nl_uu_science_gmt */

#endif /* RECONSTRUCTOR_H_ */
