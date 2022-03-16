Required libraries (Linux):
===========================

opencv_core
opencv_highgui
opencv_imgproc
opencv_calib3d

GL
GLU
glut

Required libraries (Windows):
=============================

opencv_world

GL
GLU

=========================================================================================
Information about the files:
- Most of our implementations have been made in Reconstructor.cpp, as well as in Glut.cpp. 
- For the use of Hungarian algorithm, two files, Hungarian.cpp & Hungarian.h were added from repository https://github.com/mcximing/hungarian-algorithm-cpp 
- There exist two Reconstructor.cpp files, one where we deal with occlusion and one without, they can be found in Reconstructor_Occlusions and Reconstructor_normal respectively

___________________________________________________________________________________________
For Reconstructor_Occlusions - Reconstructor.cpp:

* line 41 - we set the step size to 64 here to make the calculations for the offline phase run faster. Notice however that it is still expected to take about 1-2     minutes at the beginning of each run to do this calculation! 
* line 78-81 - this is where we increased the voxel box so that the people do not walk outside of the bounding box
* line 160-200 - this is the code for the offline phase for the occlusion where we calculate which voxels occlude other voxels and put these in a vector within   the Voxel struct (see Reconstructor.h)
* line 239-244 - this is where we get the voxels specifically for frame 533 to be used in the offline phase building the color model
* line 249-279 - offline phase of color building, see especially:
          * line 264 - here we only take the voxels of the upper body
          * line 270 - kmeans algorithm
          * line 283 - 338 - only consider the voxels that are not occluded but still in visible voxels as samples, code for taking pixel and putting in matrix
          * line 343-378 - instantiation, training and saving of the different GMM models.
* line 391-532 - online phase, called for each framed. See especially:
          * line 403-407 - loading of GMM models
          * line 419-484 - only considering the voxels that are not occluded among the visible voxels for samples
          * 493-509 - this is where we feed each sample cluster to each of the model and store the results in a costMatrix (line 490)
          * 515 - feeding the costMatrix to the Hungarian algorithm, getting the final_labels for each sample
          * 520-524 - assigning the center coordinates to the correct labels to get the same color as person they belong to
          * 527-530 - assigning the labels (color) to each person based on the final_labels output from the Hungarian algorithm
___________________________________________________________________________________________________________________________________
For Reconstructor_normal - Reconstructor.cpp:

* This file is exactly like the above one but not dealing with occlusions, therefore most functions already exist in Reconstructor_Occlusions. See notes there.
_______________________________________________________________________________________________________________________________________
For Glut.cpp:

* line 672-697 - function drawTracking() implemented to draw dots on the gridfloor according to same color as person is assigned at the time. 

________________________________________________________________________________________________________________________________________________

       
