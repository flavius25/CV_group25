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

//#define CV_CALIB_CB_ADAPTIVE_THRESH 1
//#define CV_CALIB_CB_FAST_CHECK 8
//#define CV_CALIB_CB_NORMALIZE_IMAGE 2

// Defining the dimensions of checkerboard
int CHECKERBOARD[2]{ 6,8 };

using namespace cv;

int main()
{
    // Creating vector to store vectors of 3D points for each checkerboard image
    std::vector<std::vector<cv::Point3f> > objpoints;

    // Creating vector to store vectors of 2D points for each checkerboard image
    std::vector<std::vector<cv::Point2f> > imgpoints;

    std::vector<cv::Point2f> corner_pts;

    Mat gray;

    // Defining the world coordinates for 3D points
    std::vector<cv::Point3f> objp;
    for (int i{ 0 }; i < CHECKERBOARD[1]; i++)
    {
        for (int j{ 0 }; j < CHECKERBOARD[0]; j++)
            objp.push_back(cv::Point3f(115*j, 115*i, 0));
    }


    // Extracting path of individual image stored in a given directory
    std::vector<cv::String> images;
    // Path of the folder containing checkerboard images
    //std::string path = "./data/cam1/*.avi";

    // Create a VideoCapture object and open the input file
    // If the input is the web camera, pass 0 instead of the video file name
    VideoCapture cap("./data/cam1/intrinsics.avi");

    int contor = 0;

    //now it is set to take only the first 70 frames..
    while (contor<70) {

        
        Mat frame;
        // Capture frame-by-frame

        // retrieve and read decode the frame -> takes longer
        // grab it 15 times is quicker ... 
        // FFmpeg - to use for manual segmentation
        // Jasper: grab - does not do decoding of the avi file. (All videos or compressed and since we have an .avi file, this way of taking frame-by frame (cap >> frame;) is more slow. Hence, if we use cap.grab, we would take it faster.) 
        // Though, Poppe did not mention anything about grab, so we could use frame but maybe we could also look at grab.
 
        //cap.
        // algorithm way
        // to do -> 1. REad one frame then condition for skipping next 15 frames, and so on.
        // 2. Put them all together in an array and then we take each frame and calculate objpoints imgpoints
        // 3. call calibrateCamera function and get cameraMatrix
        // 4. export to xml file.

        // mechanical way
        // 1. we need an open cv function to observe the video in here and then screen shot at some points. 

        //so we have like 4 cameras, and we want to have 4 cameraMatrix. But, the .avi file is too large(too many frames/too many images). 
        //Question: Which one should we take? A: The variant of our colleague Jasper was to skip randomly maybe by his observation to skip 15 frames and when he sees a glare frame, then he skips only 5 frames, because he looked on how many seconds does it last (the glaring thing) and he avoids then.
        cap >> frame; 

        // Either that, I want to take the next frame and then skip the next 15. (15 frames equals to 1,2 seconds from the video) or we use how Poppe recommended -> FFmpeg


        // If the frame is empty, break immediately
        if (frame.empty())
            break;

        // Display the resulting frame
        imshow("Frame", frame);

        bool success;

        // frame = cv::imread(images[i]);

         //int x = frame.size().width;
         //int y = frame.size().height;

         //int down_width = 1280;
         //int down_height = 720;
         //Mat resized_up;

         //resize(frame, resized_up, Size(down_width, down_height), INTER_LINEAR);

        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);

        // Finding checker board corners
        // If desired number of corners are found in the image then success = true  
        success = cv::findChessboardCorners(gray, cv::Size(CHECKERBOARD[0], CHECKERBOARD[1]), corner_pts, CALIB_CB_ADAPTIVE_THRESH | CALIB_CB_FAST_CHECK | CALIB_CB_NORMALIZE_IMAGE);

        /*
         * If desired number of corner are detected,
         * we refine the pixel coordinates and display
         * them on the images of checker board
        */
        if (success)
        {
            cv::TermCriteria criteria(TermCriteria::EPS | TermCriteria::MAX_ITER, 30, 0.001);

            // refining pixel coordinates for given 2d points.
            cv::cornerSubPix(gray, corner_pts, cv::Size(11, 11), cv::Size(-1, -1), criteria);

            // Displaying the detected corner points on the checker board
            //cv::drawChessboardCorners(frame, cv::Size(CHECKERBOARD[0], CHECKERBOARD[1]), corner_pts, success);

            objpoints.push_back(objp);
            imgpoints.push_back(corner_pts);
        }

        

 
        contor++;


        // Press  ESC on keyboard to exit
        char c = (char)waitKey(25);
        if (c == 27)
            break;


    }



    // When everything done, release the video capture object
    cap.release();

    //cv::glob(path, images);

   // cv::Mat frame, gray;
    // vector to store the pixel coordinates of detected checker board corners 
    
    
        //int down_width = 1280;
        //int down_height = 720;
        //Mat resized_up;

        //resize(frame, resized_up, Size(down_width, down_height), INTER_LINEAR);


        //cv::imshow("Image", frame);
        //cv::waitKey(0);

    cv::destroyAllWindows();

    cv::Mat cameraMatrix, distCoeffs, R, T;

    /*
     * Performing camera calibration by
     * passing the value of known 3D points (objpoints)
     * and corresponding pixel coordinates of the
     * detected corners (imgpoints)
    */
    cv::calibrateCamera(objpoints, imgpoints, cv::Size(gray.rows, gray.cols), cameraMatrix, distCoeffs, R, T);


    std::cout << "cameraMatrix : " << cameraMatrix << std::endl;
    std::cout << "distCoeffs : " << distCoeffs << std::endl;
    std::cout << "Rotation vector : " << R << std::endl;
    std::cout << "Translation vector : " << T << std::endl;

    return 0;
}
