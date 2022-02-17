#include <opencv2/opencv.hpp>

using namespace cv;
 
int main(int argc, char **argv)
{

	//This is camera matrix and dist_coeffs , you will get in camera calibration.(copy from intrinsic.xml)  
    Mat camera_matrix = (cv::Mat_<double>(3,3) <<  1.2195112968898779e+003, 0., 3.6448211117862780e+002, 0.,
                                                        1.2414409169216196e+003, 2.4321803868732076e+002, 0., 0., 1.);
    Mat dist_coeffs = cv::Mat::zeros(4,1,cv::DataType<double>::type); // Assuming no lens distortion
    
	vector<Point2f>corners1;// for chessboard pattern
	Mat rotation_vector; // Rotation in axis-angle form
    Mat translation_vector;

	VideoCapture capture(0);
	Mat view;
	Mat Img;
	
	vector<Point3f>  objectPoints;
	
	const string x= "X";
	const string y= "Y";
	const string z= "Z";
	bool found=false;
	vector<Point3d> point3D;
    vector<Point2d> point2D;

	int ChessboardPatternWidth=9;//Horrizonal Number of internal corners of pattern //change it accrounding to your pattern 
	int ChessboardPatternHight=6;//vertical Number of internal corners of pattern //change it accrounding to your pattern 
	Size patternSize(ChessboardPatternWidth,ChessboardPatternHight);
	float BoardBoxSize=4;//distance between 2 correns //change it accrounding to your pattern . megger it in cm or mm.
	                     //your unit of meggerment will consider as object point units.
	for (int j=0; j<patternSize.height;j++)
	 {
		 for( int i=0; i < patternSize.width;i++)
	 {
		 objectPoints.push_back(Point3f(i*BoardBoxSize,j*BoardBoxSize,0));
     }
	 }
 
	//elow are the 3d object point(world point) to drow x , y z axis.
	point3D.push_back(Point3d(0,0,-10.0)); //-z this point represents 10( cm or mm accrounding to BoardBoxSize unit  ) 
    point3D.push_back(Point3d(10.0,0,0));  //x
	point3D.push_back(Point3d(0,10.0,0));  //y
	
	////elow are the 3d object point(world point) to drow Box.
    point3D.push_back(Point3d(12, 0,-12.0));//(x,y,z)
    point3D.push_back(Point3d(12, 8,-12.0));
	point3D.push_back(Point3d(20, 8,-12.0));
	point3D.push_back(Point3d(20, 0,-12.0));

	while(1)
	{
		capture >> view;

		
	if(view.empty()!=1)
	{	found=findChessboardCorners(view, patternSize,  corners1);//This will detect pattern
	}  

	if(found )

		{
			cvtColor(view,Img,CV_BGR2GRAY);
		
		cornerSubPix(Img, corners1, Size(11, 11), Size(-1, -1),
        TermCriteria(CV_TERMCRIT_EPS + CV_TERMCRIT_ITER, 30, 0.1));
		
		//	drawChessboardCorners(view, patternSize, corners1, found);
      
		// The following are two important funtion
		cv::solvePnP(objectPoints, corners1, camera_matrix, dist_coeffs, rotation_vector, translation_vector);//Gives you rotation_vector, translation_vector
        
		//following funtion gives you point2d from point3D world point to drow them on 2d image.
        cv::projectPoints(point3D, rotation_vector, translation_vector, camera_matrix, dist_coeffs, point2D);
		

		// following are just drowing funtion to drow object on output image.

		//Tp drow x,y z axis on image.
		cv::line(view,corners1[0], point2D[0], cv::Scalar(0,0,255), 3);//z
        cv::line(view,corners1[0], point2D[1], cv::Scalar(255,0,0), 3);//x
        cv::line(view,corners1[0], point2D[2], cv::Scalar(0,255,0), 3);//y
	 
	
    	putText(view, x, Point(point2D[1].x-10,point2D[1].y-10),FONT_HERSHEY_COMPLEX_SMALL,1, cv::Scalar(255,0,0),2 );
		putText(view, y, Point(point2D[2].x-10,point2D[2].y-10),FONT_HERSHEY_COMPLEX_SMALL,1, cv::Scalar(0,255,0),2 );
		putText(view, z, Point(point2D[0].x-10,point2D[0].y-10),FONT_HERSHEY_COMPLEX_SMALL,1, cv::Scalar(0,0,255),2 );
        circle(view,point2D[0], 3, cv::Scalar(0,0,255), 4, 8,0);
        circle(view,point2D[1], 3, cv::Scalar(255,0,0), 4, 8,0);
        circle(view,point2D[2], 3, cv::Scalar(0,255,0), 4, 8,0);
	
    // To drow box on image. It will writen for pattern size 9,6.
   //If you are using diffrent change corners1 point and point2D point. 

    cv::line(view,corners1[3],  point2D[3], cv::Scalar(0,0,255),6);
    cv::line(view,corners1[21], point2D[4],cv::Scalar(0,0,255), 6);
	cv::line(view,corners1[23], point2D[5],cv::Scalar(0,0,255), 6);
    cv::line(view,corners1[5],  point2D[6], cv::Scalar(0,0,255),6);
						 
    cv::line(view,corners1[3],corners1[5], cv::Scalar(0,255,0),  6);
    cv::line(view,corners1[5],corners1[23],cv::Scalar(0,255,0) ,6);
	cv::line(view,corners1[23],corners1[21],cv::Scalar(0,255,0),  6);
    cv::line(view,corners1[21],corners1[3],cv::Scalar(0,255,0),   6);
	
	cv::line(view,point2D[3], point2D[4], cv::Scalar(255,0,0), 6);
	cv::line(view,point2D[4], point2D[5], cv::Scalar(255,0,0), 6);
	cv::line(view,point2D[5], point2D[6], cv::Scalar(255,0,0), 6); 
	cv::line(view,point2D[3], point2D[6], cv::Scalar(255,0,0), 6);

		}

 // Display image.
    cv::imshow("Output", view);
    cv::waitKey(1);
	
}

	}