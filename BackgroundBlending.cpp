
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include <iostream>

using namespace cv;


int main( void )
{

  for (int i{1}; i<5; i++){
    // Path of the folder containing checkerboard images
    String number = std::to_string(i);
    std::string path = "./cam" + number + "/background_images/*.png";
    //Declaration of vector with images
    std::vector<cv::String> images;

    cv::glob(path, images);

    std::vector<cv::Mat> image_matrices;

    for (int i{0}; i < images.size(); i++){
      cv::Mat this_image = imread(images[i]);
      image_matrices.push_back(this_image);
    }

    cv::Mat dst;
    cv::Mat mean_img = image_matrices[0];
    for (int i{0}; i < image_matrices.size(); i++){
      if (i == 0){
        continue;
      }
      else {
        double alpha = 0.5;
        double beta = (1.0 - alpha);
        addWeighted(image_matrices[i], alpha, mean_img, beta, 0.0, dst);
      }
    }

    imwrite("./cam" + number + "/background.png", mean_img);
    imshow("Background camera " + number, dst);
    waitKey(0);
  }
    return 0;

}


// for i in range(len(image_data)):
//     if i == 0:
//         pass
//     else:
//         alpha = 1.0/(i + 1)
//         beta = 1.0 - alpha
//         avg_image = cv2.addWeighted(image_data[i], alpha, avg_image, beta, 0.0)

// cv2.imwrite('avg_happy_face.png', avg_image)
// avg_image = cv2.imread('avg_happy_face.png')
// plt.imshow(avg_image)
// plt.show()







  //  double alpha = 0.5; double beta; double input;

  //  Mat src1, src2, dst;
  //  std::cout << " Simple Linear Blender " << std::endl;
  //  std::cout << "-----------------------" << std::endl;
  //  std::cout << "* Enter alpha [0.0-1.0]: ";
  //  std::cin >> input;
  //  // We use the alpha provided by the user if it is between 0 and 1
  //  if( input >= 0 && input <= 1 )
  //    { alpha = input; }
  //  src1 = imread( samples::findFile("LinuxLogo.jpg") );
  //  src2 = imread( samples::findFile("WindowsLogo.jpg") );
  //  if( src1.empty() ) { std::cout << "Error loading src1" << std::endl; return EXIT_FAILURE; }
  //  if( src2.empty() ) { std::cout << "Error loading src2" << std::endl; return EXIT_FAILURE; }
  //  beta = ( 1.0 - alpha );
  //  addWeighted( src1, alpha, src2, beta, 0.0, dst);
  //  imshow( "Linear Blend", dst );
  //  waitKey(0);
  //  return 0;


// # import all image files with the .jpg extension
// images = glob.glob ("happy_faces/*.jpg")


// image_data = []
// for img in images:
//     this_image = cv2.imread(img, 1)
//     image_data.append(this_image)

// avg_image = image_data[0]
// for i in range(len(image_data)):
//     if i == 0:
//         pass
//     else:
//         alpha = 1.0/(i + 1)
//         beta = 1.0 - alpha
//         avg_image = cv2.addWeighted(image_data[i], alpha, avg_image, beta, 0.0)

// cv2.imwrite('avg_happy_face.png', avg_image)
// avg_image = cv2.imread('avg_happy_face.png')
// plt.imshow(avg_image)
// plt.show()
