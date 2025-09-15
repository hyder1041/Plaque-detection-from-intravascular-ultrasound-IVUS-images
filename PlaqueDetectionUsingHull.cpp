/*
-----------------Pipeline Overview-----------------
1. Load and resize images.

2. Convert to grayscale + quantize intensities.

3. Apply erosion + dilation for noise removal.

4. Flood-fill borders to isolate main region.

5. Extract contours & convex hull → lumen detection.

6. Separate lumen region (blue mask).

7. Binarize lumen for contrast enhancement.

8. Detect plaques inside lumen using Canny + contours.

9. Display final segmentation results.
*/

// Include required OpenCV libraries
#include <opencv2/highgui/highgui.hpp>
#include "opencv2/core/core_c.h"
#include "opencv2/core/core.hpp"
#include "opencv2/flann/miniflann.hpp"
#include "opencv2/imgproc/imgproc_c.h"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/video/video.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/ml/ml.hpp"
#include "opencv2/highgui/highgui_c.h"
#include "opencv2/highgui/highgui.hpp"
#include <cv.h>
#include <iostream>

// Use OpenCV and std namespaces
using namespace cv;
using namespace std;

// Declare global image matrices
Mat src,resrc,quant,gray,edges,dilation_dst,erosion_dst,canny;

// Parameters for erosion/dilation
int erosion_elem = 0;
int erosion_size = 0;
int dilation_elem = 0;
int dilation_size = 0;
int const max_elem = 2;
int const max_kernel_size = 21;

// Function declarations
void Erosion( int, void* );
void Dilation( int, void* );


// Function: Apply erosion to the 'quant' image
void Erosion( int, void* )
{
  int erosion_type;
  if( erosion_elem == 0 ){ erosion_type = MORPH_RECT; }   // rectangle structuring element
  else if( erosion_elem == 1 ){ erosion_type = MORPH_CROSS; } // cross structuring element
  else if( erosion_elem == 2) { erosion_type = MORPH_ELLIPSE; } // ellipse structuring element

  // Create structuring element (3x3 rectangle here)
  Mat element = getStructuringElement( MORPH_RECT,
                                       Size( 2*1 + 1, 2*1+1 ),
                                       Point( 1, 1 ) );

  // Perform erosion
  erode( quant, erosion_dst, element );
}


// Function: Apply dilation to the 'erosion_dst' image
void Dilation( int, void* )
{
  int dilation_type;
  if( dilation_elem == 0 ){ dilation_type = MORPH_RECT; }
  else if( dilation_elem == 1 ){ dilation_type = MORPH_CROSS; }
  else if( dilation_elem == 2) { dilation_type = MORPH_ELLIPSE; }

  // Create structuring element (5x5 ellipse here)
  Mat element = getStructuringElement( MORPH_ELLIPSE,
                                       Size( 2*2 + 1, 2*2+1 ),
                                       Point( 2,2 ) );

  // Perform dilation
  dilate( erosion_dst, dilation_dst, element );
}


int main()
{
    // Path to dataset folder, you can download the dataset and change the path
	String folder = "/home/ali/IDP/Training_Set/Data_set_A/DCM";

    // Vector to hold filenames
    vector<String> filenames;    

    // Load all file paths in folder into filenames
    glob(folder, filenames);
     
    // Initialize images with given size and type
    gray = Mat(256,256, CV_32FC1);
    quant = Mat(256,256, CV_32FC1);

    // Copy of source image (currently empty)
    Mat src_copy = src.clone();

    // Used for storing contour indices
    float size[1];
    int flag=0;

    // Containers for contours
    Mat threshold_output;
    vector<vector<Point> > contours;
    vector<Vec4i> hierarchy;

    // Loop over first 1/10th of all files
    for (size_t i = 0; i < filenames.size()/10; ++i)
    {
        // Read current image
        src = imread(filenames[i]);

        // Resize image to 256x256
        resize(src,resrc,Size(256,256),INTER_NEAREST);

        // Convert to grayscale (two copies: gray, quant)
        cvtColor(resrc,gray, COLOR_BGR2GRAY);
        cvtColor(resrc,quant, COLOR_BGR2GRAY);
        
        // Loop over each pixel
        for (int j = 0; j<gray.rows; j++)
        {
            for (int k = 0; k<gray.cols; k++)
            {
                // Get pixel intensity
                Scalar inten=gray.at<uchar>(j,k);
                	
                // Invert intensity if greater than threshold
                if((inten.val[0]>25))
                {
                    quant.at<uchar>(j,k)=255-inten.val[0];
                }
                
                // Quantize intensity into fixed levels
                inten=quant.at<uchar>(j,k);
                quant.at<uchar>(j,k)=int(inten.val[0]*7/255);
                quant.at<uchar>(j,k)=int(quant.at<uchar>(j,k)*255/7);
                	
                // Threshold to binary (values >25 → 255)
                threshold( quant, quant, 25, 255,0 );
            }
        }

        // Apply erosion + dilation
        Erosion( 0, 0 );
        Dilation( 0, 0 );

        // Copy eroded image into mask for flood fill
        Mat mask;
        erosion_dst.copyTo(mask);

        // Flood fill top and bottom borders
        for (int i = 0; i < mask.cols; i++) 
        {
            if (mask.at<char>(0, i) == 0) 
                floodFill(mask, cv::Point(i, 0), 255, 0, 2, 2);

            if (mask.at<char>(mask.rows-1, i) == 0) 
                floodFill(mask, cv::Point(i, mask.rows-1), 255, 0, 2, 2);
        }

        // Flood fill left and right borders
        for (int i = 0; i < mask.rows; i++) 
        {
            if (mask.at<char>(i, 0) == 0) 
                floodFill(mask, cv::Point(0, i), 255, 0, 2, 2);

            if (mask.at<char>(i, mask.cols-1) == 0) 
                floodFill(mask, cv::Point(mask.cols-1, i), 255, 0, 2, 2);
        }

        // Copy erosion result into new image
        Mat newImage;
        erosion_dst.copyTo(newImage);

        // Replace background with white (255)
        for (int row = 0; row < mask.rows; ++row) 
        {
            for (int col = 0; col < mask.cols; ++col) 
            {
                if (mask.at<char>(row, col) == 0) 
                    newImage.at<char>(row, col) = 255;
            }
        }

        // Find contours in new image
        findContours(newImage, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));

        // Vector to hold convex hulls of contours
        vector<vector<Point> >hull( contours.size() );

        // Compute convex hull for each contour
        for( int i = 0; i < contours.size(); i++ )
        {  
            convexHull( Mat(contours[i]), hull[i], false );
        }

        // Convert grayscale image to RGB for drawing
        Mat img_rgb(gray.size(), CV_32FC3);  
        cvtColor(gray, img_rgb, CV_GRAY2RGB);       	   
        Mat drawing = Mat::zeros( newImage.size(), CV_8UC3 );

        // Loop through contours
        for( int i = 0; i< contours.size(); i++ )
        {
            // Only consider contours with large area (>4000)
            if((contourArea(contours[i])>4000))
            {
                // Draw convex hull filled in blue
                drawContours(img_rgb, hull, i, Scalar(255,0,0), CV_FILLED, 8, vector<Vec4i>(), 0, Point() );
            }
        }

        // Copy lumen-segmented image
        Mat img_rgb_copy(img_rgb.size(), CV_32FC3);
        Mat gray_copy = gray.clone();

        // Separate out blue region (lumen) from original
        for (int k = 0; k < img_rgb.rows; ++k)
        {
            for (int r = 0; r < img_rgb.cols; ++r)
            {
                Vec3b intensity = img_rgb.at<Vec3b>(k, r);
                float blue = intensity.val[0];
                float green = intensity.val[1];
                float red = intensity.val[2];

                // Keep only blue pixels, remove others
                if(!((blue==255)&&(green==0)&&(red==0)))
                {
                    gray_copy.at<uchar>(k,r)=0;
                }
            }	
        }

        // Initialize vectors for plaque segmentation
        vector<vector<Point> > contours_2;
        vector<Vec4i> hierarchy_2;

        // Copy gray_copy into final_detection
        Mat final_detection = gray_copy.clone();

        // Binarize lumen region: set pixels <100 → 0, else 255
        for (int s = 0; s <gray_copy.rows; ++s)
        {
            for (int t = 0; t <gray_copy.cols; ++t)
            {
                Scalar inten_copy=gray_copy.at<uchar>(s,t);
                if(inten_copy.val[0]<100)
                    final_detection.at<uchar>(s,t)=0;
                else
                    final_detection.at<uchar>(s,t)=255;
            }
        }

        // Apply Canny edge detection to final lumen image
        Canny(final_detection,canny,50,150);

        // Find contours of plaques
        findContours(canny, contours_2, hierarchy_2, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0) );

        // Loop through plaque contours
        for( int i = 0; i< contours_2.size(); i++ )
        {
            // Ignore very small contours (likely catheter noise)
            if((contourArea(contours_2[i])>5))
                drawContours(gray, contours_2, i,Scalar(255,0,0), 1, 8, vector<Vec4i>(), 0, Point() );   
        }

        // Resize gray image for display
        resize(gray,gray,Size(512,512),INTER_NEAREST);

        // Show intermediate and final results
        imshow("gray_copy",gray_copy);
        imshow("detected_region",final_detection);
        imshow("Hull", img_rgb);
        imshow("final",gray);

        // Invert dilation result for visualization
        Mat invSrc =  cv::Scalar::all(255) - dilation_dst;
        imshow("original",src);
        imshow("quant",invSrc);

        // Wait for key press before processing next image
        waitKey(0);
    }

    // Exit program
	return 0;
}
