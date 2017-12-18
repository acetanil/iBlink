#include <iostream>
#include <fstream>
#include <vector>
#include <opencv2/opencv.hpp>

using namespace std;

const char * eyes_cascade_name = "../models/common/haarcascade_righteye_2splits.xml";

cv::CascadeClassifier eyes_cascade;

int help()
{
	std::cout << "Find the bounding box param for indicidual person.\n";
	std::cout << "Use the 30 frames to calculate a suitable bounding\n";
	std::cout << "box in order for the svm model to work properly.\n";

	return 0;
}

int main( int argc, const char * argv[] )
{
	help();

	cv::VideoCapture cam(1);
	if(!cam.isOpened())
	{
		cout << "Failed to gather the capture.\n";
		return -1;
	}
	
	cv::Mat img;
	cam >> img;
	const int cols = img.cols;
	const int rows = img.rows;

	cout << rows << 'x' << cols << endl;

	cv::namedWindow("Camera", CV_WINDOW_AUTOSIZE);
	cv::resize(img, img, cv::Size(0.5*rows,0.5*cols));
	cv::imshow("Camera", img);

	if( !eyes_cascade.load(eyes_cascade_name) )
	{
		std::cout << "Failed to load the model file, please return.\n";
		return -1;
	}
	
	cv::Mat tmp = cv::Mat::zeros(rows, cols, CV_8UC1);
	img.convertTo(tmp, CV_8UC1);
	//cv::equalizeHist(tmp, tmp);


	// prepare for the eye region detection
	std::vector<cv::Rect> eyeRegion;
	int hgt, wdh, xorg, yorg;

	eyes_cascade.detectMultiScale(tmp, eyeRegion, 1.1, 2, 0|CV_HAAR_SCALE_IMAGE, cv::Size(250, 250));

	if(eyeRegion.size()==1)
	{
		hgt  = eyeRegion[0].height;
		wdh  = eyeRegion[0].width;
		xorg = eyeRegion[0].x;
		yorg = eyeRegion[0].y;
	}

	for(int i=0; i<30; )
	{
		cam >> img;
		cv::cvtColor(img,tmp,CV_BGR2GRAY);
		cv::equalizeHist(tmp, tmp);

		eyes_cascade.detectMultiScale(tmp, eyeRegion, 1.1, 2, 0|CV_HAAR_SCALE_IMAGE, cv::Size(250, 250));

		if(eyeRegion.size()==1)
		{
			++i; // 
			hgt  = 0.7*hgt  + 0.3*eyeRegion[0].height;
			wdh  = 0.7*wdh  + 0.3*eyeRegion[0].width;
			xorg = 0.7*xorg + 0.3*eyeRegion[0].x;
			yorg = 0.7*yorg + 0.3*eyeRegion[0].y; 

			std::cout << "Processing " << i << " frames.\n";
		}
	}

	hgt  = 0.45 * hgt;
	wdh  = 0.90 * wdh;
	xorg = xorg + 0.08 * wdh;
	yorg = yorg + 1.15 * hgt;

	cv::Rect eyeROI = cv::Rect(xorg,yorg,wdh,hgt);

	cout << xorg <<'\t'<< yorg <<'\t'<< hgt <<'\t'<< wdh << endl;

	cam >> img;
	img.resize(rows,cols);
	img.convertTo(tmp,CV_8UC1);

	cv::Mat bounding_eye = tmp(eyeROI);
	std::cout << bounding_eye.rows << 'x' << bounding_eye.cols << endl;
	cv::imwrite("../models/bounding_eye.jpg", bounding_eye);

	cv::rectangle(img, eyeROI, cv::Scalar(255, 0, 255), 4, 8 ,0);

	cv::imshow("Camera", img);

	ofstream fout("../bounding_box.txt");
	fout << xorg << '\t' << yorg << '\t' << wdh << '\t' << hgt << endl;
	fout.close();

	cv::waitKey(0);
		
	cv::destroyAllWindows();

	return 0;
}





