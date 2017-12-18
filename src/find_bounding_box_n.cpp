#include <iostream>
#include <fstream>
#include <vector>
#include <opencv2/opencv.hpp>

using namespace std;

const char * eyes_cascade_name = "../models/common/haarcascade_lefteye_2splits.xml";
const char * bounding_box_name = "../bounding_box.txt";

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

	cv::VideoCapture cam(0);
	if(!cam.isOpened())
	{
		std::cout << "Failed to gather the capture.\n";
		return -1;
	}
	
	cv::Mat img;
	cam >> img;
	cv::transpose(img,img);
	cv::flip(img,img,1);
	const int cols = img.cols;
	const int rows = img.rows;

	cout << rows << 'x' << cols << endl;

	cv::namedWindow("Camera", CV_WINDOW_AUTOSIZE);

	if( !eyes_cascade.load(eyes_cascade_name) )
	{
		std::cout << "Failed to load the model file, please return.\n";
		return -1;
	}
	
	cv::Mat tmp = cv::Mat::zeros(rows, cols, CV_8UC1);

	std::vector<cv::Rect> eyeRegion;
	int x, y, w, h;
	x = y = w = h = 0;

	while(1)
	{
		cam >> img;
		cv::cvtColor(img,img,CV_BGR2GRAY);
		cv::transpose(img,tmp);
		cv::flip(tmp,tmp,1);
		cv::equalizeHist(tmp,tmp);

		eyes_cascade.detectMultiScale(tmp, eyeRegion, 1.1, 2, 0|CV_HAAR_SCALE_IMAGE, cv::Size(250, 250));

		if(eyeRegion.size()==1)
		{
			if((eyeRegion[0].width<=cols)&&(eyeRegion[0].height<=rows))
			{
				x = eyeRegion[0].x;
				y = eyeRegion[0].y;
				w = eyeRegion[0].width;
				h = eyeRegion[0].height;

				break;
			}
		
		}
		cout << "init...\n";
	}

	for(int i=0; i<30; )
	{
		cam >> img;
		cv::cvtColor(img,tmp,CV_BGR2GRAY);
		cv::transpose(img,tmp);
		cv::flip(tmp,tmp,1);
		cv::equalizeHist(tmp,tmp);

		eyes_cascade.detectMultiScale(tmp, eyeRegion, 1.1, 2, 0|CV_HAAR_SCALE_IMAGE, cv::Size(250, 250));

		if(eyeRegion.size()==1)
		{
			if((eyeRegion[0].width<=cols)&&(eyeRegion[0].height<=rows))
			{
				++i;
				
				h = 0.7 * h + 0.3 * eyeRegion[0].height;
				w = 0.7 * w + 0.3 * eyeRegion[0].width;
				x = 0.7 * x + 0.3 * eyeRegion[0].x;
				y = 0.7 * y + 0.3 * eyeRegion[0].y; 

				std::cout << "w: " << w << "\th: " << h << endl;
				std::cout << "Processing " << i << " frames.\n";
			}
		}
	}

	h = 0.45 * h;
	w = 0.90 * w;
	x += 0.08 * w;
	y += 1.25 * h;

	cv::Rect eyeROI = cv::Rect(x,y,w,h);

	cam >> img;
	cv::cvtColor(img,img,CV_BGR2GRAY);
	cv::transpose(img,tmp);
	cv::flip(tmp,tmp,1);

	cv::Mat bounding_eye = tmp(eyeROI);
	std::cout << bounding_eye.rows << 'x' << bounding_eye.cols << endl;
	cv::imwrite("../bounding_eye.jpg", bounding_eye);

	cv::rectangle(img, eyeROI, cv::Scalar(255, 0, 255), 4, 8 ,0);

	cv::imshow("Camera", img);

	ofstream fout(bounding_box_name);
	fout << x << '\t' << y << '\t' << w << '\t' << h << endl;
	fout.close();

	cv::waitKey(0);
		
	cv::destroyAllWindows();

	return 0;
}





