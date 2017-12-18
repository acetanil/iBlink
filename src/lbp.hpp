#include <opencv/opencv.hpp>
#include <cstream>

using namespace std;

int getHopCount(uint8_t code)
{
	int k = 7;
	int cnt = 0;
	int a[8] = { 0 };

	while (code)
		a[k] = code & 1, code >>= 1, --k;

	for (k = 0; k < 7; k++)
		if (a[k] != a[k + 1]) ++cnt;

	if (a[0] != a[7]) ++cnt;

	return cnt;
}


void cal_table() 
{
	unsigned char dim = 0;
	memset(table, 0, 256);
	for (int i = 0; i < 256; i++)
		if (getHopCount(i) <= 2)
			table[i] = ++dim;
}


int olbp_faeture(cv::Mat src, cv::Mat& dst)
{
	cv::Mat tmp(src.rows-2, src.cols-2, CV_8UC1);
	// src.depth(); src.channels(); => CV_8UC3: channels = 3; depths = 8U
	if(src.depth()!=1)
	{
		cout << "[LBP Error]: Unsupport Image Type.\n";
		return -1;
	}

	// do the actuall olbp
	tmp.setTo(0);
	for(int i=1; i<src.cols-1; i++)
		for(int j=1; j<src.rows-1;j++)
		{
			unsigned char code = 0;
			unsigned char center = src.at<unsigned char>(j,i);
			// the local binary pattern for 3x3 patch
			code |= (src.at<unsigned char>(j-1,i-1)>=center) << 7;
			code |= (src.at<unsigned char>(j  ,i-1)>=center) << 6;
			code |= (src.at<unsigned char>(j+1,i-1)>=center) << 5;
			code |= (src.at<unsigned char>(j+1,i  )>=center) << 4;
			code |= (src.at<unsigned char>(j+1,i+1)>=center) << 3;
			code |= (src.at<unsigned char>(j  ,i+1)>=center) << 2;
			code |= (src.at<unsigned char>(j-1,i+1)>=center) << 1;
			code |= (src.at<unsigned char>(j-1,i  )>=center) << 0;

			tmp.at<unsigned char>(j-1,i-1) = code;
		}

	tmp.copyTo(dst);

	return 0；
}

int ulbp_feature(cv::Mat src, cv::Mat& dst)
{
	if (src.depth() != 0)
	{
		cout << "[LBP Error]: Unsupport Image Type\n";
		return -1;
	}

	const int cols = src.cols;
	const int rows = src.rows;

	cv::copyMakeBorder(src,src,1,1,1,1,cv::BORDER_REPLICATE);
	
	dst = cv::Mat::zeros(rows, cols, CV_8UC1);

	cv::MatIterator_<uint8_t> it = src.begin<uint8_t>();
	cv::MatIterator_<uint8_t> it_ = dst.begin<uint8_t>();

	const int a = src.cols;
	const int b = dst.cols;

	for(int i=1;i<=rows;i++)
		for(int j=1;j<=cols;j++)
		{
			uint8_t code = 0;
			uint8_t center = *(it+a*i+j);

			code |= (*(it+a*(i-1)+(j-1))>=center) << 7;
			code |= (*(it+a*(i  )+(j-1))>=center) << 6;
			code |= (*(it+a*(i+1)+(j-1))>=center) << 5;
			code |= (*(it+a*(i+1)+(j  ))>=center) << 4;
			code |= (*(it+a*(i+1)+(j+1))>=center) << 3;
			code |= (*(it+a*(i  )+(j+1))>=center) << 2;
			code |= (*(it+a*(i-1)+(j+1))>=center) << 1;
			code |= (*(it+a*(i-1)+(j  ))>=center);

			*(it_+b*(i-1)+(j-1)) = table[code];
		}

	return 0;
}
