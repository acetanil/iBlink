#include "opencv2/opencv.hpp"
#include <sstream>
#include <string>
#include <iostream>

using namespace std;

const int hblocks = 4;
const int wblocks = 8;

cv::Ptr<cv::ml::SVM> model;

uint8_t table[256];

string data_label_name("../datalabel.xml");
string matrix_data_name("../matlab/lbp.txt");


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


int ulbp_feature(cv::Mat src, cv::Mat& dst)
{
	if (src.depth() != 1)
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


// hs*ws is the number of cells; not the pixel number
cv::Mat calcHistogram(cv::Mat src, int hs, int ws, int dim)
{
	int height = src.rows;
	int width = src.cols;
	int maskh = height / hs;
	int maskw = width / ws;

	bool uniform = true;
	bool accumulate = false;
	float range[] = { 0, float(dim) };
	const float* histRange = { range };
	const int channels[] = { 0 };
	const int histSize = dim;

	cv::Mat dst = cv::Mat::zeros(dim, hs*ws, CV_8U);

	for (int i = 0; i < hs; i++)
		for (int j = 0; j < ws; j++)
		{
			cv::Mat hist;
			cv::Mat mask(src, cv::Rect(j*maskw, i*maskh, maskw, maskh));
			cv::calcHist(&mask, 1, 0, cv::Mat(), hist, 1, &histSize, &histRange);

			hist.col(0).copyTo(dst.col(i*ws + j));
		}
	return dst.reshape(0, 1);
}

int main()
{
	cal_table();
	cv::FileStorage fs;
	fs.open(data_label_name,cv::FileStorage::READ);

	cv::FileNode info = fs["info"];
	cv::FileNode data = fs["data"];
	cv::FileNodeIterator it = data.begin();
	cv::FileNodeIterator itend = data.end();

	const int sample_number = (int)info["samples"];

	const int dims = 59;
	const int bins = 42 * 59;

	cv::Mat allData = cv::Mat::zeros(sample_number, bins, CV_32FC1);

  	cout << endl << "reading pictures..." << endl;
	for (int cnt = 0; it!=itend; ++it)
	{
		string path;
		path = (string)(*it)["path"];

		cv::Mat img = cv::imread(path);
		cv::Mat lbp_im;
		cv::Mat Histogram(1, bins, CV_32F);

		cv::resize(img, img, cv::Size(160,80));
		cv::cvtColor(img, img, CV_BGR2GRAY);
		img.convertTo(img, CV_8SC1);	

		ulbp_feature(img, lbp_im);
		calcHistogram(lbp_im, 1, 2, dims).copyTo(Histogram.colRange( 0 * dims,  2 * dims));
		calcHistogram(lbp_im, 2, 4, dims).copyTo(Histogram.colRange( 2 * dims, 10 * dims));
		calcHistogram(lbp_im, 4, 8, dims).copyTo(Histogram.colRange(10 * dims, 42 * dims));

		Histogram.copyTo(allData.row(cnt));

		cout << "\033[1A[  INFO] processing " << ++cnt << " pictures.\n\033[0m"; 
	}

	cout << "[  INFO] lbp done." << endl;
	cout << "[  INFO] writing the data." << endl;

	ofstream fout(matrix_data_name);
	for (int i = 0; i < allData.rows; i++) 
	{
		for (int j = 0; j < allData.cols; j++)
		{
			fout << allData.at<float>(i, j) << " ";
		}
		fout << endl;
	}

	fout.close();
	fs.release();
	return 0;
}