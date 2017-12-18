#include <opencv2/opencv.hpp>
#include <cstdlib>
#include <sstream>
#include <string>

using namespace std;


const int hblocks = 4;
const int wblocks = 8;

cv::Ptr<cv::ml::SVM> svm;

string classifier_name("../models/test/svm_classifier_model.xml");
string pca_faeture_name("../models/test/PCA_Feature.txt");

uint8_t table[256];


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
cv::Mat clHistogram(cv::Mat src, int hs, int ws, int dim)
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

int main(int argc, char * argv[])
{
	// check if the image is given?
	cv::Mat img = cv::imread(argv[1]);
	if(!img.data)
	{
		cout << "[ Error] Failed to laod the image.\n";
		return -1;
	}

	cv::namedWindow("Image", CV_WINDOW_AUTOSIZE);

	cal_table();
	int r;
	const int dims = 59;
	const int bins = 42 * 59;

	svm = cv::ml::SVM::load(classifier_name);

	cv::Mat mean_X(1, bins, CV_32F);

	ifstream fin(pca_faeture_name);
	int cor= 0;
	for (int i=0; i<bins; i++)
		fin >> mean_X.at<float>(0,i);
	fin >> r;

	cout << r << endl;

	cv::Mat eigenvectors(bins, r, CV_32F);
	for (int i=0; i<r; i++)
		for (int j=0; j<bins; j++)
			fin >> eigenvectors.at<float>(j,i);
	fin.close();

	cv::Mat lbp_img;
	cv::Mat Histogram(1, bins, CV_32F);

	cv::resize(img, img, cv::Size(160,80));
	cv::cvtColor(img, img, CV_BGR2GRAY);
	img.convertTo(img, CV_8SC1);
	//img.resize(25, 45);
	cv::imshow("Image", img);

	ulbp_feature(img, lbp_img);
	clHistogram(lbp_img, 1, 2, dims).copyTo(Histogram.colRange( 0*dims,  2*dims));
	clHistogram(lbp_img, 2, 4, dims).copyTo(Histogram.colRange( 2*dims, 10*dims));
	clHistogram(lbp_img, 4, 8, dims).copyTo(Histogram.colRange(10*dims, 42*dims));

	Histogram.convertTo(Histogram, CV_32F);
	for (int i=0; i<Histogram.cols; i++)
		Histogram.at<float>(0,i) -= mean_X.at<float>(0,i);

	Histogram = Histogram * eigenvectors;
			
	int closedoropen = svm -> predict(Histogram);
			
	if(closedoropen==1)
	{
		cout << "open" << endl;
	}
	else
	{
		cout << "close" << endl;
	}

	//cv::waitKey(0);
	cv::destroyAllWindows();

	return 0;
}






