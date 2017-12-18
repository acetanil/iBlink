#include <opencv2/opencv.hpp>
#include <string>
#include <fstream>
#include <iostream>


using namespace std;

cv::Ptr<cv::ml::SVM> svm;

uint8_t table[256];

string classifier_name("../models/test/svm_classifier_model.xml");
// string pca_faeture_name("../models/test/PCA_Feature.txt");
string pca_faeture_name("../matlab/PCA_Feature.txt");


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

void cvtColor_BGR2HS(cv::Mat src, cv::Mat& dst)
{
	assert(src.channels()==3);
	const int cols = src.cols;
	const int rows = src.rows;
	const float pi = 3.14159;

	dst = cv::Mat::zeros(rows, cols, CV_8UC1);

	cv::Mat_<cv::Vec3b>::iterator it = src.begin<cv::Vec3b>();
	cv::Mat_<cv::Vec3b>::iterator itend = src.end<cv::Vec3b>();

	cv::Mat_<char>::iterator it_ = dst.begin<char>();
	cv::Mat_<char>::iterator itend_ = dst.end<char>();

	int r, g, b;
	float h, s, i, all;

	for(;it!=itend;++it)
	{
		cv::Vec3b rgb = *it;
		b = rgb[0];
		g = rgb[1];
		r = rgb[2];

		int min_val = std::min(b, std::min(g,r));
		int sum_val = b + g + r;

		i = sum_val/(3.0*255);
		s = 1 - (3.0/(sum_val+0.0001))*min_val;
		h = 0.5*((r-g)+(r-b))/(sqrt((r-g)*(r-g)+(r-b)*(g-b))+0.0001);
		h = acos(h)/(2*pi);
		h = g<=b?h:1-h;

		all = i*0.2 + h*0.5 + s;
		// cout << i << '\t' << s << '\t' << h << '\t' << all << endl;

		*(it_++) = char(all*255);
	}

	assert(it_==itend_);
	return;
}

int main(int argc, char * argv[])
{
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
	//cv::imshow("Camera", img);

	cal_table();

	int r;
	const int dims = 59;
	const int bins = 42 * 59;

	int x, y, h, w;

	svm = cv::ml::SVM::load(classifier_name);

	cv::Mat mean_X(1, bins, CV_32F);

	cout << "A\n";

	ifstream fin(pca_faeture_name);
	int cor= 0;
	for (int i=0; i<bins; i++)
		fin >> mean_X.at<float>(0,i);

	fin >> r;

	cv::Mat eigenvectors(bins, r, CV_32F);
	for (int i=0; i<r; i++)
		for (int j=0; j<bins; j++)
			fin >> eigenvectors.at<float>(j,i);

	fin.close();

	cout << "B\n";

	fin.open("../bounding_box.txt");
	fin >> x;
	fin >> y;
	fin >> w;
	fin >> h;
	fin.close();

	const cv::Rect bounding_box = cv::Rect(x, y, w, h);

	cv::Mat lbp_img, img_eye;

	cout << "begin" << endl;

	while(1)
	{
		cam >> img;
		if(!img.data) continue;

		img_eye=img(bounding_box);
		cv::resize(img_eye,img_eye,cv::Size(160,80));
		cv::cvtColor(img_eye, img_eye, CV_BGR2GRAY);
		// cvtColor_BGR2HS(img_eye, img_eye);
		img_eye.convertTo(img_eye, CV_8SC1);

		cv::imshow("Camera", img_eye);

		cv::Mat Histogram(1, bins, CV_32F);

		ulbp_feature(img_eye, lbp_img);
		calcHistogram(lbp_img, 1, 2, dims).copyTo(Histogram.colRange( 0*dims,  2*dims));
		calcHistogram(lbp_img, 2, 4, dims).copyTo(Histogram.colRange( 2*dims, 10*dims));
		calcHistogram(lbp_img, 4, 8, dims).copyTo(Histogram.colRange(10*dims, 42*dims));

		Histogram.convertTo(Histogram, CV_32F);
		for (int i=0; i<Histogram.cols; i++)
			Histogram.at<float>(0,i) -= mean_X.at<float>(0,i);
		Histogram = Histogram * eigenvectors;

		int closedoropen = svm -> predict(Histogram);
			
		if(closedoropen==1) cout << "closed" << endl;
		else cout << "open" << endl;

		char c = cv::waitKey(20);
		if(c=='q') break;
	}

	cv::destroyAllWindows();
	return 0;
}






