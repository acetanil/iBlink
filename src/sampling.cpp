#include <opencv2/opencv.hpp>
#include <fstream>

using namespace std;


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
		cout << "[ Error] Failed to gather the capture.\n";
		return -1;
	}

	cv::Mat img;
	cam >> img;
	const int cols = img.cols;
	const int rows = img.rows;

	cout << "[  Info] Size: " << rows << 'x' << cols << endl;
	cv::namedWindow("Camera", CV_WINDOW_AUTOSIZE);

	cv::imshow("Camera", img);
	cv::Mat img_;
	cv::resize(img, img_, cv::Size(cols/2, rows/2));

	int x, y, w, h;

	ifstream fin;

	fin.open("../bounding_box.txt");
	fin >> x;
	fin >> y;
	fin >> w;
	fin >> h;
	fin.close();

	const cv::Rect bounding_box = cv::Rect(x, y, w, h);
	const cv::Rect bounding_box_ = cv::Rect(x/2, y/2, w/2, h/2);


	bool flag_ongoing = true;
	bool sampling_open = false;
	bool sampling_close = false;

	cv::Mat img_eye, img_eye_;
	cv::namedWindow("Eye Samples", CV_WINDOW_AUTOSIZE);

	char img_name[50];
	unsigned int img_num = 0;
	int num_open = 500;
	int num_close = 500;

	cout << "[  Info] begin.\n";

	while(flag_ongoing)
	{
		cam >> img;
		if(!img.data) continue;

		cv::resize(img, img_, cv::Size(cols/2, rows/2));
		cv::rectangle(img_, bounding_box_, cv::Scalar(0, 255, 255), 2);

		cv::imshow("Camera", img_);
		char cmd = cv::waitKey(20);
		switch(cmd)
		{
			case 'q': 
			{
				flag_ongoing = false;
				sampling_open = false;
				sampling_close = false;
				break;
			}
			case 'o': 
			{
				sampling_open = true;
				sampling_close = !sampling_open;
				break;
			}
			case 'c':
			{
				sampling_close = true;
				sampling_open = !sampling_close;
				break;
			}
			default: break;
		}

		if(sampling_close||sampling_open)
		{
			img_eye = img(bounding_box);
			cv::resize(img_eye, img_eye, cv::Size(160,80));
			
			cv::cvtColor(img_eye, img_eye_, CV_BGR2GRAY);
			// cvtColor_BGR2HS(img_eye, img_eye_);
			img_eye_.convertTo(img_eye_, CV_8SC1);
			cv::imshow("Eye Samples", img_eye_);
			
			if(sampling_open&&num_open>0)
			{
				sprintf(img_name, "../dataset/samples0/0/img_seq_%04u.png", img_num);
				cv::imwrite(img_name, img_eye_);

				num_open--;
				img_num++;
			}
			
			if(sampling_close&&num_close>0)
			{
				sprintf(img_name, "../dataset/samples0/1/img_seq_%04u.png", img_num);
				cv::imwrite(img_name, img_eye_);

				num_close--;
				img_num++;
			}

			cout << "\033[1A[  Info] Writing\t" << img_num << " images.\n\033[0m";
		}

		if(num_open<=0&&num_close<=0) flag_ongoing = false;
	}


	cout << "[  Info] Processed\t" << img_num << " images.\n";

	cv::waitKey(2500);
	cv::destroyAllWindows();
	return 0;
}







