#include <opencv2/opencv.hpp>
#include <string>
#include <fstream>
#include <iostream>


using namespace std;

cv::Ptr<cv::ml::SVM> model;

uint8_t table[256];

string classifier_name("../models/test/svm_classifier_model.xml");
// string pca_faeture_name("../models/test/PCA_Feature.txt");
string pca_faeture_name("../matlab/PCA_Feature.txt");

string data_label_name("../datalabel.xml");

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


void svmTrain(cv::Mat src, cv::Mat labelsMat)
{
	model=cv::ml::SVM::create();
	model->setType(cv::ml::SVM::C_SVC);
		model->setKernel(cv::ml::SVM::LINEAR);	
		// model->setKernel(cv::ml::SVM::POLY);
		// model->setKernel(cv::ml::SVM::RBF);
		// model->setKernel(cv::ml::SVM::SIGMOID);
		// model->setKernel(cv::ml::SVM::INTER);
	
	model->setC(1e-12);		// (C_SVC/EPS_SVR/NU_SVR)
	model->setGamma(1e-6);	// (POLY/RBF/SIGMOID)
		//model->setP(1);			// (EPS_SVR)
	model->setNu(0.7);		// (NU_SVC/ONE_CLASS/NU_SVR)
	model->setCoef0(0.1);	// (POLY/SIGMOID)
	model->setDegree(4);	// (POLY)
	
	vector<float> weights;
	weights.push_back(0.7);
	weights.push_back(0.3);
	cv::Mat M(weights);
	model->setClassWeights(M);
	model->setTermCriteria(cv::TermCriteria(cv::TermCriteria::EPS+cv::TermCriteria::MAX_ITER, 5000, 5e-5));
	
	cv::Ptr<cv::ml::TrainData> tData;
	tData = cv::ml::TrainData::create(src, cv::ml::ROW_SAMPLE, labelsMat);

	cv::ml::ParamGrid Cgrid 	 = cv::ml::ParamGrid::ParamGrid(1e-12,1,5);		// (C_SVC/EPS_SVR/NU_SVR)
	cv::ml::ParamGrid gammaGrid  = cv::ml::ParamGrid::ParamGrid(1e-6,1e-2,10);	// (POLY/RBF/SIGMOID)
	cv::ml::ParamGrid pGrid 	 = cv::ml::ParamGrid::ParamGrid(1.0,1.0,1);		// (EPS_SVR)
	cv::ml::ParamGrid nuGrid	 = cv::ml::ParamGrid::ParamGrid(0.1,1.0,1);		// (NU_SVC/ONE_CLASS/NU_SVR)
	cv::ml::ParamGrid coeffGrid	 = cv::ml::ParamGrid::ParamGrid(0.5,0.5,1); 	// (POLY/SIGMOID)
	cv::ml::ParamGrid degreeGrid = cv::ml::ParamGrid::ParamGrid(4,4,1); 		// (POLY)

	// model->train(tData);
	model->trainAuto(tData, 10, Cgrid, gammaGrid, pGrid, nuGrid, coeffGrid, degreeGrid, true);
	model->save(classifier_name);
}


int main()
{
	cal_table();

	cv::FileStorage fs;

	fs.open(data_label_name,cv::FileStorage::READ);

	cv::FileNode info = fs["info"];
	cv::FileNode data = fs["data"];
	cv::FileNodeIterator it = data.begin();
	const cv::FileNodeIterator itend = data.end();
	const cv::FileNodeIterator it_ = data.begin();
	
	const int sample_number = (int)info["samples"];

	int r;
	int correct_number = 0;
	const int dims = 59;
	const int bins = 42 * 59;

	const int ratio = 2;
	const int training_number = sample_number / ratio;
	const int testing_number = sample_number - training_number;

	cv::Mat mean_X(1, bins, CV_32F);

	cout << "[  INFO] loading pca data..." << endl;

	ifstream fin(pca_faeture_name);

	for (int i = 0; i < bins; i++)
		fin >> mean_X.at<float>(0, i);
	fin >> r;
	cv::Mat eigenvectors(bins, r, CV_32F);
	for (int i = 0; i < r; i++)
		for (int j = 0; j < bins; j++)
			fin >> eigenvectors.at<float>(j, i);

	fin.close();

	cv::Mat dataset = cv::Mat::zeros(training_number, r, CV_32FC1);
	cv::Mat labels  = cv::Mat::zeros(training_number, 1, CV_32SC1);
	cv::Mat dataset_ = cv::Mat::zeros(testing_number, r, CV_32FC1);
	cv::Mat labels_  = cv::Mat::zeros(testing_number, 1, CV_32SC1);

	int cnt1 = 0;
	int cnt2 = 0;

	cout << endl << "[  INFO] reading pictures...\n\n" << endl;

	for (int cnt=0; it!=itend; ++it)
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

		Histogram.convertTo(Histogram, CV_32F);
		for (int i = 0; i < Histogram.cols; i++)
			Histogram.at<float>(0, i) -= mean_X.at<float>(0, i);
		Histogram = Histogram * eigenvectors;

		if(cnt%ratio==0)
		{
			Histogram.row(0).copyTo(dataset.row(cnt1));
			labels.at<int>(cnt1++,0) = (int)(*it)["label"];
		}
		else
		{
			Histogram.row(0).copyTo(dataset_.row(cnt2));
			labels_.at<int>(cnt2++,0) = (int)(*it)["label"];
		}
		
		
		cout << "\033[1A[  INFO] precessing " << ++cnt << " pictures.\033[0m" << endl;
	}

	cout << "[  INFO] lbp done." << endl;

	cout << endl << "[  INFO] svm training...";
	svmTrain(dataset, labels);
	cout << endl << "[  INFO] svm training done.\n" << endl;

	cout << "[  INFO] predict results:" << endl;

	for (int cnt = 0; cnt < testing_number; cnt++) 
	{
		int label = (model->predict(dataset_.row(cnt)) == 1) ? 1 : 0;
		if (label == labels_.at<int>(cnt, 0)) 
		{
			// cout << "\033[1A[RESULT] the picture No." << i+1 << "\tis: ";
			// if (label == 0) cout << "open   "; else cout << "closed ";
			// cout << "| correct" << "\033[0m" << endl;
			correct_number++;
		}
		else 
		{
			// int idx = (cnt/(ratio-1))*ratio + (cnt%(ratio-1));
			// it = it_;
			// it+=cnt;
			// cout << "\033[31m[RESULT] the picture No." << cnt+1 << "\tis: ";
			// if (label == 0) cout << "open  "; else cout << "closed";
			// cout << " | incorrect -> ";
			// cout << (string)(*it)["path"] << "\033[0m" << endl;
			// cout << "[  INFO] the 436d vector is: \n" << dataset.row(cnt) << endl;
		}
	}

	cout << "[  INFO] " << training_number << " pictures were trained, ";
	cout << testing_number << " pictures were predicted" << endl;
	cout << "[  INFO] " << correct_number << " predictions are correct, ";
	cout << testing_number - correct_number << " predictions are incorrect" << endl;
	cout << "[  INFO] correct rate is " << (double)correct_number / testing_number << endl;

	cout << "\n[  INFO] the fine-tuning paramers are: " << endl;
	cout << "[  PARA] C: " << model->getC() << endl;
	cout << "[  PARA] Gamma: " << model->getGamma() << endl;
	cout << "[  PARA] P: " << model->getP() << endl;
	cout << "[  PARA] Nu: " << model->getNu() << endl;
	cout << "[  PARA] Coef0: " << model->getCoef0() << endl;
	cout << "[  PARA] Degree: " << model->getDegree() << endl;

	cv::destroyAllWindows();
	fs.release();

	cout << "a";
	return 0;
}
