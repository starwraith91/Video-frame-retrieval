#pragma once
#include "stdafx.h"
#include "FeatureExtraction.h"
#include "MachineLearning.h"

Mat ExtractSURFDescriptor(string path)
{
	Mat pImg = imread( path.c_str() );

	SurfFeatureDetector detector;
	vector<KeyPoint> keypoints;
	detector.detect(pImg, keypoints);
	
	// computing descriptors
	Mat descriptors;
	SurfDescriptorExtractor extractor;
	extractor.compute(pImg, keypoints, descriptors);

	return descriptors;
}

Mat ExtractSIFTDescriptor(string path)
{
	Mat pImg = imread(path.c_str());

	SiftFeatureDetector detector;
	vector<KeyPoint> keypoints;
	detector.detect(pImg, keypoints);

	// computing descriptors
	Mat descriptors;
	SiftDescriptorExtractor extractor;
	extractor.compute(pImg, keypoints, descriptors);

	return descriptors;
}

#define _NUM_CLASSES 5

void CreateVocaburary(BOWImgDescriptorExtractor &bowDE, int dictionarySize)
{
	//---dictionary size = number of cluster's centroids
	TermCriteria tc(CV_TERMCRIT_ITER + CV_TERMCRIT_EPS, 10, 0.01);
	int retries = 1;
	int flags = KMEANS_PP_CENTERS;

	//Get a large set of image to be training set
	string trainpath = "Data/Key frames/";

	vector<string> _listClass;
	_listClass.push_back("Gokaiger_ep40");
	_listClass.push_back("Shinkenger_vs_Goseiger");
	_listClass.push_back("Super_Hero_Taisen");

	int numClass = (int)_listClass.size();
	for (int classIndex = 0; classIndex < numClass; classIndex++)
	{
		BOWKMeansTrainer bowTrainer(dictionarySize, tc, retries, flags);

		string path = trainpath + _listClass[classIndex] + "/";
		vector<string> _listStringName = ReadFileList(path.c_str());
		int numFileName = (int)_listStringName.size();
		for (int j = 0; j < numFileName; j++)
		{
			string filename = path + _listStringName[j];

			cout << filename << endl;

			Mat descriptors = ExtractSURFDescriptor(filename);

			if (!descriptors.empty())
			{
				bowTrainer.add(descriptors);
			}
			else
			{
				cout << "This image doesn't contain any feature point" << endl;
			}
		}	

		ClusterFeature(bowTrainer, bowDE, _listClass[classIndex]);
	}

	//ClusterFeature(bowTrainer, bowDE, "");

}

void ClusterFeature(BOWKMeansTrainer bowTrainer, BOWImgDescriptorExtractor &bowDE, string dictionaryName)
{
	vector<Mat> descriptors = bowTrainer.getDescriptors();
	int count = 0;
	for (vector<Mat>::iterator iter = descriptors.begin(); iter != descriptors.end(); iter++)
	{
		count += iter->rows;
	}
	cout << "Clustering " << count << " features" << endl;
	Mat dictionary = bowTrainer.cluster();
	bowDE.setVocabulary(dictionary);

	string filename = "Data/BOW_data/dictionaries/" + dictionaryName + "_cluster.xml";
	FileStorage fs(filename, FileStorage::WRITE);
	fs << "dictionary" << dictionary;
}

Mat LoadBOWDictionaryFromFile(string filename)
{
	Mat dictionary;

	// read
	FileStorage fs(filename, FileStorage::READ);
	fs["dictionary"] >> dictionary;

	return dictionary;
}

void CreateBOWTrainingSet(int dictionarySize, SurfFeatureDetector detector, BOWImgDescriptorExtractor bowDE)
{
	//Get a large set of image to be training set
	string trainpath = "Data/Key frames/";

	string _listClass[_NUM_CLASSES];
	_listClass[0].assign("Gokaiger_ep40");
	_listClass[1].assign("Home_Alone_1");
	_listClass[2].assign("Shinkenger_vs_Goseiger");

	Mat bowDescriptor;

	for (int classIndex = 0; classIndex < 3; classIndex++)
	{
		string vocabularyPath = "Data/BOW_data/dictionaries/" + _listClass[classIndex] + "_cluster.xml";
		bowDE.setVocabulary(LoadBOWDictionaryFromFile(vocabularyPath));

		Mat trainingData(0, dictionarySize, CV_32FC1);

		Mat trainingLabel(0, 1, CV_32FC1);

		string path = trainpath + _listClass[classIndex] + "/";
		vector<string> _listStringName = ReadFileList(path.c_str());

		cout << "There are " << _listStringName.size() << " files in this folder" << endl;

		int numFileName = (int)_listStringName.size();
		for (int filenameIndex = 0; filenameIndex < numFileName; filenameIndex++)
		{
			string filename = path + _listStringName[filenameIndex];

			cout << filename << endl;

			Mat testImage = imread(filename);

			vector<KeyPoint> keypoint;
			detector.detect(testImage, keypoint);

			if (keypoint.size() > 0)
			{
				bowDE.compute(testImage, keypoint, bowDescriptor);

				trainingData.push_back(bowDescriptor);

				int shotID = IdentifyShotFromKeyFrame(_listStringName[filenameIndex]);
				trainingLabel.push_back((float)shotID);
			}
			else
			{
				cout << "No keypoint can be found in this image. Ignore!" << endl;
			}
		}

		cout << "There are " << trainingData.size() << " sample images with " << trainingLabel.size() << " classes." << endl;

		//Write training data to file for future use。	
		string pathTrainData = "Data/BOW_data/descriptors/" + _listClass[classIndex] + ".xml";
		FileStorage fs(pathTrainData, FileStorage::WRITE);
		fs << "training_data_BOW" << trainingData;
		fs << "training_label_BOW" << trainingLabel;
		cout << "Finish creating training data for " << _listClass[classIndex] << endl;

		//Encoding class name values
		string pathMapData = "Data/" + _listClass[classIndex] + "_map.txt";
		map<float, float> mapClassEncode = ClassEncoding(pathMapData,trainingLabel);
		cout << "Finish encoding class name to number for " << _listClass[classIndex] << endl;

		//Initialize ANN for training
		string pathANNModel = "Data/ANN_Model/" + _listClass[classIndex] + ".txt";
		CreateANNTrainingModel(pathANNModel, dictionarySize, mapClassEncode, trainingData, trainingLabel);
		cout << "Finish creating ANN training model for " << _listClass[classIndex] << endl;
	}
}

Mat ExtractBOWFeature(BOWImgDescriptorExtractor bowDE, SurfFeatureDetector detector, Mat image)
{
	Mat bowDescriptor;
	vector<KeyPoint> keypoint;
	detector.detect(image, keypoint);
	bowDE.compute(image, keypoint, bowDescriptor);

	return bowDescriptor;
}

void LoadBOWTrainingSet(string path, Mat &training_data, Mat &training_label)
{
	// read:
	FileStorage fs(path, FileStorage::READ);
	fs["training_data_BOW"] >> training_data;
	fs["training_label_BOW"] >> training_label;
}

//---------BASIC IMAGE PROCESSING-----------//

Mat EdgeDetection(Mat img)
{
	Mat edgeImg; 
	cvtColor(img, edgeImg, CV_BGR2GRAY);
	Sobel(edgeImg, edgeImg, CV_8U, 1, 1);

	return edgeImg;
}

int CountEdgePixel(Mat imgEdge)
{
	int count = 0;
	int imageSize = imgEdge.rows * imgEdge.cols;
	for (int i = 0; i < imageSize; i++)
	{
		if (imgEdge.at<uchar>(i) > 0)
		{
			count++;
		}
	}

	return count;
}

double CalculateEdgeMatchingRate(Mat imgEdge1, Mat imgEdge2)
{
	double matchingRate = 0;

	//Count number of matched edge pixel
	int countMatchPixel = 0;
	int imageSize = imgEdge1.rows * imgEdge1.cols;
	for (int i = 0; i < imageSize; i++)
	{
		uchar pixel1 = imgEdge1.at<uchar>(i);
		uchar pixel2 = imgEdge2.at<uchar>(i);
		if (pixel1 == pixel2)
		{
			countMatchPixel++;
		}
	}

	//int edgePixel = MAX( CountEdgePixel(imgEdge1) , CountEdgePixel(imgEdge2) );
	//if (edgePixel > 0)
	//{
		matchingRate = (double)countMatchPixel / (double)imageSize;
	//}

	return matchingRate;
}