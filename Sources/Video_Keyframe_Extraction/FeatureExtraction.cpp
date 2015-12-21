#pragma once
#include "stdafx.h"
#include "FeatureExtraction.h"

Mat ExtractSURFDescriptor(string path)
{
	Mat pImage = imread(path.c_str());

	SurfFeatureDetector detector;
	vector<KeyPoint> keypoints;
	detector.detect(pImage, keypoints);
	
	// computing descriptors
	Mat descriptors;
	SurfDescriptorExtractor extractor;
	extractor.compute(pImage, keypoints, descriptors);

	return descriptors;
}

Mat ExtractSIFTDescriptor(string path)
{
	Mat pImage = imread(path.c_str());

	SiftFeatureDetector detector;
	vector<KeyPoint> keypoints;
	detector.detect(pImage, keypoints);

	// computing descriptors
	Mat descriptors;
	SiftDescriptorExtractor extractor;
	extractor.compute(pImage, keypoints, descriptors);

	return descriptors;
}

int numClassVideo = 0;
void CreateVocaburary(string categoryName, BOWImgDescriptorExtractor &bowDE, int dictionarySize)
{
	//---dictionary size = number of cluster's centroids
	TermCriteria tc(CV_TERMCRIT_ITER + CV_TERMCRIT_EPS, 10, 0.01);
	int retries = 1;
	int flags = KMEANS_PP_CENTERS;

	//Get a large set of images to be training set
	string trainpath = "Data/Key frames/" + categoryName + "/";
	vector<string> _listClass = ReadFileList(trainpath);

	int numClass = (int)_listClass.size();
	BOWKMeansTrainer bowTrainer(dictionarySize, tc, retries, flags);
	for (int classIndex = 0; classIndex < numClass; classIndex++)
	{
		string path = trainpath + _listClass[classIndex] + "/";
		vector<string> _listStringName = ReadFileList(path.c_str());
		Shuffle(&_listStringName[0], _listStringName.size());

		int numFileName = numClass / 10;
		for (int j = 0; j < numFileName; j++)
		{
			string filename = path + _listStringName[j];

			cout << filename << endl;

			//Mat descriptors = ExtractSURFDescriptor(filename);
			Mat descriptors = ExtractSIFTDescriptor(filename);

			if (!descriptors.empty())
			{
				bowTrainer.add(descriptors);
			}
			else
			{
				cout << "This image doesn't contain any feature point" << endl;
			}
		}	
	}
	ClusterFeature(categoryName, bowTrainer, bowDE, "");
}

void ClusterFeature(string categoryName, BOWKMeansTrainer bowTrainer, BOWImgDescriptorExtractor &bowDE, string dictionaryName)
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

	string filename = "Data/Training_Data/" + categoryName + "_BOW_dictionary.xml";
	FileStorage fs(filename, FileStorage::WRITE);
	fs << "dictionary" << dictionary;
}

Mat LoadBOWDictionaryFromFile(string filename)
{
	Mat dictionary;

	// read
	FileStorage fs(filename, FileStorage::READ);
	fs["dictionary"] >> dictionary;

	fs.release();

	return dictionary;
}

void CreateBOWTrainingSet(string categoryName, int dictionarySize, SiftFeatureDetector detector, BOWImgDescriptorExtractor bowDE)
{
	//Get a large set of image to be training set
	string trainpath = "Data/Keyframes/" + categoryName + "/";

	vector<string> _listClass = ReadFileList(trainpath);
	numClassVideo = _listClass.size();

	Mat bowDescriptor;

	string vocabularyPath = "Data/Training_Data/" + categoryName + "_BOW_dictionary.xml";
	Mat dictionary = LoadBOWDictionaryFromFile(vocabularyPath);
	if (dictionary.empty())
	{
		CreateVocaburary(categoryName, bowDE, dictionarySize);
	}
	else
	{
		bowDE.setVocabulary(dictionary);
	}	

	Mat trainingData(0, dictionarySize, CV_32FC1);
	Mat trainingLabel(0, 1, CV_32SC1);

	//Get list of raw video to get the fps to store
	string pathRawData = "Data/Raws/" + categoryName + "/";
	vector<string> _listRawVideos = ReadFileList(pathRawData);

	string pathTrainData = "Data/Training_Data/" + categoryName + "/BOW/";
	for (int classIndex = 0; classIndex < numClassVideo; classIndex++)
	{
		//Check for data existence
		FileStorage fs(pathTrainData + _listClass[classIndex] + ".xml", FileStorage::READ);
		if (fs.isOpened())
		{
			cout << "Training data for " << _listClass[classIndex] << " is already exist" << endl;
			fs.release();
			continue;
		}

		string path = trainpath + _listClass[classIndex] + "/";
		vector<string> _listStringName = ReadFileList(path.c_str());

		cout << "There are " << _listStringName.size() << " files in folder " << _listClass[classIndex] << endl;

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
				int startID = IdentifyStartIDFromKeyFrame(_listStringName[filenameIndex]);
				int keyID = IdentifyKeyIDFromKeyFrame(_listStringName[filenameIndex]);
				Mat tempMat(1, 3, CV_32SC1);
				tempMat.at<int>(0, 0) = shotID;
				tempMat.at<int>(0, 1) = startID;
				tempMat.at<int>(0, 2) = keyID;
				trainingLabel.push_back(tempMat);
			}
			else
			{
				cout << "No keypoint can be found in this image. Ignore!" << endl;
			}
		}

		cout << "There are " << trainingData.size() << " sample images with " << trainingLabel.size() << " classes." << endl;

		//Write training data to file for future use	
		if (CreateDirectoryA(pathTrainData.c_str(), NULL) || ERROR_ALREADY_EXISTS == GetLastError())
		{
			FileStorage fs(pathTrainData + _listClass[classIndex] + ".xml", FileStorage::WRITE);
			fs << "data_matrix" << trainingData;
			fs << "data_label" << trainingLabel;

			VideoCapture cap(pathRawData + _listRawVideos[classIndex]);
			float fps = (float)cap.get(CV_CAP_PROP_FPS);
			fs << "video_fps" << fps;
		}
		else
		{
			cout << "Cannot find and create " << pathTrainData << " directory" << endl;
		}

		trainingData.release();

		trainingLabel.release();
	}

	////Encoding class name values
	//string pathMapData = "Data/ANN_map.txt";
	//map<float, float> mapClassEncode = ClassEncoding(pathMapData, trainingLabel);
	//cout << "Finish encoding class name " << endl;

	//Initialize ANN for training
	//string pathANNModel = "Data/ANN_Model.txt";
	//CreateANNTrainingModel(pathANNModel, dictionarySize, trainingData, trainingLabel, numClassVideo);
	//cout << "Finish creating ANN training model"<< endl;
}

void CreateMPEGTrainingSet(string categoryName)
{
	//Get a large set of image to be training set
	string trainpath = "Data/Keyframes/" + categoryName + "/";
	vector<string> _listClass = ReadFileList(trainpath);
	numClassVideo = _listClass.size();

	Mat trainingData(0, 1, CV_32FC1);
	Mat trainingLabel(0, 3, CV_32SC1);

	//Get list of raw video to get the fps to store
	string pathRawData = "Data/Raws/" + categoryName + "/";
	vector<string> _listRawVideos = ReadFileList(pathRawData);

	//For each keyframe in each video, extract MPEG7 feature then store all of them in a XML file 
	//Beside feature vector, XML file also store shotID, startFrameID, keyframeID and video FPS
	string pathTrainData = "Data/Training_data/" + categoryName + "/MPEG7/";
	for (int classIndex = 0; classIndex < numClassVideo; classIndex++)
	{
		//Check for data existence	
		FileStorage fsR(pathTrainData + _listClass[classIndex] + ".xml", FileStorage::READ);
		if (fsR.isOpened())
		{
			cout << "Training data for " << _listClass[classIndex] << " is already exist" << endl;
			fsR.release();
			continue;
		}

		string path = trainpath + _listClass[classIndex] + "/";
		vector<string> _listStringName = ReadFileList(path.c_str());

		cout << "There are " << _listStringName.size() << " files in folder " << _listClass[classIndex] << endl;

		int numFileName = (int)_listStringName.size();
		for (int filenameIndex = 0; filenameIndex < numFileName; filenameIndex++)
		{
			string filename = path + _listStringName[filenameIndex];
			cout << filename << endl;

			Mat testImage  = imread(filename);
			Mat descriptor = ExtractMPEGFeature(testImage);

			trainingData.push_back(descriptor);

			int shotID   = IdentifyShotFromKeyFrame(_listStringName[filenameIndex]);
			int startID  = IdentifyStartIDFromKeyFrame(_listStringName[filenameIndex]);
			int keyID	 = IdentifyKeyIDFromKeyFrame(_listStringName[filenameIndex]);
			Mat tempMat(1, 3, CV_32SC1);
			tempMat.at<int>(0, 0) = shotID;
			tempMat.at<int>(0, 1) = startID;
			tempMat.at<int>(0, 2) = keyID;
			trainingLabel.push_back(tempMat);
		}

		cout << "There are " << trainingData.size() << " sample images with " << trainingLabel.size() << " classes." << endl;

		//Write training data to file for future use	
		
		if (CreateDirectoryA(pathTrainData.c_str(), NULL) || ERROR_ALREADY_EXISTS == GetLastError())
		{
			FileStorage fsW(pathTrainData + _listClass[classIndex] + ".xml", FileStorage::WRITE);
			fsW << "data_matrix" << trainingData;
			fsW << "data_label" << trainingLabel;

			VideoCapture cap(pathRawData + _listRawVideos[classIndex]);
			float fps = (float)cap.get(CV_CAP_PROP_FPS);
			fsW << "video_fps"  << fps;
			fsW << "video_name" << _listClass[classIndex];

			fsW.release();
		}
		else
		{
			cout << "Cannot find and create " << pathTrainData << " directory" << endl;
		}

		trainingData.release();

		trainingLabel.release();
	}
}

Mat ExtractBOWFeature(BOWImgDescriptorExtractor bowDE, SiftFeatureDetector detector, Mat image)
{
	Mat bowDescriptor;
	vector<KeyPoint> keypoint;
	detector.detect(image, keypoint);
	bowDE.compute(image, keypoint, bowDescriptor);

	return bowDescriptor;
}

Mat ExtractMPEGFeature(Mat pImage)
{
	vector<float> featureVector;

	Feature *featureExtractor = new Feature();
	Frame *frame = new Frame(pImage);

	//Extract Scalable Color Descriptor
	ScalableColorDescriptor *scalable = featureExtractor->getScalableColorD(frame,true,64);
	for (int i = 0; i < scalable->m_NumberOfCoefficients; i++)
	{
		featureVector.push_back(scalable->m_ScalableHistogram[i]);
	}
	delete scalable;

	//Extract Color Layout Descriptor
	int numYCoef = 10;
	int numCCoef = 6;
	ColorLayoutDescriptor *layout = featureExtractor->getColorLayoutD(frame, numYCoef, numCCoef);
	for (int i = 0; i < numYCoef; i++)
	{
		featureVector.push_back((float)layout->m_y_coeff[i]);
	}
	for (int i = 0; i < numCCoef; i++)
	{
		featureVector.push_back((float)layout->m_cb_coeff[i]);
		featureVector.push_back((float)layout->m_cr_coeff[i]);
	}
	float magnitude = GetMagnitude(featureVector);
	for (int i = 0; i < featureVector.size(); i++)
	{
		if (magnitude>0)
			featureVector[i] /= magnitude;
		else
			featureVector[i] = 0;
	}
	delete layout;

	//Extract Edge Histogram Descriptor
	EdgeHistogramDescriptor *edge = featureExtractor->getEdgeHistogramD(frame);
	for (int i = 0; i < edge->GetSize(); i++)
	{
		featureVector.push_back( (float)edge->GetEdgeHistogramD()[i] );
	}
	delete edge;

	delete frame;

	delete featureExtractor;

	return ToMat(featureVector);
}

bool LoadDataFromFile(string path, Mat &training_data, Mat &training_label, float &videoFPS)
{
	// read:
	FileStorage fs(path, FileStorage::READ);

	if (fs.isOpened())
	{
		fs["data_matrix"] >> training_data;
		fs["data_label"] >> training_label;
		fs["video_fps"] >> videoFPS;
		fs.release();
		return true;
	}
	else
	{
		return false;
	}
}

//---------BASIC IMAGE PROCESSING-----------//

KeyFrameDescriptor CalcMPEGDescriptor(Mat img)
{
	KeyFrameDescriptor descriptor;

	Feature *featureExtractor = new Feature();

	Frame *frame = new Frame(img);

	descriptor.colorDesc = featureExtractor->getColorStructureD(frame, 64);
	descriptor.edgeDesc = featureExtractor->getEdgeHistogramD(frame);

	Mat grayImg(img.rows, img.cols, CV_8U);
	cvtColor(img, grayImg, CV_BGR2GRAY);
	frame->setGray(grayImg);
	descriptor.textureDesc = featureExtractor->getHomogeneousTextureD(frame);

	delete frame;

	delete featureExtractor;

	return descriptor;
}
