#include "stdafx.h"
#include "VideoShotRetrieval.h"

extern int dictionarySize;
extern SiftFeatureDetector detector;
extern BOWImgDescriptorExtractor bowDE;

vector<string> SetupModel(string categoryName, vector<Mat> &trainData, vector<Mat> &trainLabels, int database_type)
{
	cout << "Loading training data..." << endl;

	//Get data stored in database
	string path;
	if (database_type == 1)
		path = "Data/Training_Data/" + categoryName + "/BOW/";
	else
		path = "Data/Training_Data/" + categoryName + "/MPEG7/";

	vector<string> _listTrainingData = ReadFileList(path);

	for (int i = 0; i < (int)_listTrainingData.size(); i++)
	{
		string pathData = path + _listTrainingData[i];
		Mat trainingData, trainingLabel;
		LoadBOWTrainingSet(pathData, trainingData, trainingLabel);

		trainData.push_back(trainingData);
		trainLabels.push_back(trainingLabel);
	}

	return _listTrainingData;
}

float CalcDistanceToFeatureLine(Mat f1, Mat f2, Mat fx)
{
	Mat a = fx - f1;
	Mat b = f2 - f1;
	double numerator   = a.dot(b);
	double denominator = b.dot(b);
	
	float muy = (float)(numerator / denominator);
	Mat   px = f1 + muy*(f2 - f1);
	return CalcVectorMagnitude((fx - px));
}

bool FindNextShot(int &startIndex, Mat &subFeatureMatrix, Mat &subFeatureLabel, Mat trainingData, Mat trainingLabel)
{
	if (!subFeatureMatrix.empty())
		subFeatureMatrix.release();
	if (!subFeatureLabel.empty())
		subFeatureLabel.release();

	int nextID = -1;
	int ID = trainingLabel.at<int>(startIndex, 0);
	subFeatureMatrix.push_back(trainingData.row(startIndex));
	subFeatureLabel.push_back(trainingLabel.row(startIndex));

	for (int i = startIndex+1; i < trainingData.rows; i++)
	{
		nextID = trainingLabel.at<int>(i, 0);
		startIndex = i;
		if (nextID == ID)
		{
			subFeatureMatrix.push_back(trainingData.row(i));
			subFeatureLabel.push_back(trainingLabel.row(i));
		}
		else
		{			
			break;
		}
	}

	if (nextID == -1)
		return false;
	return true;
}

vector<int> VideoShotRetrieval(string videoName, Mat trainingData, Mat trainingLabel, Mat queryFeature)
{
	vector<float> _listShotDistance;
	vector<int>	  _listIndex;
	vector<int>   _listShotID;
	vector<int>   _listKeyID;

	int startIndex = 0,prevIndex = 0;
	Mat featureMat,labelMat;
	bool haveMore = true;
	do
	{
		//Push back all rows feature of the same shot into a temporary matrix and process them in there
		int currentShotID = trainingLabel.at<int>(startIndex, 0);
		haveMore = FindNextShot(startIndex, featureMat, labelMat, trainingData, trainingLabel);
		
		//If shot only has 1 keyframe, calculate distance to that frame directly
		float minDistance = -1;
		float minDistLabel = -1;
		if (featureMat.rows == 1)
		{
			Mat f2 = featureMat.row(0);
			minDistLabel = 0;
			minDistance = CalcEuclideanDistance(f2, queryFeature);
		}
		else //If not, calculate distance using NFL method
		{
			Mat f1 = featureMat.row(0);
			for (int i = 1; i < featureMat.rows; i++)
			{
				Mat f2 = featureMat.row(i);
				float distance = CalcDistanceToFeatureLine(f1, f2, queryFeature);
				if (minDistance == -1 || minDistance > distance)
				{
					minDistance = distance;
					minDistLabel = i - 1;
				}
				f1 = f2;
			}
		}

		//Update list of distance and shot ID
		_listShotDistance.push_back(minDistance);
		_listShotID.push_back(currentShotID);

		if (featureMat.rows > 1)
		{
			float distance1 = CalcEuclideanDistance(queryFeature, featureMat.row(minDistLabel));
			float distance2 = CalcEuclideanDistance(queryFeature, featureMat.row(minDistLabel + 1));
			if (distance2 < distance1)
				minDistLabel++;
		}
		int keyID = labelMat.at<int>(minDistLabel, 1) + labelMat.at<int>(minDistLabel, 2);
		_listKeyID.push_back(keyID);

		prevIndex = startIndex;
	} 
	while (haveMore);

	//Sort list of distance and adjust list shot ID along
	for (int i = 0; i < _listShotDistance.size(); i++)
		_listIndex.push_back(i);
	Sort(_listShotDistance, _listIndex, true);

	VideoCapture cap(videoName);
	float fps = cap.get(CV_CAP_PROP_FPS);

	for (int i = 0; i < 5; i++)
	{
		char numstr[21];
		float seconds = _listKeyID[_listIndex[i]] / fps;
		float hh = seconds / 3600;
		int mm = int(seconds / 60) % 60;
		int ss = int(seconds) % 60;
		string time = _itoa(hh, numstr, 10);
		time += ":";
		time += _itoa(mm, numstr, 10);
		time += ":";
		time += _itoa(ss, numstr, 10);
		cout << i + 1 << ". shot " << _listShotID[_listIndex[i]] << " at " << time << " with distance = " << _listShotDistance[i] << endl;
	}
	
	return _listShotID;
}

vector<int> TestFrameRetrieval(int numClass, Mat queryFeature, vector<Mat> trainData)
{
	//---No feature? Stop all action
	vector<int> _listCandidateVideoID;
	if (queryFeature.empty())
		return _listCandidateVideoID;

	//---Retrieve videos that may contain query image
	vector<float> _listDistance;
	vector<int> _listClassID;
	for (int i = 0; i < numClass; i++)
	{
		float distance = CalcDistanceFromSet(queryFeature, trainData[i]);
		_listDistance.push_back(distance);
		_listClassID.push_back(i);
	}
	Sort(_listDistance, _listClassID, true);

	//Only take 3 videos as candidate
	_listCandidateVideoID.push_back(_listClassID[0]);
	_listCandidateVideoID.push_back(_listClassID[1]);
	_listCandidateVideoID.push_back(_listClassID[2]);

	return _listCandidateVideoID;
}

void TestDatabase(string categoryName, int database_type)
{
	//Load database
	vector<Mat> trainData, trainLabels;
	vector<string> _listTrainingData = SetupModel(categoryName, trainData, trainLabels, database_type);

	//Load dictionary for BOW model
	if (database_type == 1)
		bowDE.setVocabulary(LoadBOWDictionaryFromFile("Data/Training_Data/" + categoryName + "_BOW_dictionary.xml"));

	int countTotal = 0;
	int countMatch = 0;
	string pathTest = "Data/Test_images/" + categoryName + "/";
	vector<string> _listTestClass = ReadFileList(pathTest);
	vector<string> _litsRawClass = ReadFileList("Data/Raws/" + categoryName + "/");

	for (int index = 0; index < _listTestClass.size(); index++)
	{
		string pathClass = pathTest + _listTestClass[index] + "/";
		vector<string> _listTestImage = ReadFileList(pathClass);
		for (int i = 0; i < _listTestImage.size(); i++)
		{
			string testpath = pathClass + _listTestImage[i];
			Mat testImage = imread(testpath);

			//---Extract feature based on database_type
			Mat queryFeature;
			if (database_type == 1)
				queryFeature = ExtractBOWFeature(bowDE, detector, testImage);
			else
				queryFeature = ExtractMPEGFeature(testImage);

			//For each test image, retrieve list of video and check if there's any correct video in candidate list
			vector<int> predictLabels = TestFrameRetrieval(_listTestClass.size(), queryFeature, trainData);
			if (predictLabels.size() == 0)
			{
				countTotal--;
			}
			else
			{
				cout << _listTestImage[i] << " can be belongs to: " << endl;
				for (int j = 0; j < predictLabels.size(); j++)
				{
					int predictLabel = predictLabels[j];
					cout << _listTestClass[predictLabel] << endl;
					if (predictLabel == index)
					{
						countMatch++;						
						//break;
					}					
					string videoName = "Data/Raws/" + categoryName + "/" + _litsRawClass[predictLabel];
					VideoShotRetrieval(videoName, trainData[predictLabel], trainLabels[predictLabel], queryFeature);
				}
				cout << endl;
			}
		}
		countTotal += _listTestImage.size();
	}

	float accuracy = (float)countMatch / (float)countTotal * 100.0f;
	cout << "Accuracy = " << accuracy << endl;
}