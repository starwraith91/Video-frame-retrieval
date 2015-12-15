#include "stdafx.h"
#include "VideoShotRetrieval.h"

extern int dictionarySize;
extern SiftFeatureDetector detector;
extern BOWImgDescriptorExtractor bowDE;

vector<string> SetupModel(string categoryName, vector<Mat> &trainData, vector<Mat> &trainLabels, int database_type)
{
	cout << "Loading training data..." << endl << endl;

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

float ShotRetrievalPerformance(vector<int> _listRetrieveShotID, int trueShotID, int numShotConsider)
{
	int countShotMatch = 0;
	float nuy = 0;

	if (_listRetrieveShotID.size() > 0)
	{
		float sumW = 0;
		for (int j = 1; j <= 1; j++)
			sumW += 1.0f / (float)j;

		int minNum = MIN(_listRetrieveShotID.size(), numShotConsider);
		for (int k = 1; k <= numShotConsider; k++)
		{			
			//Weight score			
			if (_listRetrieveShotID[k - 1] == trueShotID)
			{
				float w = (1.0f / sumW) * (1.0f / (float)k);
				nuy += w;
			}			
		}
	}

	cout << "Weight score = " << nuy << endl;

	return nuy;
}

vector<int> RetrieveShot(string videoName, Mat trainingData, Mat trainingLabel, Mat queryFeature)
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
		float minDistLabel1 = -1;
		float minDistLabel2 = -1;
		if (featureMat.rows == 1)
		{
			Mat f2 = featureMat.row(0);
			minDistLabel1 = minDistLabel2 = 0;
			minDistance = CalcEuclideanDistance(f2, queryFeature);
		}
		else //If not, calculate distance using NFL method
		{			
			for (int i = 0; i < featureMat.rows-1; i++)
			{
				Mat f1 = featureMat.row(i);
				for (int j = i + 1; j < featureMat.rows; j++)
				{
					Mat f2 = featureMat.row(j);
					float distance = CalcDistanceToFeatureLine(f1, f2, queryFeature);
					if (minDistance == -1 || minDistance > distance)
					{
						minDistance = distance;
						minDistLabel1 = i;
						minDistLabel2 = j;
					}
				}
			}
		}

		//Update list of distance and shot ID
		_listShotDistance.push_back(minDistance);
		_listShotID.push_back(currentShotID);

		int minDistLabel = minDistLabel1;
		if (featureMat.rows > 1)
		{
			float distance1 = CalcEuclideanDistance(queryFeature, featureMat.row(minDistLabel1));
			float distance2 = CalcEuclideanDistance(queryFeature, featureMat.row(minDistLabel2));
			if (distance2 < distance1)
				minDistLabel = minDistLabel2;
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

//	float threshold = 0.3f;
	vector<int> _listResultShotID;
	for (int i = 0; i < _listShotDistance.size(); i++)
	{
		if (i < 5)
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

		_listResultShotID.push_back(_listShotID[_listIndex[i]]);
	}
	
	return _listResultShotID;
}

vector<int> RetrieveVideo(int numClass, Mat queryFeature, vector<Mat> trainData)
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

	//Only take 3 videos as candidate and distance is under a threshold
	for (int i = 0; i < _listClassID.size(); i++)
	{
		_listCandidateVideoID.push_back(_listClassID[i]);
	}

	return _listCandidateVideoID;
}

void TestIndividualImage(string imagePath, string categoryName, int database_type)
{
	//Load image 
	Mat testImage = imread(imagePath);

	//Load database
	vector<Mat> trainData, trainLabels;
	vector<string> _listTrainingData = SetupModel(categoryName, trainData, trainLabels, database_type);

	//Load dictionary for BOW model
	if (database_type == 1)
		bowDE.setVocabulary(LoadBOWDictionaryFromFile("Data/Training_Data/" + categoryName + "_BOW_dictionary.xml"));

	vector<string> _litsRawClass = ReadFileList("Data/Raws/" + categoryName + "/");

	clock_t t;

	t = clock();

	//---Extract feature based on database_type
	Mat queryFeature;
	if (database_type == 1)
		queryFeature = ExtractBOWFeature(bowDE, detector, testImage);
	else
		queryFeature = ExtractMPEGFeature(testImage);

	//For each test image, retrieve list of video and check if there's any correct video in candidate list
	vector<int> predictLabels = RetrieveVideo(_litsRawClass.size(), queryFeature, trainData);
	if (predictLabels.size() > 0)
	{
		cout << "This image can belong to: " << endl;
		for (int j = 0; j < 3; j++)
		{
			int predictLabel = predictLabels[j];
			cout << _litsRawClass[predictLabel] << endl;

			string videoName = "Data/Raws/" + categoryName + "/" + _litsRawClass[predictLabel];
			RetrieveShot(videoName, trainData[predictLabel], trainLabels[predictLabel], queryFeature);
		}
		cout << endl;
	}
	else
	{
		cout << "No video match the query image" << endl << endl;
	}

	t = clock() - t;

	cout << "It took " << ((float)t) / CLOCKS_PER_SEC << " seconds to complete all tasks" << endl;
}

void TestVideoRetrieval(string categoryName, int database_type)
{
	//Load database
	vector<Mat> trainData, trainLabels;
	vector<string> _listTrainingData = SetupModel(categoryName, trainData, trainLabels, database_type);

	//Load dictionary for BOW model
	if (database_type == 1)
		bowDE.setVocabulary(LoadBOWDictionaryFromFile("Data/Training_Data/" + categoryName + "_BOW_dictionary.xml"));

	float avgReciprocalRank = 0.0f;
	float avgPrecision = 0.0f;
	float avgRecall = 0.0f;
	string pathTest = "Data/Test_images/" + categoryName + "/";
	vector<string> _listTestClass = ReadFileList(pathTest);

	int countTotal = 0;
	for (int index = 0; index < _listTestClass.size(); index++)
	{
		float avgClassRR = 0.0f;

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
			int countMatch = 0;
			int countMatchRank = 0;
			int countRetrievedMatch = 0;
			int numMatchRetrieve = 3;
			vector<int> predictLabels = RetrieveVideo(_listTestClass.size(), queryFeature, trainData);

			//Calculate precision and recall from retrieved list
			if (predictLabels.size() > 0)
			{
				cout << _listTestImage[i] << " can belong to: " << endl;
				for (int j = 0; j < predictLabels.size(); j++)
				{
					int predictLabel = predictLabels[j];
					cout << _listTestClass[predictLabel] << endl;

					countMatchRank++;
					if (predictLabel == index)
					{
						if (j < numMatchRetrieve)
						{
							countRetrievedMatch++;
						}
						countMatch++;
						break;
					}
				}
				cout << endl;
			}
			else
			{
				cout << "No video match the query image" << endl << endl;
			}

			float reciprocalrank = 0.0f;
			float precision = 0.0f;
			float recall = 0.0f;
			if (predictLabels.size() > 0)
			{
				reciprocalrank = (float)countMatch / (float)countMatchRank;
				precision = (float)countRetrievedMatch / (float)numMatchRetrieve;
				recall = countRetrievedMatch;
			}
				
			avgClassRR += reciprocalrank;
			avgPrecision += precision;
			avgRecall += recall;
		}

		countTotal += _listTestImage.size();

		avgReciprocalRank += avgClassRR;
	}

	avgReciprocalRank = avgReciprocalRank / countTotal * 100.0f;
	avgPrecision = avgPrecision / countTotal * 100.0f;
	avgRecall = avgRecall / countTotal * 100.0f;

	cout << "MAP = " << avgReciprocalRank << endl;
	cout << "Average Precision = " << avgPrecision << endl;
	cout << "Average Recall = " << avgRecall << endl;
}

void TestShotRetrieval(string categoryName, int database_type)
{
	string resultFile = "Data/Result_" + categoryName + "_ShotRetrieval.txt";
	ofstream out(resultFile);

	//Load database
	vector<Mat> trainData, trainLabels;
	vector<string> _listTrainingData = SetupModel(categoryName, trainData, trainLabels, database_type);

	//Load dictionary for BOW model
	if (database_type == 1)
		bowDE.setVocabulary(LoadBOWDictionaryFromFile("Data/Training_Data/" + categoryName + "_BOW_dictionary.xml"));

	string pathTest = "Data/Test_images/" + categoryName + "/";
	vector<string> _listTestClass = ReadFileList(pathTest);
	vector<string> _litsRawClass = ReadFileList("Data/Raws/" + categoryName + "/");

	int countTotal = 0;
	vector<float> _listPrecisionShot;
	for (int index = 0; index < _listTestClass.size(); index++)
	{
		float avgPrecisionShotRetrieval = 0.0f;

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

			//For each test image, retrieve list of shot and check if there's any correct shot in candidate list
			string videoName = "Data/Raws/" + categoryName + "/" + _litsRawClass[index];
			int trueShotID = IdentifyShotFromKeyFrame(_listTestImage[i]);

			cout << "Retrieve shot in video " << _litsRawClass[index] << " using " << _listTestImage[i] << endl;
			vector<int> _listShotID = RetrieveShot(videoName, trainData[index], trainLabels[index], queryFeature);

			//Calculate precision and recall
			if (_listShotID.size() > 0)
			{
				float precision = ShotRetrievalPerformance(_listShotID, trueShotID, 5);
				avgPrecisionShotRetrieval += precision;

				cout << endl;
			}
			else
			{
				cout << "No shot match the query image" << endl << endl;
			}
		}

		countTotal += _listTestImage.size();

		avgPrecisionShotRetrieval = avgPrecisionShotRetrieval / _listTestImage.size();// *100.0f;
		cout << _listTestClass[index] << " = " << avgPrecisionShotRetrieval << endl;

		_listPrecisionShot.push_back(avgPrecisionShotRetrieval);
	}

	float avgPrecision = 0.0f;
	for (int i = 0; i < _listPrecisionShot.size(); i++)
	{
		out << _listTestClass[i].c_str() << "			" << _listPrecisionShot[i] << endl;
		avgPrecision += _listPrecisionShot[i];
	}
	out << "Average precision = " << avgPrecision / (float)_listPrecisionShot.size() << endl;

	cout << endl << "Average precision = " << avgPrecision / (float)_listPrecisionShot.size() << endl;

	out.close();
}