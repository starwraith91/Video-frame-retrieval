#include "stdafx.h"
#include "VideoShotRetrieval.h"

extern int dictionarySize;
extern SiftFeatureDetector detector;
extern BOWImgDescriptorExtractor bowDE;

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

Mat CalculateFeatureList(vector<Mat> listKeyFrame)
{
	Mat featureList(0, dictionarySize, CV_32FC1);

	for (int i = 0; i < listKeyFrame.size(); i++)
	{
		//Mat feature = GetColorStructureDescriptor(listKeyFrame[i], featureSize);
		Mat feature = ExtractBOWFeature(bowDE, detector, listKeyFrame[i]);
		featureList.push_back(feature);
	}

	return featureList;
}

float CalculateNFLDistance(vector<string> listFileName, Mat queryFeature)
{
	//Calculate descriptor for query image
	Mat fx = queryFeature;

	//Get list of image corresponding to the list of name
	vector<Mat> listKeyFrame;
	for (int i = 0; i < listFileName.size(); i++)
	{
		Mat frame = imread(listFileName[i]);
		listKeyFrame.push_back(frame);
	}
	
	//Calculate descriptor for a list of key frame
	Mat keyframeDescriptor = CalculateFeatureList(listKeyFrame);

	float minDistance = -1.0f;

	if (keyframeDescriptor.rows > 1)
	{
		for (int i = 0; i < keyframeDescriptor.rows - 1; i++)
		{
			Mat fi = keyframeDescriptor.row(i);
			//for (int j = i + 1; j < keyframeDescriptor.rows; j++)

			int j = i + 1;
			{
				Mat   fj = keyframeDescriptor.row(j);
				float distance = CalcDistanceToFeatureLine(fi, fj, fx);
				if (minDistance == -1.0f || distance < minDistance)
				{
					minDistance = distance;
				}
			}
		}
	}
	else
	{
		Mat px = keyframeDescriptor.row(0);
		float distance = CalcVectorMagnitude( (fx - px) );

		if (minDistance == -1.0f || distance < minDistance)
		{
			minDistance = distance;
		}
	}

	return minDistance;
}

vector<string> GetKeyframeList(vector<string> listFileName, string path, int index)
{
	vector<string> listShotFrame;
	listShotFrame.push_back( path + listFileName[index] );

	int queryID = IdentifyShotFromKeyFrame(listFileName[index]);
	for (int i = index+1; i < listFileName.size(); i++)
	{
		int ID = IdentifyShotFromKeyFrame( listFileName[i] );
		if (ID == queryID)
		{
			string pathName = path + listFileName[i];
			listShotFrame.push_back(pathName);
		}
		else
		{
			break;
		}
	}

	return listShotFrame;
}

void VideoShotRetrieval(string videoName, Mat queryFeature)
{
	string pathName = "Data/Key frames/" + videoName + "/";
	vector<string> listFileName = ReadFileList(pathName);

	int step = 0;
	int shotID = 0;
	float minDistance = -1;

	for (int i = 0; i < listFileName.size(); i+=step)
	{
		vector<string> listKeyframe = GetKeyframeList(listFileName, pathName, i);
		float distance = CalculateNFLDistance(listKeyframe, queryFeature);
		if (minDistance == -1 || distance < minDistance)
		{
			shotID = IdentifyShotFromKeyFrame(listFileName[i]);
			minDistance = distance;
		}
		step = listKeyframe.size();
	}

	cout << "This image belongs to shot " << shotID << " of video " << videoName << endl;
	cout << "with minimum distance of " << minDistance << endl;
}

void VideoShotRetrieval(string videoName, Mat trainingData, Mat trainingLabel, Mat queryFeature)
{
	int shotID = -1;	
	float minDistance = -1;

	int prevID = trainingLabel.at<int>(0, 0);
	Mat f1 = trainingData.row(0);
	int countFrame = 1;
	for (int i = 1; i<trainingData.rows-1; i++)
	{
		int currentID = trainingLabel.at<int>(i, 0);
		Mat f2 = trainingData.row(i);
		if (prevID != currentID)
		{
			if (countFrame == 1)
			{
				float distance = CalcEuclidianDistance(f2, queryFeature);
				if (minDistance == -1 || minDistance > distance)
				{
					minDistance = distance;
					shotID = currentID;
				}
			}
			countFrame = 1;
		}
		else
		{			
			float distance = CalcDistanceToFeatureLine(f1, f2, queryFeature);
			if (minDistance == -1 || minDistance > distance)
			{
				minDistance = distance;
				shotID = currentID;
			}
			countFrame++;
		}

		f1 = f2;
		prevID = currentID;
	}

	cout << "This image belongs to shot " << shotID << endl;
	cout << "with minimum distance of " << sqrt(minDistance) << endl;
}

vector<string> SetupBOWModel(vector<Mat> &trainData, vector<Mat> &trainLabels)
{
	cout << "Loading training data..." << endl;

	//Get data stored in database
	string path = "Data/BOW_Data/";
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

int TestFrameClassification(vector<string> _listName, string imageName, Mat testImage, vector<Mat> trainData, vector<Mat> trainLabels, int numClass, int database_type)
{
	//---Classify image into video class
	Mat testFeature;
	
	if (database_type == 1)
		testFeature = ExtractBOWFeature(bowDE, detector, testImage);
	else
		testFeature = ExtractMPEGFeature(testImage);

	if (testFeature.empty())
		return -1;

	int label = -1;
	int indexDist = -1;
	float minDistance = -1;
	for (int i = 0; i < numClass; i++)
	{
		//vector<string> _listKeyframe = ReadFileList("Data/Key frames/"+_listTrainingData[i].substr(0, _listTrainingData[i].size() - 4));
		float distance = CalcDistanceFromSet(testFeature, trainData[i], indexDist);
		//cout << "Match best with " << _listKeyframe[indexDist] << " in " << _listTrainingData[i].substr(0, _listTrainingData[i].size() - 4) << " with min dist = " << distance << endl;
		if (minDistance == -1 || minDistance > distance)
		{
			label = i;
			minDistance = distance;
		}
	}

	cout << imageName << " belongs to video " << _listName[label] << endl;

	//---Search shot that may contain query frame
	string videoPath = "Data/Raws/";
	vector<string> _listVideoName = ReadFileList(videoPath);
	for (int i = 0; i < _listVideoName.size(); i++)
	{
		string name = _listVideoName[i].substr(0, _listVideoName[i].size() - 4);
		if (_listName[label].compare(name) == 0)
		{
			videoPath = videoPath + _listVideoName[i];
			break;
		}
	}
	VideoShotRetrieval(videoPath, trainData[label], trainLabels[label], testFeature);
	cout << endl;

	return label;
}

void TestDatabase(int database_type)
{
	vector<Mat> trainData, trainLabels;
	vector<string> _listTrainingData = SetupBOWModel(trainData, trainLabels);

	if (database_type == 1)
		bowDE.setVocabulary(LoadBOWDictionaryFromFile("Data/BOW_dictionary.xml"));

	int countTotal = 0;
	int countMatch = 0;
	vector<string> _listTestClass = ReadFileList("Data/Test_images/");
	for (int index = 0; index < _listTestClass.size(); index++)
	{
		string pathClass = "Data/Test_images/" + _listTestClass[index] + "/";
		vector<string> _listTestImage = ReadFileList(pathClass);
		for (int i = 0; i < _listTestImage.size(); i++)
		{
			string testpath = pathClass + _listTestImage[i];
			Mat testImage = imread(testpath);
			int predictLabel = TestFrameClassification(_listTestClass, _listTestImage[i], testImage, trainData, trainLabels, _listTestClass.size(), database_type);
			if (predictLabel == index)
			{
				countMatch++;
			}
			else
			{
				if (predictLabel == -1)
				{
					countTotal--;
				}
			}
		}
		countTotal += _listTestImage.size();
	}

	float accuracy = (float)countMatch / (float)countTotal * 100.0f;
	cout << "Accuracy = " << accuracy << endl;
}