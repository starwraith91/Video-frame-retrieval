﻿// Video_Keyframe_Extraction.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include "Create.h"
#include "VideoShotRetrieval.h"
#include "FeatureExtraction.h"
#include "KeyframeExtraction.h"

//--------Using SURF as feature extractor and FlannBased for assigning a new point to the nearest one in the dictionary
Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("FlannBased");
Ptr<DescriptorExtractor> extractor = new SiftDescriptorExtractor();
SiftFeatureDetector detector(500);

//---dictionary size=number of cluster's centroids
int dictionarySize = 1000;
BOWImgDescriptorExtractor bowDE(extractor, matcher);

////-----TEMPORARY_FUNCTION-----//

//Return name of file without extension
void VideoSegmentation(string path,string result,int level,int numFrame,float percent,int width=0,int height=0)
{	
	VideoCapture cap(path);

	double frameCount = cap.get(CV_CAP_PROP_FRAME_COUNT);
	double fps = cap.get(CV_CAP_PROP_FPS);
	double w = cap.get(CV_CAP_PROP_FRAME_WIDTH);
	double h = cap.get(CV_CAP_PROP_FRAME_HEIGHT);

	cout << "Summarizing video " + path << endl;
	cout << "Frame count: " << frameCount << endl;
	cout << "FPS: " << fps << endl;
	cout << "Width: " << w << endl;
	cout << "Height: " << h << endl;

	if (width!=0 && height!=0)
		run(path.c_str(), result.c_str(), level, numFrame, percent, cvSize(width, height));
	else
		run(path.c_str(), result.c_str(), level, numFrame, percent, cvSize(w, h));

	cap.release();
}

void VideoShotExtraction(string videoName, string categoryName="", int width = 0, int height = 0)
{
	//Run video summarization code
	string path = "Data/Raws/" + categoryName + "/";
	string folderName = GetName(videoName);
	if (CreateDirectoryA(path.c_str(), NULL) || ERROR_ALREADY_EXISTS == GetLastError())
	{
		string result = "Data/Video_shots/" + categoryName + "/" + folderName + "_shot.avi";

		VideoSegmentation(path + videoName, result, 1, -1, 1.0f, width, height);
	}

	//Run video shot extraction code
	VideoCapture cap(path + videoName);
	string shotDir = "Data/Video_shots/" + categoryName + "/";
	if (CreateDirectoryA(shotDir.c_str(), NULL) || ERROR_ALREADY_EXISTS == GetLastError())
	{
		string shotpath = shotDir + folderName + "_shot.avi.txt";
		VideoShotExtractor(cap, categoryName + "/" + folderName, shotpath, width, height);
	}
	else
	{
		cout << "Cannot open shot directory" << endl;
	}
}

void CreateDatabase(string categoryName, int database_type)
{
	string pathCategory = "Data/Training_Data/" + categoryName + "/";
	if (CreateDirectoryA(pathCategory.c_str(), NULL) || ERROR_ALREADY_EXISTS == GetLastError())
	{
		if (database_type == 1)
		{
			CreateBOWTrainingSet(categoryName, dictionarySize, detector, bowDE);
		}
		else
		{
			CreateMPEGTrainingSet(categoryName);
		}
	}
}

void RandomizeFrameTest(string rawVideoPath, string categoryName, string videoName, int numFrame)
{
	srand(time(NULL));
	vector<string> _listShot = ReadFileList(rawVideoPath+"/");
	Shuffle(&_listShot[0], _listShot.size());

	for (int i = 0; i < numFrame; i++)
	{
		VideoCapture cap(rawVideoPath + "/" + _listShot[i]);
		int numFrame = cap.get(CV_CAP_PROP_FRAME_COUNT);
		double randomIndex = (double)(rand() % numFrame);

		//Seek to frame with random ID
		cap.set(CV_CAP_PROP_POS_FRAMES, (double)randomIndex);
		Mat testImage;
		cap.read(testImage);

		//Write this test image out to the folder
		char buffer[21];
		_itoa(randomIndex, buffer, 10);
		string name = GetName(_listShot[i]);
		string testFileName = "Data/Test_images/" + categoryName + "/" + videoName + "/" + name + "_" + buffer + ".jpg";

		//Write image to test data
		imwrite(testFileName, testImage);
	}
}

void CreateTestSet(string categoryName = "")
{
	string pathRaw = "Data/Keyframes/" + categoryName + "/";
	vector<string> _listRawName = ReadFileList(pathRaw.c_str());

	string pathTest = "Data/Test_images/" + categoryName + "/";
	if (CreateDirectoryA(pathTest.c_str() , NULL) || ERROR_ALREADY_EXISTS == GetLastError())
	{
		int numFilePerVideo = 100;// _listRawName.size() * 10;
		for (int i = 0; i < _listRawName.size(); i++)
		{
			string rawVideoPath = "Data/Video_shots/" + categoryName + "/" + _listRawName[i];

			string testVideoPath = pathTest + _listRawName[i];

			cout << "Extract test frame from video " << _listRawName[i] << endl;
			if (CreateDirectoryA(testVideoPath.c_str(), NULL) || ERROR_ALREADY_EXISTS == GetLastError())
			{
				DeleteAllFiles(testVideoPath);

				RandomizeFrameTest(rawVideoPath, categoryName, _listRawName[i], numFilePerVideo);
			}
			else
			{
				cout << "Cannot find and create " << testVideoPath << " directory" << endl;
			}
		}
	}
	else
	{
		cout << "Something is wrong..." << endl;
	}
}

//-----MAIN RUNNING CODE-----//

void KeyframeExtraction(string categoryName)
{
	string videoName;
	cout << "Please specify video name (abc.xyz): ";
	cin >> videoName;
	cout << endl;

	string choice;
	cout << "Do you want to summarize the video before extracting keyframe (Y/N): ";
	cin >> choice;
	if (choice.compare("Y") == 0 || choice.compare("y") == 0)
		VideoShotExtraction(videoName, categoryName);

	string foldername = GetName(videoName);
	ExtractAndSaveKeyFrame(foldername, categoryName);
}

void UserInterface()
{
	//------------CATEGORY_SELECTION---------------//
	int categoryID = -1;
	string categoryName = "";
	vector<string> categoryList = ReadFileList("Data/Raws/");
	cout << "Please specify category name of videos: " << endl;
	do
	{
		for (int i = 0; i < categoryList.size(); i++)
		{
			cout << i << ". " << categoryList[i] << endl;
		}
		cout << "Enter your choice: ";
		cin >> categoryID;
	} while (categoryID < 0 && categoryID >= categoryList.size());
	categoryName = categoryList[categoryID];

	//-----------ACTION_SELECTION-------------//
	int choice = -1;
	do
	{
		system("CLS");
		cout << "Please specify task to do:" << endl;
		cout << "1. Keyframe extraction" << endl;
		cout << "2. Create feature database" << endl;
		cout << "3. Evaluate video retrieval using set of test image" << endl;
		cout << "4. Evaluate shot retrieval using set of test image" << endl;
		cout << "5. Test with an arbitrary image" << endl;
		cout << "Please enter your choice: ";
		cin >> choice;
		cout << endl;
	} while (choice<1 || choice>5);

	//-----------PROCESSING-----------//

	switch (choice)
	{
	case 1:
		KeyframeExtraction(categoryName);
		break;
	case 2:
	case 3:
	case 4:
	case 5:
		int database_type = -1;
		cout << "What kind of feature do you want to use: " << endl;
		cout << "1. Bag of Word" << endl;
		cout << "2. MPEG-7" << endl;
		cout << "Please enter your choice: ";
		cin >> database_type;
		cout << endl;

		if (choice == 2)
		{
			CreateDatabase(categoryName, database_type);
		}
		else if (choice == 3)
		{
			string choice3;
			cout << "Do you want to create new test set (Y/N)? ";
			cin >> choice3;
			if (choice3.compare("Y") == 0 || choice3.compare("y") == 0)
			{
				CreateTestSet(categoryName);
			}
			TestVideoRetrieval(categoryName, database_type);
		}
		else if (choice == 4)
		{
			string choice4;
			cout << "Do you want to create new test set (Y/N)? ";
			cin >> choice4;
			if (choice4.compare("Y") == 0 || choice4.compare("y") == 0)
			{
				CreateTestSet(categoryName);
			}
			cout << endl;

			int num_retrieve;
			cout << "How many items do you want to retrieve? ";
			cin >> num_retrieve;
			cout << endl;
			TestShotRetrieval(categoryName, database_type, num_retrieve);
		}
		else
		{
			string imagePath;
			cout << "Please specify path to query image: ";
			cin >> imagePath;
			TestIndividualImage(imagePath, categoryName, database_type);
		}

		break;
	}
}

int _tmain(int argc, _TCHAR* argv[])
{
	//string videoName = "Data/shot_523_57122.avi";
	//VideoCapture tempCap(videoName);
	//int numframe = tempCap.get(CV_CAP_PROP_FRAME_COUNT);
	//for(int i=0;i<numframe;i++)
	//{
	//	char buffer[21];
	//	_itoa(i, buffer, 10);
	//	string filename = "Data/Temp/frame_";
	//	filename.append(buffer);
	//	filename.append(".jpg");
	//
	//	Mat frame = ExtractFrameFromVideo(tempCap, i);
	//	imwrite(filename, frame);
	//}
	//vector<int> chosenIDs = KeyframeCurvatureExtractor(tempCap);
	//for (int i = 0; i < chosenIDs.size(); i++)
	//{
	//	char buffer[21];
	//	_itoa(chosenIDs[i], buffer, 10);
	//	string filename = "Data/shot_6_2475_";
	//	filename.append(buffer);
	//	filename.append(".jpg");

	//	Mat frame = ExtractFrameFromVideo(tempCap, chosenIDs[i]);
	//	imwrite(filename, frame);
	//}

	bool isContinue;
	do
	{
		UserInterface();

		string choiceContinue;
		cout << "Is there anything else you want to do (Y/N)? ";
		cin >> choiceContinue;
		if (choiceContinue.compare("Y") == 0 || choiceContinue.compare("y") == 0)
		{
			isContinue = true;
		}
		else
		{
			isContinue = false;
		}
		system("CLS");

	} while (isContinue);



	return 0;
}

