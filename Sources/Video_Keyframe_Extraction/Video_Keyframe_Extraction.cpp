// Video_Keyframe_Extraction.cpp : Defines the entry point for the console application.
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
string GetName(string filename)
{
	for (int i = 0; i<filename.size(); i++)
	{
		if (filename[i] == '.')
		{
			return filename.substr(0, i);
		}
	}
	return "";
}

//
//string IdentifyIDFromKeyFrame(string filename)
//{
//	int shotID = -1;
//
//	int lc_right = 0, lc_left = 0;
//	int countLC = 0;
//	for (int characterIndex = 0; characterIndex < filename.size(); characterIndex++)
//	{
//		if (filename[characterIndex] == '_')
//		{
//			if (countLC == 1)
//				lc_left = characterIndex + 1;
//			countLC++;
//		}
//		else if (filename[characterIndex] == '.')
//		{
//			lc_right = characterIndex;
//			break;
//		}
//	}
//
//	return filename.substr(lc_left, lc_right - lc_left);
//}
//
//void RepairKeyframeInfo(string videoShotPath, string videoKeyframePath)
//{
//	vector<int> _listShotID;
//	vector<string> _listShot = ReadFileList(videoShotPath);
//	for (int i = 0; i < _listShot.size(); i++)
//	{
//		int ID = IdentifyShotFromKeyFrame(_listShot[i]);
//		_listShotID.push_back(ID);
//	}
//	Sort(_listShotID, _listShot, true);
//
//	vector<int> _listKeyShotID;
//	vector<string> _listKeyframe = ReadFileList(videoKeyframePath);
//	for (int i = 0; i < _listKeyframe.size(); i++)
//	{
//		int ID = IdentifyShotFromKeyFrame(_listKeyframe[i]);
//		_listKeyShotID.push_back(ID);
//	}
//	Sort(_listKeyShotID, _listKeyframe, true);
//
//	char buffer[21];
//	int prevShotID = IdentifyShotFromKeyFrame(_listKeyframe[0]);
//	int ShotIndex = 0;
//	for (int i = 0; i < _listKeyframe.size(); i++)
//	{
//		int shotID = IdentifyShotFromKeyFrame(_listKeyframe[i]);
//		if (shotID != prevShotID)
//		{
//			ShotIndex++;
//		}
//		string shotName = GetName(_listShot[ShotIndex]);
//		string keyName = IdentifyIDFromKeyFrame(_listKeyframe[i]);
//		shotName += "_" + keyName + ".jpg";
//
//		string oldname = videoKeyframePath + _listKeyframe[i];
//		string newname = videoKeyframePath + shotName;
//		rename(oldname.c_str(), newname.c_str());
//
//		cout << _listKeyframe[i] << " ---> " << shotName << endl;
//
//		prevShotID = shotID;
//	}
//}

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
		string result = "Data/Video shots/" + categoryName + "/" + folderName + "_shot.avi";

		VideoSegmentation(path + videoName, result, 1, -1, 1.0f, width, height);
	}

	//Run video shot extraction code
	VideoCapture cap(path + videoName);
	string shotDir = "Data/Video shots/" + categoryName + "/";
	if (CreateDirectoryA(shotDir.c_str(), NULL) || ERROR_ALREADY_EXISTS == GetLastError())
	{
		string shotpath = shotDir + folderName + "_shot.avi.txt";
		VideoShotExtractor(cap, categoryName + "/" + folderName, shotpath, width, height);
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
	string pathRaw = "Data/Key frames/" + categoryName + "/";
	vector<string> _listRawName = ReadFileList(pathRaw.c_str());

	string pathTest = "Data/Test_images/" + categoryName + "/";
	if (CreateDirectoryA(pathTest.c_str() , NULL) || ERROR_ALREADY_EXISTS == GetLastError())
	{
		int numFilePerVideo = 100;// _listRawName.size() * 10;
		for (int i = 0; i < _listRawName.size(); i++)
		{
			string rawVideoPath = "Data/Video shots/" + categoryName + "/" + _listRawName[i];

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

	string choice;
	cout << "Do you want to summarize the video before extracting keyframe (Y/N): ";
	cin >> choice;
	if ( choice.compare("Y")==0 )
		VideoShotExtraction(videoName, categoryName);

	string foldername = GetName(videoName);
	ExtractAndSaveKeyFrame(foldername, categoryName);
}


int _tmain(int argc, _TCHAR* argv[])
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
	} 
	while (categoryID < 0 && categoryID >= categoryList.size());
	categoryName = categoryList[categoryID];

	//string _videoPath = "Data/Key frames/" + categoryName + "/";
	//vector<string> _listVideo = ReadFileList(_videoPath);
	//for (int i = 0; i < _listVideo.size(); i++)
	//{
	//	string videoShotPath = "Data/Video shots/" + categoryName + "/" + _listVideo[i] + "/";
	//	string videoKeyframePath = _videoPath + _listVideo[i] + "/";
	//	RepairKeyframeInfo(videoShotPath, videoKeyframePath);
	//}

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
	}
	while (choice<1 || choice>5);

	//-----------PROCESSING-----------//
	clock_t t;

	t = clock();

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
			char choice3;
			cout << "Do you want to create new test set (Y/N)? ";
			cin >> choice3;
			if (choice3 == 'Y' || choice3 == 'y')
			{
				CreateTestSet(categoryName);
			}
			TestVideoRetrieval(categoryName, database_type);
		}	
		else if (choice == 4)
		{
			char choice4;
			cout << "Do you want to create new test set (Y/N)? ";
			cin >> choice4;
			if (choice4 == 'Y' || choice4 == 'y')
			{
				CreateTestSet(categoryName);
			}
			TestShotRetrieval(categoryName, database_type);
		}
		else
		{
			string imagePath;
			cout << "Please input path to image you want to query: ";
			cin >> imagePath;
			TestIndividualImage(imagePath, categoryName, database_type);
		}

		break;
	}	

	t = clock() - t;

	cout << "It took " << ((float)t) / CLOCKS_PER_SEC << " seconds to complete all tasks" << endl;

	return 0;
}

