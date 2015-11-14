// Video_Keyframe_Extraction.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include "Create.h"
#include "VideoShotRetrieval.h"
#include "FeatureExtraction.h"
#include "KeyframeExtraction.h"
#include "MachineLearning.h"

//--------Using SURF as feature extractor and FlannBased for assigning a new point to the nearest one in the dictionary
Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("FlannBased");
Ptr<DescriptorExtractor> extractor = new SiftDescriptorExtractor();
SiftFeatureDetector detector(500);

//---dictionary size=number of cluster's centroids
int dictionarySize = 1000;
BOWImgDescriptorExtractor bowDE(extractor, matcher);

void VideoSegmentation(string path,string result,int level,int numFrame,float percent,int width=0,int height=0)
{	
	VideoCapture cap(path);

	double frameCount = cap.get(CV_CAP_PROP_FRAME_COUNT);
	double fps = cap.get(CV_CAP_PROP_FPS);
	double w = cap.get(CV_CAP_PROP_FRAME_WIDTH);
	double h = cap.get(CV_CAP_PROP_FRAME_HEIGHT);

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

void VideoShotExtraction(string videoName, string nameExtension, string categoryName="", int width = 0, int height = 0)
{
	//Run video summarization code
	string path = "Data/Raws/" + categoryName + "/";
	if (CreateDirectoryA(path.c_str(), NULL) || ERROR_ALREADY_EXISTS == GetLastError())
	{
		string result = "Data/Video shots/" + categoryName + "/"  + videoName + "_shot.avi";

		VideoSegmentation(path + videoName + nameExtension, result, 1, -1, 1.0f, width, height);
	}

	//Run video shot extraction code
	VideoCapture cap(path + videoName + nameExtension);
	string shotDir = "Data/Video shots/" + categoryName + "/";
	if (CreateDirectoryA(shotDir.c_str(), NULL) || ERROR_ALREADY_EXISTS == GetLastError())
	{
		string shotpath = shotDir + videoName + "_shot.avi.txt";

		VideoShotExtractor(cap, categoryName + "/" + videoName, shotpath, width, height);
	}
}

void CreateDatabase(string categoryName, int database_type)
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

void RandomizeFrameTest(string rawVideoPath, string categoryName, string videoName, int numFrame)
{
	srand(time(NULL));
	vector<int> listIndex;

	VideoCapture cap(rawVideoPath);
	int numCapFrame = cap.get(CV_CAP_PROP_FRAME_COUNT);
	for (int i = 0; i < numCapFrame; i++)
	{
		listIndex.push_back(i);
	}
	Shuffle(&listIndex[0], numCapFrame);

	for (int i = 0; i < numFrame; i++)
	{
		//Seek to frame with random ID
		cap.set(CV_CAP_PROP_POS_FRAMES, (double)listIndex[i]);
		Mat testImage;
		cap.read(testImage);

		//Write this test image out to the folder
		char *buffer = new char[255];
		_itoa_s(listIndex[i], buffer, 255, 10);
		string testFileName = "Data/Test_images/" + categoryName + "/" + videoName + "/" + videoName + "_ID_" + buffer + ".png";

		//Write image to test data
		imwrite(testFileName, testImage);
		delete buffer;
	}
}

void CreateTestSet(int numTestFile, string categoryName = "")
{
	string pathRaw = "Data/Raws/" + categoryName + "/";
	vector<string> _listRawName = ReadFileList(pathRaw.c_str());

	string pathTest = "Data/Test_images/" + categoryName + "/";
	if (CreateDirectoryA(pathTest.c_str() , NULL) || ERROR_ALREADY_EXISTS == GetLastError())
	{
		int numFilePerVideo = numTestFile / _listRawName.size();
		for (int i = 0; i < _listRawName.size(); i++)
		{
			string rawVideoPath = pathRaw + _listRawName[i];
			string videoName = _listRawName[i].substr(0, _listRawName[i].size() - 4);
			string testVideoPath = pathTest + videoName;

			cout << "Extract test frame from video " << videoName << endl;
			if (CreateDirectoryA(testVideoPath.c_str(), NULL) || ERROR_ALREADY_EXISTS == GetLastError())
			{
				RandomizeFrameTest(rawVideoPath, categoryName, videoName, numFilePerVideo);
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

int _tmain(int argc, _TCHAR* argv[])
{
	clock_t t;

	t = clock();

	string videoName = "Vacation";

	string nameExtension = ".mp4";

	string categoryName = "RipDVD";

	VideoShotExtraction(videoName, nameExtension, categoryName);

	ExtractAndSaveKeyFrame(videoName, categoryName);

	//CreateTestSet(1000, categoryName);

	//int database_type = 1;

	//CreateDatabase(categoryName, database_type);

	//TestDatabase(categoryName, database_type);

	t = clock() - t;

	cout << "It took " << ((float)t) / CLOCKS_PER_SEC << " seconds to complete all tasks" << endl;

	return 0;
}

