// Video_Keyframe_Extraction.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include "Create.h"
#include "VideoShotRetrieval.h"
#include "FeatureExtraction.h"
#include "KeyframeExtraction.h"
#include "MachineLearning.h"

void VideoSegmentation(string path,string result,int level, int numFrame, float percent)
{	
	VideoCapture cap(path);

	double frameCount = cap.get(CV_CAP_PROP_FRAME_COUNT);
	double fps = cap.get(CV_CAP_PROP_FPS);
	double width = cap.get(CV_CAP_PROP_FRAME_WIDTH);
	double height = cap.get(CV_CAP_PROP_FRAME_HEIGHT);

	cout << "Frame count: " << frameCount << endl;
	cout << "FPS: " << fps << endl;
	cout << "Width: " << width << endl;
	cout << "Height: " << height << endl;

	run(path.c_str(), result.c_str(), level, numFrame, percent, cvSize(640, 480));

	cap.release();
}

void VideoShotExtraction(string videoName,string nameExtension)
{
	////Run video summarization code
	string path = "Data/Raws/" + videoName + nameExtension;

	string result = "Data/Video shots/" + videoName + "_shot.avi";

	VideoSegmentation(path, result, 0, -1, 1);

	//Run video shot extraction code
	VideoCapture cap(path);

	string shotpath = "Data/Video shots/" + videoName + "_shot.avi.txt";
	
	VideoShotExtractor(cap, videoName, shotpath);
}

void TestQueryVideoByFrame(string videoName,string nameExtension)
{
	//Take a random chosen test image from the chosen video
	string videoname = "Data/Raws/" + videoName + nameExtension;
	VideoCapture cap(videoname);
	int numFrame = cap.get(CV_CAP_PROP_FRAME_COUNT);
	srand(time(NULL));
	int randomFrame = rand() % numFrame;
	cap.set(CV_CAP_PROP_POS_FRAMES, (double)randomFrame);
	Mat testImage;
	cap.read(testImage);

	//Write this test image out to the folder
	char *buffer = new char[255];
	_itoa_s(randomFrame, buffer, 255, 10);
	string testFileName = "Data/Test_images/" + videoName + "_shot_";
	testFileName.append(buffer);
	testFileName.append(".png");
	imwrite(testFileName, testImage);
	delete buffer;

	cout << testFileName << endl;

	Mat testFrame = imread(testFileName);
	VideoShotRetrieval(videoName, testFrame);
}

int _tmain(int argc, _TCHAR* argv[])
{
	string videoName = "Super_Hero_Taisen";

	string nameExtension = ".mkv";

//	VideoShotExtraction(videoName, nameExtension);

	ExtractAndSaveKeyFrame(videoName);

	TestQueryVideoByFrame(videoName, nameExtension);

	return 0;
}

