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
	srand((unsigned int)time(NULL));
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

void CreateBOWDictionary()
{
	//--------Using SURF as feature extractor and FlannBased for assigning a new point to the nearest one in the dictionary
	Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("FlannBased");
	Ptr<DescriptorExtractor> extractor = new SurfDescriptorExtractor();
	SurfFeatureDetector detector(500);

	//---dictionary size=number of cluster's centroids
	int dictionarySize = 1000;
	BOWImgDescriptorExtractor bowDE(extractor, matcher);

//	CreateVocaburary(bowDE, dictionarySize);

	CreateBOWTrainingSet(dictionarySize, detector, bowDE);
}

int TestANNModel()
{
	//--------Using SURF as feature extractor and FlannBased for assigning a new point to the nearest one in the dictionary
	Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("FlannBased");
	Ptr<DescriptorExtractor> extractor = new SurfDescriptorExtractor();
	SurfFeatureDetector detector(500);

	//---dictionary size=number of cluster's centroids
	int dictionarySize = 1000;
	BOWImgDescriptorExtractor bowDE(extractor, matcher);

	bowDE.setVocabulary( LoadBOWDictionaryFromFile("Data/BOW_dictionary.xml") );

	CvANN_MLP mlp;
	mlp.load("Data/ANN_Model.txt");

	Mat testImage = imread("Data/Test_images/shot_17251.png");

	Mat bowDescriptor;
	vector<KeyPoint> keypoint;
	detector.detect(testImage, keypoint);
	if (keypoint.size() > 0)
	{
		bowDE.compute(testImage, keypoint, bowDescriptor);

		Mat response(1, dictionarySize, CV_32F);
		mlp.predict(bowDescriptor, response);

		int predictLabel = 0;
		int maxValue = response.at<float>(0, 0);
		for (int i = 1; i < response.cols; i++)
		{
			int value = response.at<float>(0, i);
			if (maxValue < value)
			{
				maxValue = value;
				predictLabel = i;
			}
		}

		return predictLabel;
	}
	else
	{
		cout << "No keypoint can be found in this image. Cannot do predict class." << endl;

		return -1;
	}
}

int _tmain(int argc, _TCHAR* argv[])
{
	//string videoName = "Shinkenger_vs_Goseiger";

	//string nameExtension = ".flv";

	//VideoShotExtraction(videoName, nameExtension);

	//ExtractAndSaveKeyFrame(videoName);

	//TestQueryVideoByFrame(videoName, nameExtension);

	//CreateBOWDictionary();

	int classLabel = TestANNModel();

	string videoName = "", nameExtension = "";

	switch (classLabel)
	{
	case 0:
		videoName = "Gokaiger_Ep40";
		nameExtension = ".mp4";
		break;
	case 1:
		videoName = "Shinkenger_vs_Goseiger";
		nameExtension = ".flv";
		break;
	case 2:
		videoName = "Super_Hero_Taisen";
		nameExtension = ".mkv";
		break;
	}

	cout << "This frame belong to video " << videoName << nameExtension << endl;

	return 0;
}

