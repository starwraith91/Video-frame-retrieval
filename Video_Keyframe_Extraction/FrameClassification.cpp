#include "stdafx.h"
#include "FrameClassification.h"

void CreateTrainingModel(string path)
{
	vector<string> ListVideoClass = ReadFileList(path);
	int numClass = (int)ListVideoClass.size();
	for (int i = 0; i < numClass; i++)
	{
		string pathFolderClass = path + ListVideoClass[i] + "/";
		vector<string> ListKeyFrame = ReadFileList(pathFolderClass);
		int numFrame = (int)ListKeyFrame.size();

		for (int j = 0; j < numFrame; j++)
		{
			//Extract feature for each key frame of a video

		}
	}

	//Create training model base on the feature matrix

	//Save model into file
	
}

int ImageClassification(Mat image)
{
	//Load model from file

	//Classify input image into one of the pre-defined classes

	return 0;
}