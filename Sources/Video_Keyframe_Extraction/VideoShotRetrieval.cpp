#include "stdafx.h"
#include "VideoShotRetrieval.h"

float CalculateFeatureLineCoef(Mat f1, Mat f2, Mat fx)
{
	Mat a = fx - f1;
	Mat b = f2 - f1;
	double numerator   = a.dot(b);
	double denominator = b.dot(b);
	return (float)(numerator / denominator);
}

Mat CalculateFeatureList(vector<Mat> listKeyFrame,int featureSize)
{
	Mat featureList(0, featureSize, CV_32FC1);

	for (int i = 0; i < listKeyFrame.size(); i++)
	{
		Mat feature = GetColorStructureDescriptor(listKeyFrame[i], featureSize);
		featureList.push_back(feature);
	}

	return featureList;
}

float CalculateNFLDistance(vector<string> listFileName, Mat queryImage, int featureSize)
{
	//Calculate descriptor for query image
	Mat fx = GetColorStructureDescriptor(queryImage, featureSize);

	//Get list of image corresponding to the list of name
	vector<Mat> listKeyFrame;
	for (int i = 0; i < listFileName.size(); i++)
	{
		Mat frame = imread(listFileName[i]);
		listKeyFrame.push_back(frame);
	}
	
	//Calculate descriptor for a list of key frame
	Mat keyframeDescriptor = CalculateFeatureList(listKeyFrame, featureSize);

	float minDistance = -1.0f;

	if (keyframeDescriptor.rows > 1)
	{
		for (int i = 0; i < keyframeDescriptor.rows - 1; i++)
		{
			Mat fi = keyframeDescriptor.row(i);
			for (int j = i + 1; j < keyframeDescriptor.rows; j++)
			{
				Mat   fj = keyframeDescriptor.row(j);
				float muy = CalculateFeatureLineCoef(fi, fj, fx);
				Mat   px = fi + muy*(fj - fi);
				float distance = CalcVectorMagnitude((fx - px));

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

void VideoShotRetrieval(string videoName, Mat queryImage)
{
	string pathName = "Data/Key frames/" + videoName + "/";
	vector<string> listFileName = ReadFileList(pathName);

	int step = 0;
	int shotID = 0;
	float minDistance = -1;

	for (int i = 0; i < listFileName.size(); i+=step)
	{
		vector<string> listKeyframe = GetKeyframeList(listFileName, pathName, i);
		float distance = CalculateNFLDistance(listKeyframe, queryImage, 64);
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