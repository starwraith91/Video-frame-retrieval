#ifndef __VIDEO_SHOT_RETRIEVAL__
#define __VIDEO_SHOT_RETRIEVAL__

#pragma once
#include "stdafx.h"
#include "BasicFunction.h"

//Calculate feature for each key frame and store all of them in matrix
//Each feature is a vector 1 x featureSize
Mat CalculateFeatureList(vector<Mat> listKeyFrame, int featureSize);

//Calculate distance from a query image to a list of key frame represent a video shot
float CalculateNFLDistance(vector<string> listFileName, Mat queryImage, int featureSize);

//Calculate feature line coefficient from 2 keypoint and a query point
float CalculateFeatureLineCoef(Mat f1, Mat f2, Mat fx);

//Return a list of name of key frame of the same shot
vector<string> GetKeyframeList(vector<string> listFileName, string path, int index);

//Return the name of the shot which contain the query image
void VideoShotRetrieval(string videoName, Mat queryImage);

#endif