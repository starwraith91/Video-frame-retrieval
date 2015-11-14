#ifndef __VIDEO_SHOT_RETRIEVAL__
#define __VIDEO_SHOT_RETRIEVAL__

#pragma once
#include "stdafx.h"
#include "BasicFunction.h"
#include "FeatureExtraction.h"

//Calculate feature for each key frame and store all of them in matrix
//Each feature is a vector 1 x featureSize
Mat CalculateFeatureList(vector<Mat> listKeyFrame);

//Calculate distance from a query image to a list of key frame represent a video shot
float CalculateNFLDistance(vector<string> listFileName, Mat queryFeature);
float CalculateNFLDistance(vector<string> listFileName, Mat queryFeature);

//Calculate feature line coefficient from 2 keypoint and a query point
float CalcDistanceToFeatureLine(Mat f1, Mat f2, Mat fx);

//Return a list of name of key frame of the same shot
vector<string> GetKeyframeList(vector<string> listFileName, string path, int index);

//Return the name of the shot which contain the query image
//If no data is input, program will extract features from default keyframe folder
void VideoShotRetrieval(string videoName, Mat queryFeature);
void VideoShotRetrieval(string videoName, Mat trainingData, Mat trainingLabel, Mat queryFeature);

//Test system efficiency with a set of test frame from database
void TestDatabase(string categoryName, int database_type);

#endif