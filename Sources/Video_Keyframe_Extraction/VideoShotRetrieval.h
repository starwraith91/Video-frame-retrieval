#ifndef __VIDEO_SHOT_RETRIEVAL__
#define __VIDEO_SHOT_RETRIEVAL__

#pragma once
#include "stdafx.h"
#include "BasicFunction.h"
#include "FeatureExtraction.h"

//Calculate feature line coefficient from 2 keypoint and a query point
float CalcDistanceToFeatureLine(Mat f1, Mat f2, Mat fx);

//Return the name of the shot which contain the query image
//If no data is input, program will extract features from default keyframe folder
vector<int> VideoShotRetrieval(string videoName, Mat trainingData, Mat trainingLabel, Mat queryFeature);

//Test system efficiency with a set of test frame from database
void TestDatabase(string categoryName, int database_type);

#endif