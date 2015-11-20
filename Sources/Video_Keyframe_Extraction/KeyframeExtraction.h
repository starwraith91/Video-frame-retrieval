#ifndef __KEYFRAME_EXTRACTION__
#define __KEYFRAME_EXTRACTION__

#pragma once
#include "stdafx.h"
#include "FeatureExtraction.h"

///----------SUPPORT-METHOD----------///

bool IsHighCurvaturePoint(vector<float> listAngle, int windowSize, int index);

vector<int> CalcCurvatureAnglePoint(vector<float> listDistance, float angleMax, int dMin, int dMax);

///----------MAIN-METHOD-----------///

//Extract shot base on each pair of frames from a summary video into a small video
void VideoShotExtractor(VideoCapture cap,string destinationFolder,string shotpath,int width=0, int height=0);

//Extract key-frame using curvature point detection algorithm
vector<int> KeyframeCurvatureExtractor(VideoCapture cap);

void ExtractAndSaveKeyFrame(string videoName,string categoryName);

#endif