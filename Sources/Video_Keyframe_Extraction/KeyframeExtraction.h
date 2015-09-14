#ifndef __KEYFRAME_EXTRACTION__
#define __KEYFRAME_EXTRACTION__

#pragma once
#include "stdafx.h"
#include "FeatureExtraction.h"

struct KeyFrame
{
	int frameID;
	double distCurrentNext;
	double distLastNext;
};

struct KeyFrameDescriptor
{
	ColorStructureDescriptor *colorDesc;
	EdgeHistogramDescriptor *edgeDesc;
	HomogeneousTextureDescriptor *textureDesc;

	KeyFrameDescriptor()
	{
		colorDesc = NULL;
		edgeDesc = NULL;
		textureDesc = NULL;
	}

	~KeyFrameDescriptor()
	{
		//DeleteDescriptor();
	}

	KeyFrameDescriptor& operator = (KeyFrameDescriptor desc)
	{
		this->colorDesc   = desc.colorDesc;
		this->edgeDesc	  = desc.edgeDesc;
		this->textureDesc = desc.textureDesc;
		return *this;
	}

	void DeleteDescriptor()
	{
		if (colorDesc!=NULL)
			delete colorDesc;

		if (edgeDesc != NULL)
			delete edgeDesc;

		if (textureDesc != NULL)
			delete textureDesc;
	}
};

///----------SUPPORT-METHOD----------///

double GetSDOfDistance(vector<KeyFrame> listKeyFrame, double mean);

vector<float> GetMomentDescriptor(Mat image);

//Extract color, edge and texture hist
KeyFrameDescriptor CalcMPEGDescriptor(Mat img);

bool IsHighCurvaturePoint(vector<float> listAngle, int windowSize, int index);

vector<int> CalcCurvatureAnglePoint(vector<float> listDistance, float angleMax, int dMin, int dMax);

///----------MAIN-METHOD-----------///

//Remove redundant keyframe
void RemoveRedundantKeyframes(VideoCapture cap, vector<Mat> &listKeyframes);

//Extract shot base on each pair of frames from a summary video into a small video
void VideoShotExtractor(VideoCapture cap, string destinationFolder, string shotpath);

//Extract key-frame using moments of YCbCr color space
vector<int> KeyframeMomentExtractor(VideoCapture cap);

//Extract key-frame using curvature point detection algorithm
vector<int> KeyframeCurvatureExtractor(VideoCapture cap);

void ExtractAndSaveKeyFrame(string videoName);

#endif