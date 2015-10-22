#ifndef __FEATURE_EXTRACTION__
#define __FEATURE_EXTRACTION__

#pragma once

#include "BasicFunction.h"

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
		this->colorDesc = desc.colorDesc;
		this->edgeDesc = desc.edgeDesc;
		this->textureDesc = desc.textureDesc;
		return *this;
	}

	void DeleteDescriptor()
	{
		if (colorDesc != NULL)
			delete colorDesc;

		if (edgeDesc != NULL)
			delete edgeDesc;

		if (textureDesc != NULL)
			delete textureDesc;
	}
};

//--------VIDEO SHOT CLASSIFICATION--------//

Mat ExtractBOWFeature(BOWImgDescriptorExtractor bowDE, SiftFeatureDetector detector, Mat image);

Mat ExtractMPEGFeature(Mat image);

Mat LoadBOWDictionaryFromFile(string filename);

void ClusterFeature(BOWKMeansTrainer bowTrainer, BOWImgDescriptorExtractor &bowDE,string dictionaryName);

void CreateVocaburary(BOWImgDescriptorExtractor &bowDE, int dictionarySize);

void CreateBOWTrainingSet(int dictionarySize, SiftFeatureDetector detector, BOWImgDescriptorExtractor bowDE);

void CreateMPEGTrainingSet();

bool LoadBOWTrainingSet(string path, Mat &training_data, Mat &training_label);

//Extract color moment descriptor
vector<float> GetMomentDescriptor(Mat image);

//Extract color, edge and texture hist
KeyFrameDescriptor CalcMPEGDescriptor(Mat img);

//---------BASIC IMAGE PROCESSING-----------//

Mat EdgeDetection(Mat img);

int CountEdgePixel(Mat imgEdge);

double CalculateEdgeMatchingRate(Mat imgEdge1, Mat imgEdge2);

#endif