#ifndef __FEATURE_EXTRACTION__
#define __FEATURE_EXTRACTION__

#pragma once

#include "BasicFunction.h"

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

void ClusterFeature(string categoryName, BOWKMeansTrainer bowTrainer, BOWImgDescriptorExtractor &bowDE, string dictionaryName);

void CreateVocaburary(string categoryName, BOWImgDescriptorExtractor &bowDE, int dictionarySize);

void CreateBOWTrainingSet(string categoryName, int dictionarySize, SiftFeatureDetector detector, BOWImgDescriptorExtractor bowDE);

void CreateMPEGTrainingSet(string categoryName);

bool LoadDataFromFile(string path, Mat &training_data, Mat &training_label, float &videoFPS);

//Extract color, edge and texture hist
KeyFrameDescriptor CalcMPEGDescriptor(Mat img);

#endif