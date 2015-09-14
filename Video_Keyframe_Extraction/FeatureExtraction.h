#ifndef __FEATURE_EXTRACTION__
#define __FEATURE_EXTRACTION__

#pragma once

#include "BasicFunction.h"

//--------VIDEO SHOT CLASSIFICATION--------//

Mat ExtractSURFescriptor(string path);

Mat ExtractSIFTDescriptor(string path);

Mat ExtractBOWFeature(BOWImgDescriptorExtractor bowDE, SurfFeatureDetector detector, Mat image);

Mat LoadBOWDictionaryFromFile(string filename);

void ClusterFeature(BOWKMeansTrainer bowTrainer, BOWImgDescriptorExtractor &bowDE,string dictionaryName);

void CreateVocaburary(BOWImgDescriptorExtractor &bowDE, int dictionarySize);

void CreateBOWTrainingSet(int dictionarySize, SurfFeatureDetector detector, BOWImgDescriptorExtractor bowDE);

void LoadBOWTrainingSet(string path, Mat &training_data, Mat &training_label);

//---------BASIC IMAGE PROCESSING-----------//

Mat EdgeDetection(Mat img);

int CountEdgePixel(Mat imgEdge);

double CalculateEdgeMatchingRate(Mat imgEdge1, Mat imgEdge2);

#endif