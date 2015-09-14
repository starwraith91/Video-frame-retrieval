#ifndef __MACHINE_LEARNING__
#define __MACHINE_LEARNING__

#pragma once
#include "stdafx.h"
#include "KeyframeExtraction.h"
#include "FeatureExtraction.h"

map<float, float> ClassEncoding(string path, Mat labels);

map<float, float> LoadClassCode(string path);

void CreateANNTrainingModel(string path, int inputSize, map<float, float> mapClassEncode, Mat training_data, Mat training_label);

void LoadANNModel(string path, CvANN_MLP &mlp);

#endif