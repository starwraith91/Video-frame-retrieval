#ifndef __FRAME_CLASSIFICATION__
#define __FRAME_CLASSIFICATION__

#pragma once
#include "stdafx.h"
#include "BasicFunction.h"

void CreateTrainingModel(string path);

int ImageClassification(Mat image);


#endif