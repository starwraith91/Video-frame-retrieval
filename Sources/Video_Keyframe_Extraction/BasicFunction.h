#ifndef __BASIC_FUNCTION__
#define __BASIC_FUNCTION__

#pragma once

//Generic header
#include "stdafx.h"
#include "dirent.h"

//IO header
#include <fstream>
#include <iostream>
#include <vector>
#include <map>
#include <string>
#include <time.h>
#include <set>
#include <algorithm>
#include <Windows.h>

//OpenCV header
#include "core.hpp"
#include "highgui.hpp"
#include "imgproc.hpp"
#include "features2d.hpp"
#include "nonfree.hpp"
#include "objdetect.hpp"
#include "ml.hpp"

//MPEG7FexLib header
#include "Feature.h"

#define _USE_MATH_DEFINES 
#define ROUND(a) (int)(a+0.5)
#include <math.h>
#include <omp.h>

using namespace std;
using namespace cv;
using namespace XM;

//------BASIC_FUNCTION------//

vector<string> ReadFileList(string path);

void ShowDict(map<string, int> _dict);

string GetName(string filename);

int IdentifyShotFromKeyFrame(string filename);

int IdentifyStartIDFromKeyFrame(string filename);

int IdentifyKeyIDFromKeyFrame(string filename);

string GetFileNameExtension(string filename);

void DeleteAllFiles(string directoryPath);

//----MATHEMATICAL_FUNCTION----//

void NormalizeFeatureVector(vector<float> &listValue);

float GetMagnitude(vector<float> &listValue);

//----TEMPLATE_FUNCTION----//

template<class T>
T GetMedian(vector<T> listValue)
{
	for (int i = 0; i < (int)listValue.size() - 1; i++)
	{
		for (int j = i + 1; j < (int)listValue.size(); j++)
		{
			T temp = listValue[i];
			listValue[i] = listValue[j];
			listValue[j] = temp;
		}
	}

	int centerIndex = (int)listValue.size() / 2;
	if (listValue.size() % 2 == 0)
	{
		return (listValue[centerIndex] + listValue[centerIndex + 1]) / 2.0;
	}
	else
	{
		return listValue[centerIndex];
	}
}

template<class T>
float GetMean(vector<T> listValue)
{
	float sum = 0;

	for (int i = 0; i < (int)listValue.size(); i++)
	{
		float tempVal = (float)listValue[i];
		sum += tempVal;
	}

	return (sum / (float)listValue.size());
}

template<class T>
float GetStandardDeviation(vector<T> listValue, float mean)
{
	float sum = 0;

	for (int i = 0; i < (int)listValue.size(); i++)
	{
		float tempVal = (float)listValue[i] - mean;
		sum += tempVal*tempVal;
	}

	sum /= (float)listValue.size();

	return sqrt(sum);
}

template<class T>
float GetSkewness(vector<T> listValue, float mean)
{
	float sum = 0;

	for (int i = 0; i < (int)listValue.size(); i++)
	{
		float tempVal = (float)listValue[i] - mean;
		sum += tempVal*tempVal*tempVal;
	}

	sum /= (float)listValue.size();

	return cbrt(sum);
}

template<class T>
float GetKurtosis(vector<T> listValue, float mean)
{
	float sum = 0;

	for (int i = 0; i < (int)listValue.size(); i++)
	{
		float tempVal = (float)listValue[i] - mean;
		sum += tempVal*tempVal*tempVal*tempVal;
	}

	sum /= (float)listValue.size();

	return pow(sum,1.0/4.0);
}

template <class T>
void Sort(vector<T> &listValue, bool ascending = false)
{
	for (int i = 0; i < (int)listValue.size() - 1; i++)
	{
		for (int j = i; j < (int)listValue.size(); j++)
		{
			if (ascending)
			{
				if (listValue[i] < listValue[j])
				{
					T temp = listValue[i];
					listValue[i] = listValue[j];
					listValue[j] = temp;
				}
			}
			else
			{
				if (listValue[i] > listValue[j])
				{
					T temp = listValue[i];
					listValue[i] = listValue[j];
					listValue[j] = temp;
				}
			}
		}
	}
}

template <class T, class X>
void Sort(vector<T> &listValue, vector<X> &listSubValue, bool ascending=false)
{
	for (int i = 0; i < (int)listValue.size() - 1; i++)
	{
		for (int j = i; j < (int)listValue.size(); j++)
		{
			if (!ascending)
			{
				if (listValue[i] < listValue[j])
				{
					T temp = listValue[i];
					listValue[i] = listValue[j];
					listValue[j] = temp;

					X tempSub = listSubValue[i];
					listSubValue[i] = listSubValue[j];
					listSubValue[j] = tempSub;
				}
			}
			else
			{
				if (listValue[i] > listValue[j])
				{
					T temp = listValue[i];
					listValue[i] = listValue[j];
					listValue[j] = temp;

					X tempSub = listSubValue[i];
					listSubValue[i] = listSubValue[j];
					listSubValue[j] = tempSub;
				}
			}
		}
	}
}

template <class T>
void Shuffle(T *arr, size_t n)
{
	if (n > 1)
	{
		size_t i;
		for (i = n-1; i>0; i--)
		{
			size_t j = rand() % i;
			T t = arr[j];
			arr[j] = arr[i];
			arr[i] = t;
		}
	}
}

float Clampf(float value, float minValue, float maxValue);

Mat ToMat(vector<float> vec);

vector<float> ToVector(Mat mat);

//-------BASIC_IMAGE_PROCESSING-----//

float CalcVectorMagnitude(Mat mat);

float CalcEuclideanDistance(Mat a, Mat b);

float CalcDistanceFromSet(Mat a, Mat featureMatrix);

float CalcEntropy(float value,float total);

//Extract a single frame from a video
Mat ExtractFrameFromVideo(VideoCapture cap, int frameID);

#endif
