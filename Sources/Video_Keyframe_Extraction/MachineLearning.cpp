#pragma once
#include "stdafx.h"
#include "MachineLearning.h"

//map<float, float> ClassEncoding(string path, Mat labels)
//{
//	map<float, float> mapClassName;
//
//	//Count total number of different class in output
//	float previousValue = labels.at<float>(0, 0);
//
//	//Init map value
//	float k = 0;
//	mapClassName.insert( pair<float, float>(previousValue, k) );
//
//	for (int i = 1; i < labels.rows; i++)
//	{
//		float currentValue = labels.at<float>(i, 0);		
//		if (currentValue != previousValue)
//		{
//			k++;
//			mapClassName.insert(pair<float, float>(currentValue, k));
//		}
//		previousValue = currentValue;
//	}
//
//	ofstream out(path);
//	map<float, float>::iterator index;
//	for (index = mapClassName.begin(); index != mapClassName.end(); index++)
//	{
//		out << index->first << "	" << index->second << endl;
//	}
//	out.close();
//
//	return mapClassName;
//}
//
//map<float, float> LoadClassCode(string path)
//{
//	ifstream in(path);
//
//	map<float, float> mapClassName;
//
//	while (!in.eof())
//	{
//		float key, value;
//		in >> key;
//		in >> value;
//		mapClassName.insert( pair<float, float>(key, value) );
//
//		cout << "( " << key << " , " << value << " )" << endl;
//	}
//
//	in.close();
//
//	return mapClassName;
//}

void CreateANNTrainingModel(string path, int featureSize, Mat training_data, Mat training_label, int numClass)
{
	CvTermCriteria term_crit = cvTermCriteria(CV_TERMCRIT_ITER, 1000, 0.001);

	CvANN_MLP_TrainParams ANN_Params(term_crit, CvANN_MLP_TrainParams::RPROP, 0.01, FLT_EPSILON);

//	int numClass = mapClassEncode.size();

	int layer_sz[] = { featureSize, 500, numClass };
	Mat layer_sizes = Mat(1, (int)(sizeof(layer_sz) / sizeof(layer_sz[0])), CV_32S, layer_sz);
	CvANN_MLP mlp(layer_sizes, CvANN_MLP::SIGMOID_SYM);

	// 1. unroll the responses
	int sampleCount = training_label.rows;

	Mat outputMat(sampleCount, numClass, CV_32FC1, Scalar(0.0));
	for (int i = 0; i < sampleCount; i++)
	{
		float cls_label = training_label.at<float>(i,0);

//		int classIndex = (int)mapClassEncode[cls_label];

		outputMat.at<float>(i, cls_label) = 1.0f;

//		classIndex = 0;
	}

	ofstream out("Data/training_label.txt");
	for (int row = 0; row < outputMat.rows; row++)
	{
		for (int col = 0; col < outputMat.cols; col++)
		{
			Point point(col, row);
			out << outputMat.at<float>(point);
		}
		out << endl;
	}
	out.close();

	mlp.train(training_data, outputMat, cv::Mat(), cv::Mat(), ANN_Params);

	mlp.save(path.c_str());
}

void LoadANNModel(string path, CvANN_MLP &mlp)
{
	mlp.clear();

	mlp.load( path.c_str() );
}

