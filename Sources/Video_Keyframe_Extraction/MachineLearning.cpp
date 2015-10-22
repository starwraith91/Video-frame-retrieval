#pragma once
#include "stdafx.h"
#include "MachineLearning.h"

void CreateANNTrainingModel(string path, int featureSize, Mat training_data, Mat training_label, int numClass)
{
	CvTermCriteria term_crit = cvTermCriteria(CV_TERMCRIT_ITER, 1000, 0.0001);

	CvANN_MLP_TrainParams ANN_Params(term_crit, CvANN_MLP_TrainParams::RPROP, 0.01, FLT_EPSILON);
	//CvANN_MLP_TrainParams ANN_Params(term_crit, CvANN_MLP_TrainParams::BACKPROP, 0.1, 0.1);

//	int numClass = mapClassEncode.size();

	int layer_sz[] = { featureSize, 500, numClass };
	Mat layer_sizes = Mat(1, (int)(sizeof(layer_sz) / sizeof(layer_sz[0])), CV_32S, layer_sz);
	CvANN_MLP mlp(layer_sizes, CvANN_MLP::SIGMOID_SYM, 0.6, 1.0);

	// 1. unroll the responses
	int sampleCount = training_label.rows;

	Mat outputMat(sampleCount, numClass, CV_32FC1, Scalar(0.0));
	for (int i = 0; i < sampleCount; i++)
	{
		int cls_label = (int)training_label.at<float>(i,0);

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

