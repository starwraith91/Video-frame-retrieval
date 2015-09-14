#pragma once
#include "stdafx.h"
#include "KeyframeExtraction.h"

double GetSDOfDistance(vector<KeyFrame> listKeyFrame, double mean)
{
	double sum = 0;
	int n = (int)listKeyFrame.size();
	for (int i = 0; i < n; i++)
	{
		double value = listKeyFrame[i].distCurrentNext;
		sum += (value - mean)*(value - mean);
	}
	return sqrt(sum);
}

///-------NEAREST FEATURE POINT METHOD--------///

vector<float> GetMomentDescriptor(Mat image)
{
	Mat normImage;
	image.convertTo(normImage, CV_32FC3);

	vector<float> featureVector;

	//Get a list of Y values from all pixels
	vector<float> listYValues, listCrValues, listCbValues;
	for (int i = 0; i < normImage.rows; i++)
	{
		for (int j = 0; j < normImage.cols; j++)
		{
			Vec3f value = normImage.at<Vec3f>(i, j);
			listYValues .push_back(value.val[0]);
			listCrValues.push_back(value.val[1]);
			listCbValues.push_back(value.val[2]);
//			cout << value.val[0] << "	" << value.val[1] << "	" << value.val[2] << endl
		}
	}

	float meanY		= GetMean<float>(listYValues);
	float sdY		= GetStandardDeviation<float>(listYValues,meanY);
	float skewY		= GetSkewness<float>(listYValues, meanY);
//	float kurtosisY = GetKurtosis<float>(listYValues, meanY);

//	float meanCr	 = GetMean<float>(listCrValues);
//	float sdCr		 = GetStandardDeviation<float>(listCrValues, meanCr);
//	float skewCr	 = GetSkewness<float>(listCrValues, meanCr);
//	float kurtosisCr = GetKurtosis<float>(listCrValues, meanCr);

//	float meanCb	 = GetMean<float>(listCbValues);
//	float sdCb		 = GetStandardDeviation<float>(listCbValues, meanCb);
//	float skewCb	 = GetSkewness<float>(listCbValues, meanCb);
//	float kurtosisCb = GetKurtosis<float>(listCbValues, meanCb);

	featureVector.push_back(meanY);
	featureVector.push_back(sdY);
	featureVector.push_back(skewY);
//	featureVector.push_back(kurtosisY);

//	featureVector.push_back(meanCb);
//	featureVector.push_back(sdCb);
//	featureVector.push_back(skewCb);
//	featureVector.push_back(kurtosisCb);

//	featureVector.push_back(meanCr);
//	featureVector.push_back(sdCr);
//	featureVector.push_back(skewCr);
//	featureVector.push_back(kurtosisCr);

	NormalizeFeatureVector(featureVector);

	return featureVector;
}

///-------CURVATURE POINT METHOD--------///

KeyFrameDescriptor CalcMPEGDescriptor(Mat img)
{
	KeyFrameDescriptor descriptor;

	Feature *featureExtractor = new Feature();

	Frame *frame = new Frame(img);

	descriptor.colorDesc = featureExtractor->getColorStructureD(frame, 64);
	descriptor.edgeDesc = featureExtractor->getEdgeHistogramD(frame);

	Mat grayImg(img.rows,img.cols,CV_8U);
	Frame *grayFrame = new Frame(img.cols,img.rows);
	cvtColor(img, grayImg, CV_BGR2GRAY);
	grayFrame->setGray(grayImg);
	descriptor.textureDesc = featureExtractor->getHomogeneousTextureD(grayFrame);
	delete grayFrame;

	delete frame;

	delete featureExtractor;

	return descriptor;
}

bool IsHighCurvaturePoint(vector<float> listAngle,int windowSize,int index)
{
	int n = listAngle.size();
	int startIndex = index - windowSize;
	int endIndex   = index + windowSize;
	bool check = true;

	for (int j = startIndex; j <= endIndex; j++)
	{
		if (j == index || j<0 || j>n - 1)
		{
			continue;
		}

		if (listAngle[index] >= listAngle[j])
		{
			check = false;
			break;
		}
	}

	return check;
}

vector<int> CalcCurvatureAnglePoint(vector<float> listDistance, float angleMax, int dMin, int dMax)
{
	vector<float> _listAngle;
	vector<int> keyframeID;
	int n = listDistance.size() - dMin;

	_listAngle.resize(listDistance.size());
	_listAngle[0] = _listAngle[listDistance.size()-1] = 0.0f;


	int leftIndex = 0;
	int rightIndex = 0;
	int currentIndex = 0;
	float alphaLC = angleMax;
	float dOP, dPR, dOR;

	omp_set_num_threads(2);
	#pragma omp parallel for shared(n,dMin,dMax,angleMax,_listAngle,listDistance) private(currentIndex,leftIndex,rightIndex,alphaLC, dOP, dPR, dOR)
	for (currentIndex = dMin; currentIndex < n; currentIndex++)
	{
		alphaLC = angleMax;
		for (leftIndex = currentIndex - dMax; leftIndex <= currentIndex - dMin; leftIndex++)
		{
			if (leftIndex >= 0)
			{
				dOP = sqrtf(pow(listDistance[currentIndex] - listDistance[leftIndex], 2) + (float)pow(currentIndex - leftIndex, 2));
				for (rightIndex = currentIndex + dMin; rightIndex <= currentIndex + dMax; rightIndex++)
				{
					if (rightIndex < n)
					{
						dPR = sqrtf(pow(listDistance[currentIndex] - listDistance[rightIndex], 2) + (float)pow(currentIndex - rightIndex, 2));
						dOR = sqrtf(pow(listDistance[leftIndex] - listDistance[rightIndex], 2) + (float)pow(leftIndex - rightIndex, 2));

						float numerator = dOP*dOP + dPR*dPR - dOR*dOR;
						float denominator = 2 * dOP * dPR;
						float alpha = acosf(numerator / denominator) * 180.0f / (float)M_PI;

						if (alpha < alphaLC)
						{
							alphaLC = alpha;
						}
					}
				}
			}
		}

		if (alphaLC < angleMax)
		{
			_listAngle[currentIndex] = alphaLC;
		}
		else
		{
			_listAngle[currentIndex] = 180.0f;
		}
	}
//	_listAngle.push_back(0.0f);

	//Check for redundant curvature points
	int windowSize = 5;
	int lastIndex = 0;
	int i = windowSize;

	cout << "Select frame: ";
	int numAngle = (int)_listAngle.size();
	while (i < numAngle)
	{
		if (IsHighCurvaturePoint(_listAngle, windowSize, i))
		{
			int index = (lastIndex + i) / 2;
			keyframeID.push_back(index);

			lastIndex = i;

			i += (windowSize+1);

			cout << index << " ";
		}
		else
		{
			i++;
		}
	}
	cout << endl;

	return keyframeID;
}

//Remove redundant keyframe
void RemoveRedundantKeyframes(VideoCapture cap, vector<int> &listKeyframes)
{
	Mat prevEdge, currentEdge;

	cout << "Removed redundant frames ";

	//Calculate edge matching rates
	int numKeyframe = (int)listKeyframes.size();
	for (int i = 0; i < numKeyframe; i++)
	{
		Mat frame = ExtractFrameFromVideo(cap, listKeyframes[i]);
		if (i == 0)
		{
			prevEdge = EdgeDetection(frame);
		}
		else
		{
			currentEdge = EdgeDetection(frame);

			double matchingRate = CalculateEdgeMatchingRate(prevEdge, currentEdge);
			if (matchingRate > 0.6f)
			{
				cout << listKeyframes[i] << " ";
				listKeyframes[i] = -1;
			}

			prevEdge = currentEdge;
		}
	}
	cout << endl;
}

void VideoShotExtractor(VideoCapture cap, string destinationFolder, string shotpath)
{
	ifstream in(shotpath);
	vector<int> listFrameIndex;

	int tempVal = 0;
	do
	{
		in >> tempVal;
		listFrameIndex.push_back(tempVal);
	} while (!in.eof());

	in.close();

	int fps		 = (int)cap.get(CV_CAP_PROP_FPS);
	int numFrame = (int)cap.get(CV_CAP_PROP_FRAME_COUNT);
	int width	 = (int)cap.get(CV_CAP_PROP_FRAME_WIDTH);
	int height	 = (int)cap.get(CV_CAP_PROP_FRAME_HEIGHT);

	int fourcc = static_cast<int>(cap.get(CV_CAP_PROP_FOURCC));

	//	char EXT[] = { (char)(fourcc & 0XFF), (char)((fourcc & 0XFF00) >> 8), (char)((fourcc & 0XFF0000) >> 16), (char)((fourcc & 0XFF000000) >> 24), 0 };
	//	cout << "Input codec type: " << EXT << endl;

	Size sizeFrame(width, height);
	int size[] = { width, height };

	cout << "Number of frames: " << numFrame << endl;

	int lastIndex = 0;
	int numKeyframe = (int)listFrameIndex.size();
	for (int i = 1; i < numKeyframe; i++)
	{
		if (abs(listFrameIndex[i] - listFrameIndex[lastIndex]) >= fps)
		{
			//Init values
			Mat tempFrame;
			int numCount = 0;

			//Video writer
			char *buffer = new char[255];
			string tempStr = _itoa(i, buffer, 10);
			string filename = "Data/Video shots/" + destinationFolder + "/shot_" + tempStr + ".avi";

			cout << "shot_" + tempStr + ".avi" << endl;

			//CV_FOURCC('I', 'Y', 'U', 'V')
			VideoWriter outputWriter(filename, CV_FOURCC('X', 'V', 'I', 'D'), fps, sizeFrame, true);

			for (int j = listFrameIndex[lastIndex]; j < listFrameIndex[i]; j++)
			{
				cap.read(tempFrame);

				if (tempFrame.empty())
					continue;

				outputWriter.write(tempFrame);

				numCount++;

				cout << j << endl;
			}

			lastIndex = i;

			cout << "done" << endl;
		}
	}
}

//Extract key-frame using moments of YCbCr color space
vector<int> KeyframeMomentExtractor(VideoCapture cap)
{
	//Extract all frame from video as a list
	vector<int> keyFrames;

	int numFrame = (int)cap.get(CV_CAP_PROP_FRAME_COUNT);

	if (numFrame <= 2)
	{
		keyFrames.push_back(numFrame/2);
		return keyFrames;
	}

	//Setup parameter to used for Simple Breakpoint method to decide the key-frames
	double _totalDistance = 0.0;
	vector<KeyFrame> _listKeyframeCandidate;

	//Extract Color Structure Descriptor and compute distance to choose candidate key-frames
	Mat currentDescriptor;
	Mat previousDescriptor;

	for (int i = 0; i < numFrame; i++)
	{
		Mat frameMat,normMat;
		cap.read(frameMat); 
		frameMat.convertTo(normMat, CV_32FC3); normMat *= 1.0 / 255.0;
		cvtColor(normMat, normMat, CV_BGR2YCrCb);
		if (i == 0)
		{
			previousDescriptor = ToMat(GetMomentDescriptor(normMat));
		}
		else
		{
			currentDescriptor = ToMat(GetMomentDescriptor(normMat));

			double _distance   = CalcEuclidianDistance(previousDescriptor,currentDescriptor);

			KeyFrame candidate;
			candidate.frameID = i;
			candidate.distCurrentNext = _distance;
			_listKeyframeCandidate.push_back(candidate);

			_totalDistance += _distance;

			//Exchange info and free up memory
			previousDescriptor = currentDescriptor;
		}
	}

	//Calculate threshold and pick the key-frames when distance calculated is greater than threshold
	if (_listKeyframeCandidate.size() != 0)
	{
		double _threshold = _totalDistance / (double)_listKeyframeCandidate.size();
		//double _sd		= GetSDOfDistance(_listKeyframeCandidate, _mean);
		//double _threshold = _mean + _sd/4; 
		cout << "Threshold = " << _threshold << endl;

		int numFrame = (int)_listKeyframeCandidate.size();
		for (int i = 0; i < numFrame; i++)
		{
			if (_listKeyframeCandidate[i].distCurrentNext > _threshold)
			{
				cout << "Selected frame " << _listKeyframeCandidate[i].frameID << endl;
				keyFrames.push_back(_listKeyframeCandidate[i].frameID);
			}
		}

		if (keyFrames.size() == 0)
		{
			keyFrames.push_back(numFrame/2);
			cout << "No key frames is found. Get first frame. ";
		}
	}

	return keyFrames;
}

//Extract key-frame using curvature point detection algorithm
vector<int> KeyframeCurvatureExtractor(VideoCapture cap)
{
	//Extract all frame from video as a list
	vector<int> keyFrames;

	int numFrame = (int)cap.get(CV_CAP_PROP_FRAME_COUNT);

	if (numFrame <= 2)
	{
		keyFrames.push_back(numFrame / 2);
		return keyFrames;
	}

	//Setup parameter to used for curvature point method
	vector<float> _listDistance;
	KeyFrameDescriptor previousDescriptor, currentDescriptor;

	//Extract Color Structure Descriptor and compute distance to choose candidate key-frames
	_listDistance.resize(numFrame - 1);
	for (int i = 0; i < numFrame; i++)
	{
		Mat frameMat;
		cap.read(frameMat);

		if (i == 0)
		{
			previousDescriptor = CalcMPEGDescriptor(frameMat);
		}
		else
		{
			currentDescriptor = CalcMPEGDescriptor(frameMat);

			float dC = currentDescriptor.colorDesc->distance(previousDescriptor.colorDesc);
			float dS = currentDescriptor.edgeDesc->distance(previousDescriptor.edgeDesc);
			float dT = currentDescriptor.textureDesc->distance(previousDescriptor.textureDesc);
			float distance = dC*dS + dS*dT + dT*dC;

			_listDistance[i-1] = distance;

			//Exchange info and free up memory
			previousDescriptor.DeleteDescriptor();
			previousDescriptor = currentDescriptor;
		}
	}
	previousDescriptor.DeleteDescriptor();

//	vector<int> listHighCurvaturePoint;
	float angleMax = 90;
	int dMax = 3;
	int dMin = 1;
	keyFrames = CalcCurvatureAnglePoint(_listDistance, angleMax, dMin, dMax);

	return keyFrames;
}

void ExtractAndSaveKeyFrame(string videoName)
{
	//Extract key frames from a list of shots
	string filepath = "Data/Video shots/" + videoName + "/";
	string destPath = "Data/Key frames/" + videoName + "/";
	vector<string> listFileName = ReadFileList(filepath);

	int j = 0;
	int i = 0;
	int n = (int)listFileName.size();
	int numFrame = 0;
	int count = 0;

//	#pragma omp parallel shared(n) private(i,j,numFrame,count)
	{
//		#pragma omp for
		for (j = 0; j < n; j++)
		{
			string filename = filepath + listFileName[j];

			VideoCapture cap(filename);

			//vector<int> keyFrames = KeyframeMomentExtractor(cap);
			vector<int> keyFrames = KeyframeCurvatureExtractor(cap);

			//RemoveRedundantKeyframes(cap, keyFrames);

			count = 0;
			numFrame = (int)keyFrames.size();

			for (i = 0; i < numFrame; i++)
			{
				if (keyFrames[i] != -1)
				{
					string resultPath = destPath;

					string name = listFileName[j].substr(0, listFileName[j].size() - 4);
					resultPath.append(name);
					resultPath.append("_");

					char *buffer = new char[255];
					resultPath.append(_itoa(keyFrames[i], buffer, 10));
					resultPath.append(".jpg");
					delete buffer;

					Mat frame = ExtractFrameFromVideo(cap, keyFrames[i]);
					if (!frame.empty())
						imwrite(resultPath, frame);
					else
						cout << "Cannot extract keyframe " << i << endl;

					count++;
				}
			}
			cout << "There're " << count << " key frames in " << listFileName[j] << endl;
			cout << endl;
		}
	}
}