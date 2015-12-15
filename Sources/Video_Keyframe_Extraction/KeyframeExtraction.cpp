#pragma once
#include "stdafx.h"
#include "KeyframeExtraction.h"

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

//	omp_set_num_threads(4);
	#pragma omp parallel for shared(n,dMin,dMax,angleMax,_listAngle,listDistance) private(currentIndex,leftIndex,rightIndex)
	for (currentIndex = dMin; currentIndex < n; currentIndex++)
	{
		float alphaLC = angleMax;
		for (leftIndex = currentIndex - dMax; leftIndex <= currentIndex - dMin; leftIndex++)
		{
			if (leftIndex >= 0)
			{
				float left2 = (listDistance[currentIndex] - listDistance[leftIndex])*(listDistance[currentIndex] - listDistance[leftIndex]);
				float leftIndex2 = (float)(currentIndex - leftIndex)*(currentIndex - leftIndex);
				float dOP = left2 + leftIndex2;

				for (rightIndex = currentIndex + dMin; rightIndex <= currentIndex + dMax; rightIndex++)
				{
					if (rightIndex < n)
					{
						float right2 = (listDistance[currentIndex] - listDistance[rightIndex])*(listDistance[currentIndex] - listDistance[rightIndex]);
						float rightIndex2 = (float)(currentIndex - rightIndex)*(currentIndex - rightIndex);
						float dPR = right2 + rightIndex2;

						float boundary2 = (listDistance[leftIndex] - listDistance[rightIndex])*(listDistance[leftIndex] - listDistance[rightIndex]);
						float boundaryIndex2 = (float)(leftIndex - rightIndex)*(leftIndex - rightIndex);
						float dOR = boundary2 + boundaryIndex2;

						//float numerator = dOP*dOP + dPR*dPR - dOR*dOR;
						float numerator = dOP + dPR - dOR;
						float denominator = 2 * sqrtf(dOP) * sqrtf(dPR);
						float fraction = Clampf(numerator/denominator, -1.0f, 1.0f);

						float value = acosf(fraction) * 180.0f / (float)M_PI;
						float alpha = (int)(value + 0.5f);

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

void VideoShotExtractor(VideoCapture cap, string destinationFolder, string shotpath,int width,int height)
{
	ifstream in(shotpath);
	vector<int> listFrameIndex;

	string foldername = "Data/Video_shots/" + destinationFolder;
	if (CreateDirectoryA(foldername.c_str(), NULL) || ERROR_ALREADY_EXISTS == GetLastError())
	{
		int tempVal = 0;
		do
		{
			in >> tempVal;
			listFrameIndex.push_back(tempVal);
		} while (!in.eof());

		in.close();

		int fps		 = (int)cap.get(CV_CAP_PROP_FPS);
		int numFrame = (int)cap.get(CV_CAP_PROP_FRAME_COUNT);

		int w=0, h=0;
		if (width != 0 && height != 0)
		{
			w = width;
			h = height;
		}
		else
		{
			w = (int)cap.get(CV_CAP_PROP_FRAME_WIDTH);
			h = (int)cap.get(CV_CAP_PROP_FRAME_HEIGHT);
		}

		int fourcc	 = static_cast<int>(cap.get(CV_CAP_PROP_FOURCC));

		//	char EXT[] = { (char)(fourcc & 0XFF), (char)((fourcc & 0XFF00) >> 8), (char)((fourcc & 0XFF0000) >> 16), (char)((fourcc & 0XFF000000) >> 24), 0 };
		//	cout << "Input codec type: " << EXT << endl;

		Size sizeFrame(w, h);
		int size[] = { w, h };

		cout << "Number of frames: " << numFrame << endl;

		int lastIndex = 0;
		int realIndex = 1;
		int numKeyframe = (int)listFrameIndex.size();
		for (int i = 1; i < numKeyframe; i++)
		{
			if (abs(listFrameIndex[i] - listFrameIndex[lastIndex]) >= fps)
			{
				//Init values
				Mat tempFrame;

				int startID = listFrameIndex[lastIndex];
				if (listFrameIndex[lastIndex] > 0)
					startID++;

				//Video writer
				char *buffer = new char[255];
				string shotID	   = _itoa(realIndex, buffer, 10);
				string shotStartID = _itoa(startID, buffer, 10);
				string filename    = foldername + "/shot_" + shotID + "_" + shotStartID + ".avi";

				cout << "shot_" + shotID + ".avi" << endl;

				//CV_FOURCC('I', 'Y', 'U', 'V')
				VideoWriter outputWriter(filename, CV_FOURCC('X', 'V', 'I', 'D'), fps, sizeFrame, true);

				for (int j = startID; j <= listFrameIndex[i]; j++)
				{
					cap.read(tempFrame);

					if (tempFrame.empty())
						continue;

					Mat resizedFrame;
					resize(tempFrame, resizedFrame, sizeFrame);

					outputWriter.write(resizedFrame);

					cout << j << endl;
				}

				//getchar();

				lastIndex = i;

				realIndex++;

				cout << "done" << endl;
			}

		}
	}

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

	clock_t t;

	t = clock();

	//Extract MPEG-7 feature Descriptor and compute distance to choose candidate key-frames
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

	t = clock() - t;

	cout << "It took " << ((float)t) / CLOCKS_PER_SEC << " seconds to calculate all distances" << endl;

//	vector<int> listHighCurvaturePoint;
	float angleMax = 90; //60;
	int dMax = 3;
	int dMin = 1;
	keyFrames = CalcCurvatureAnglePoint(_listDistance, angleMax, dMin, dMax);

	return keyFrames;
}

void ExtractAndSaveKeyFrame(string videoName, string categoryName="")
{
	//Extract key frames from a list of shots
	string filepath = "Data/Video_shots/" + categoryName + "/" + videoName + "/";
	string destPath = "Data/Keyframes/" + categoryName + "/" + videoName + "/";
	vector<string> listFileName = ReadFileList(filepath);

	if ( !CreateDirectoryA(destPath.c_str(), NULL) && ERROR_ALREADY_EXISTS != GetLastError())
	{
		cout << "Cannot create folder" << endl;
		return;
	}

	int j = 0;
	int i = 0;
	int n = (int)listFileName.size();
	int numFrame = 0;
	int count = 0;

	//omp_set_num_threads(2);
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

					char buffer[21];
					resultPath.append(_itoa(keyFrames[i], buffer, 10));
					resultPath.append(".jpg");

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