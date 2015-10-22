#include "stdafx.h"
#include "BasicFunction.h"

void ShowDict(map<string, int> _dict)
{
	map<string, int>::iterator it;
	for (it = _dict.begin(); it != _dict.end(); it++)
	{
		cout << "(" << it->first << "," << it->second << ")" << endl;
	}
}

vector<string> ReadFileList(string path)
{
	DIR *dir;
	struct dirent *ent;
	vector<string> _listStringName;
	if ((dir = opendir(path.c_str())) != NULL)
	{
		/* print all the files and directories within directory */
		while ((ent = readdir(dir)) != NULL)
		{
			char *pName = ent->d_name;
			if (strcmp(pName, "..") != 0 && strcmp(pName, ".") != 0)
			{
				string tempString(pName);
				_listStringName.push_back(tempString);

				//printf("%s\n", ent->d_name);
			}
		}
		closedir(dir);
	}
	else
	{
		/* could not open directory */
		cout << "Cannot open directory" << endl;
	}

	return _listStringName;
}

int IdentifyShotFromKeyFrame(string filename)
{
	int shotID = -1;

	int lc_right = 0, lc_left = 0;
	int countLC = 0;
	for (int characterIndex = filename.size() - 1; characterIndex >= 0; characterIndex--)
	{
		if (filename[characterIndex] == '_')
		{
			if (countLC == 0)
			{
				lc_right = characterIndex;
			}
			else
			{
				lc_left = characterIndex + 1;
				string tempStr = filename.substr(lc_left, lc_right - lc_left);
				shotID = atoi(tempStr.c_str());

				break;
			}
			countLC++;
		}
	}

	return shotID;
}

Mat GetColorStructureDescriptor(Mat image, int featureSize)
{
	//Initialize feature extractor
	Feature featureExtractor;

	//Import image to a new structure
	Frame *frame = new Frame(image);

	//Extract color structure descriptor
	ColorStructureDescriptor *descriptor = featureExtractor.getColorStructureD(frame, featureSize);

	Mat feature(1, featureSize, CV_32FC1);
	for (int i = 0; i < featureSize; i++)
	{
		feature.at<float>(0, i) = (float)descriptor->GetElement(i);
	}

	delete frame;

	return feature;
}

float GetMagnitude(vector<float> &listValue)
{
	float sum = 0;
	for (int i = 0; i < (int)listValue.size(); i++)
	{
		float tempVal = listValue[i];
		sum += tempVal*tempVal;
	}
	return sqrt(sum);
}


void NormalizeFeatureVector(vector<float> &listValue)
{
	float vectorLength = GetMagnitude(listValue);
	for (int i = 0; i < (int)listValue.size(); i++)
	{
		listValue[i] /= (float)vectorLength;
	}
}

float CalcVectorMagnitude(Mat mat)
{
	float sum = 0;
	for (int i = 0; i < mat.cols; i++)
	{
		float tempVal = mat.at<float>(0, i);
		sum += (tempVal*tempVal);
	}
	return sqrt(sum);
}

float CalcEuclidianDistance(Mat a, Mat b)
{
	float totalValue = 0;
	for (int i = 0; i < a.cols; i++)
	{
		float tempA = a.at<float>(cvPoint(i, 0));
		float tempB = b.at<float>(cvPoint(i, 0));
		float dist = (tempA - tempB)*(tempA - tempB);

		totalValue += dist;
	}

	//return sqrt(totalValue);
	return totalValue;
}

float CalcDistanceFromSet(Mat a, Mat featureMatrix,int &index)
{
	float minDistance = -1;
	for (int i = 0; i < featureMatrix.rows; i++)
	{
		Mat b = featureMatrix.row(i);
		float distance = CalcEuclidianDistance(a, b);
		if (minDistance == -1 || minDistance > distance)
		{
			index = i;
			minDistance = distance;
		}
	}
	return minDistance;
}

Mat ToMat(vector<float> vec)
{
	int numCols = (int)vec.size();
	Mat matFeature(1, numCols, CV_32FC1);
	for (int i = 0; i < numCols; i++)
	{
		matFeature.at<float>(0, i) = vec[i];
	}
	return matFeature;
}

vector<float> ToVector(Mat mat)
{
	vector<float> vectorFeature;
	for (int i = 0; i < mat.cols; i++)
	{
		vectorFeature.push_back(mat.at<float>(cvPoint(i, 0)));
	}
	return vectorFeature;
}

float Clampf(float value, float minValue, float maxValue)
{
	float resultVal = value;

	if (resultVal < minValue)
	{
		resultVal = minValue;
	}
	else if (resultVal > maxValue)
	{
		resultVal = maxValue;
	}

	return resultVal;
}

float CalcEntropy(float value, float total)
{
	if (value > 0)
	{
		float colorProportion = value / total;
		return -colorProportion*log2(colorProportion);
	}
	return 0;
}

Mat ExtractFrameFromVideo(VideoCapture cap, int frameID)
{
	Mat frame;
	cap.set(CV_CAP_PROP_POS_FRAMES, (double)frameID);
	cap.read(frame);

	return frame;
}