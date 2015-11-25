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

bool myfunction(string i, string j) { return (i<j); }

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
			}
		}
		closedir(dir);
	}
	else
	{
		/* could not open directory */
		cout << "Cannot open directory" << endl;
	}

	//sort(_listStringName.begin(), _listStringName.end(), myfunction);

	return _listStringName;
}

int IdentifyShotFromKeyFrame(string filename)
{
	int shotID = -1;

	int lc_right = 0, lc_left = 0;
	int countLC = 0;
	for (int characterIndex = 0; characterIndex < filename.size(); characterIndex++)
	{
		if (filename[characterIndex] == '_')
		{
			if (countLC == 0)
			{
				lc_left = characterIndex + 1;
			}
			else
			{
				lc_right = characterIndex;
				string tempStr = filename.substr(lc_left, lc_right - lc_left);
				shotID = atoi(tempStr.c_str());

				break;
			}
			countLC++;
		}
	}

	return shotID;
}

int IdentifyStartIDFromKeyFrame(string filename)
{
	int shotID = -1;

	int lc_right = 0, lc_left = 0;
	int countLC = 0;
	for (int characterIndex = 0; characterIndex < filename.size(); characterIndex++)
	{
		if (filename[characterIndex] == '_')
		{
			if (countLC == 1)
			{
				lc_left = characterIndex + 1;
			}
			else if (countLC == 2)
			{
				lc_right = characterIndex;
				string tempStr = filename.substr(lc_left, lc_right - lc_left);
				shotID = atoi(tempStr.c_str());

				break;
			}
			countLC++;
		}
	}

	return shotID;
}

int IdentifyKeyIDFromKeyFrame(string filename)
{
	int shotID = -1;

	int lc_right = 0, lc_left = 0;
	int countLC = 0;
	for (int characterIndex = 0; characterIndex < filename.size(); characterIndex++)
	{
		if (filename[characterIndex] == '_')
		{
			if (countLC == 2)
				lc_left = characterIndex + 1;
			countLC++;
		}
		else if (filename[characterIndex] == '.')
		{
			lc_right = characterIndex;
			string tempStr = filename.substr(lc_left, lc_right - lc_left);
			shotID = atoi(tempStr.c_str());

			break;
		}
	}

	return shotID;
}

string GetFileNameExtension(string filename)
{
	int count = 1;
	for (int i = filename.size()-1; i >= 0; i--)
	{
		if (filename[i] == '.')
		{
			return filename.substr(i,count);
		}
		else
		{
			count++;
		}
	}
	return NULL;
}

void DeleteAllFiles(string directoryPath)
{
	vector<string> listFiles = ReadFileList(directoryPath);
	for (int i = 0; i < listFiles.size(); i++)
	{
		string filepath = directoryPath + "/" + listFiles[i];
		remove(filepath.c_str());
	}
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

float CalcEuclideanDistance(Mat a, Mat b)
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

float CalcDistanceFromSet(Mat a, Mat featureMatrix)
{
	float minDistance = -1;
	for (int i = 0; i < featureMatrix.rows; i++)
	{
		Mat b = featureMatrix.row(i);
		float distance = CalcEuclideanDistance(a, b);
		if (minDistance == -1 || minDistance > distance)
		{
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