#ifndef __FUNCTION_H__
#define __FUNCTION_H__
#endif

#include <vector>
#include "frame.h"
#include <iostream>
#include <fstream>
#include "opencv2\core\cv.h"
#include "opencv2\highgui\highgui_c.h"
#include "opencv2\imgproc\imgproc_c.h"
#include "opencv2\highgui\highgui.hpp"
#include <atlstr.h>

using namespace std;
using namespace cv;

IplImage*     ToMeshing( IplImage* pImageSource, int rateCol, int rateRow );

int		      inDexRegion( int i, int rate );

void		  startup( vector< int > &dataReInter, vector<int> &m_regionInterest, vector<int> &m_threshold, vector< int > &m_indexRegion, int level, float percent);

void          diffFrame( Mat m_ImageSource,Mat m_pImageNext, vector< int > &data, vector<int> RegionInterest, vector< int > m_indexRegion );

vector< int > finishResult( vector<frame> frameResult, vector< int > &redundacyOne );

vector< int > eliminateOverlap( vector< int > resultSource );

void          createVideo( vector<frame> m_result, CString m_mfcFolder, char* fileName );

double		  MSECalculator(Mat preFrame, Mat nextFrame);

vector<int>   PositionOfFrameShotDetection(vector<double> ListMSE, int windowLength, int Td);

double        ThresholdMT(vector<double> SubListMSE, int Td);

bool          IsLastFrameMax(vector<double> SubListMSE);

vector<double> FrameInWindow(vector<double> ListMSE, int Pos, int windowLength);

double        Mean(vector<double> SubListMSE);

double        Variance(vector<double> SubListMSE, double Mean);

vector<int>   DetectionShot(char* fileName);

void		  writeVideoSummaryInfor(char* fileName, vector<int> arrIndex);

vector<int>   loadVideoSummaryInfor(char* fileName);

vector< int > findOverLap( vector <int> vectorOne, vector < int > vectorTwo );

vector< int > lookupTable( int cell );

vector< frame > trackObject( int index, vector<int> interCell, vector< vector<int> > frameCellArray );

void addFrame( IplImage* pImageSource, vector< frame > frameArr, int indexResult, int step);

vector<vector<frame>> makeImage( IplImage* pImageSource, vector<int> interCell, vector<int> arrIndex , vector<vector<int>> frameCellArray, int step);

IplImage* get_frame(CvCapture* capture, int frame_idx);

vector<vector<frame>>  eliminateOverlapTrack(vector<vector<frame>> source);