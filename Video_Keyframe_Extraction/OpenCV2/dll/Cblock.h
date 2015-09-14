#pragma once
#include <vector>
#include <time.h>
#include <stdlib.h>
#include <iostream>
using namespace std;
class Cblock
{
public:
	Cblock(vector<int> data);
	~Cblock(void);
	vector<vector<int>> data;
	vector<int> toSamples();
};

