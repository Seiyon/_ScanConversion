#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <vector>
#include <iostream>
#include <exception>
#include <sstream>
#include <chrono>
#include <fstream>

#define INPUT_FORMAT unsigned char
#define OUTPUT_FORMAT unsigned char
#define TIME_FORMAT double

struct ImageParam {
	float fSamplingFreqHz;
	float fCenterFreqHz;
	float fPitchM;
	float fElementNum; // caution : it should be same of height size which is lateral direction 
	float fSpeedOfSoundMps;
	float fAxialStepM;
	float fLeteralStepM;
	float fGridStepM;
};

struct dataInfo {
	int iWidth;
	int iHeigth;
	int iUnitDataSize; //char(byte) = 1, short = 2, float = 4, double = 8 
};