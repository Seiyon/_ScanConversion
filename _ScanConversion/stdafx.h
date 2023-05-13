#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <vector>
#include <iostream>
#include <exception>
#include <sstream>

#define INPUT_FORMAT char
#define OUTPUT_FORMAT char

struct ImageParam {
	float fSamplingFreqMHz;
	float fCenterFreqMHz;
	float fPitchMm;
	float fElementNum; // caution : it should be same of width size 
	float fSpeedOfSoundMps;
	float fAxialStepM;
	float fLeteralStepM;
	float fGridStepM;
};

struct dataInfo {
	int iWidth;
	int iHeigth;
	int iUnitDataSize; // short = 2, uint = 4 ..
};