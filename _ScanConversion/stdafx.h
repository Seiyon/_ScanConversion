#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
//include helper functions
#include "Common/helper_cuda.h"
#include "Common/helper_functions.h"

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

template<typename T>
void CheckBool(T result, char const* const func, const char* const file, int const line) {

	if (!result) {
		std::stringstream stm;
		stm << "Process Failed at " << file << ", function" << func << ", code line"<< line <<"\n";
		throw std::exception(stm.str().c_str());
	}
#ifdef _DEBUG
	else {
		fprintf(stderr, "Process Success at %s, function %s, code line %d\n", file, func, line);
	}
#endif
}

#define IF_FALSE_ERROR(val) CheckBool((val), #val, __FILE__, __LINE__)
