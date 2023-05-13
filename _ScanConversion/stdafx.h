#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <vector>
#include <iostream>
#include <exception>
#include <format>
#include <sstream>


struct dataInfo {
	int width;
	int heigth;
	int unitDataSize; // short = 2, uint = 4 ..
};