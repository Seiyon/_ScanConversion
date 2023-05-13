
#include "stdafx.h"
#include "kernel.cuh"

using namespace std;

int main()
{
    int width = 512;
    int height = 512;
    dataInfo inputInfo = { width, height, sizeof(char)};
    vector<char> vInput((size_t)width * height, 0);

    for (int i = 0; i < width; i++) {
        for (int j = 0; j < height; j++) {
            vInput[(size_t)i + j * width] = (char)floor(255*((float)j)/(float)height);
            //gray gradation to height direction 
        }
    }

    int outWidth = 768;
    int outHeight = 892;
    dataInfo outputInfo = {outWidth, outHeight, sizeof(char)};
    vector<char> vOutput((size_t)outWidth * outHeight, 0);
    
    ScanConversion(&vInput[0], &inputInfo, &vOutput[0], &outputInfo);

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaError_t cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

    return 0;
}
