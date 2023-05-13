
#include "stdafx.h"
#include "kernel.cuh"

using namespace std;

int main()
{
    int width = 128;
    int height = 2000;
    dataInfo inputInfo = { width, height, sizeof(INPUT_FORMAT)};
    vector<INPUT_FORMAT> vInput((size_t)width * height, 0);

    for (int i = 0; i < width; i++) {
        for (int j = 0; j < height; j++) {
            vInput[i + (size_t)j * width] = (INPUT_FORMAT)floor(((1 << (sizeof(INPUT_FORMAT) * 8)) - 1)*((float)j)/(float)height);
            //gray gradation to height direction 
        }
    }

    int outWidth = 768;
    int outHeight = 892;
    dataInfo outputInfo = {outWidth, outHeight, sizeof(OUTPUT_FORMAT)};
    vector<OUTPUT_FORMAT> vOutput((size_t)outWidth * outHeight, 0);
    
    ImageParam param = {
        40.f,                                                        //float fSamplingFreqMHz;
        5.f,                                                         //float fCenterFreqMHz;
        0.3f,                                                        //float fPitchMm;
        128.f,                                                       //float fElementNum; // caution : it should be same of width size 
        1540.f,                                                      //float fSpeedOfSoundMps;
        1540.f / 2.f / 40e6f,                                        //float fAxialStepM;
        0.3e-3f,                                                     //float fLeteralStepM;
        fmaxf(1540.f / 2.f / 40e6f * (float)height / (float)outHeight, 
            0.3e-3f * (float)width / (float)outHeight)               //float fGridStepM;
    };

    ScanConversion(&vInput[0], &inputInfo, &vOutput[0], &outputInfo, &param);

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaError_t cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

    return 0;
}
