
#include "stdafx.h"
#include "interface.h"

using std::string, std::vector, std::stringstream, std::ifstream, std::ofstream, std::ios, std::function, std::cout;

string GetStringFormat(const dataInfo* const info);
bool ReadBinFile(string filePath, INPUT_FORMAT** _data, int* datalen);
void TestPerformanceFunction(
    const INPUT_FORMAT* const input,
    const dataInfo* const inputInfo,
    OUTPUT_FORMAT* const output,
    const dataInfo* const outputInfo,
    const ImageParam* const param,
    const int count,
    CALLBACK_TYPE callback,
    string title = ""
);

int main()
{
    int width = 500; // the number of decimated sample (2000 / 4)
    int height = 128; // the number of element
    dataInfo inputInfo = { width, height, sizeof(INPUT_FORMAT)};
    vector<INPUT_FORMAT> vInput((size_t)width * height, 0);

    // generate example data
    for (int i = 0; i < width; i++) {
        for (int j = 0; j < height; j++) {
            vInput[i + (size_t)j * width] = (INPUT_FORMAT)floor(((1 << 8) - 1)*((float)j)/(float)height);
            //gray gradation to height direction 
        }
    }

    int outWidth = 892;
    int outHeight = 768;
    dataInfo outputInfo = {outWidth, outHeight, sizeof(OUTPUT_FORMAT)};
    vector<OUTPUT_FORMAT> vOutput((size_t)outWidth * outHeight, 0);
    
    float fsHz = 10e6f; //decimated sampling frequency (40MHz / 4)
    float fcHz = 5e6f;
    float pitchM = 0.3e-3f;
    float el = 128.f;
    float c = 1540.f;
    float axialStep = c / 2.f / fsHz;
    float lateralStep = pitchM;
    float gridStep = fmaxf(axialStep * (float)width / (float)outWidth,
                            pitchM * (float)height / (float)outHeight);

    ImageParam param = {
        fsHz,           //float fSamplingFreqMHz;
        fcHz,           //float fCenterFreqMHz;
        pitchM,         //float fPitchMm;
        el,             //float fElementNum; // caution : it should be same of width size 
        c,              //float fSpeedOfSoundMps;
        axialStep,      //float fAxialStepM;
        lateralStep,    //float fLeteralStepM;
        gridStep        //float fGridStepM;
    };

    CALLBACK_TYPE scanConvFunc = ScanConversion;
    CALLBACK_TYPE scanConvTextFunc = ScanConversionTexture;

    int testCount = 100;
    //ScanConversion(&vInput[0], &inputInfo, &vOutput[0], &outputInfo, &param);
    //ScanConversionTexture(&vInput[0], &inputInfo, &vOutput[0], &outputInfo, &param);

    //double ProcessTimeMicroSecond = ScanConversionTexture(&vInput[0], &inputInfo, &vOutput[0], &outputInfo, &param);
    TestPerformanceFunction(&vInput[0], &inputInfo, &vOutput[0], &outputInfo, &param, testCount, ScanConversion,
        "ScanConversion");
    TestPerformanceFunction(&vInput[0], &inputInfo, &vOutput[0], &outputInfo, &param, testCount, ScanConversionTexture,
        "ScanConvertionTexture");

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    checkCudaErrors(cudaDeviceReset());

    //write data to project folder
    IF_FALSE_ERROR(WriteBinFile(GetStringFormat(&inputInfo), &vInput[0], GetTotalSize(&inputInfo)));
    IF_FALSE_ERROR(WriteBinFile(GetStringFormat(&outputInfo), &vOutput[0], GetTotalSize(&outputInfo)));

    return 0;
}

string GetStringFormat(const dataInfo * const info) {
    stringstream stm;
    stm << "capture_" << info->iWidth << "x" << info->iHeigth << "_imageType_" <<
        ((info->iUnitDataSize == 1) ? "8-bit" :
            (info->iUnitDataSize == 2) ? "16-bit_signed" :
            (info->iUnitDataSize == 4) ? "32-bit_real" :
            (info->iUnitDataSize == 8) ? "64-bit_real" : "unkown") << ".raw";
    return stm.str();
}

bool ReadBinFile(string filePath, INPUT_FORMAT** _data, int* datalen)
{
    ifstream is(filePath, ifstream::binary);
    if (is) {
        // seekg를 이용한 파일 크기 추출
        is.seekg(0, is.end);
        int length = (int)is.tellg();
        is.seekg(0, is.beg);

        // malloc으로 메모리 할당
        char* buffer = (char*)malloc(length);

        // read data as a block:
        is.read((char*)buffer, length);
        is.close();
        *_data = (INPUT_FORMAT*)buffer;
        *datalen = length / sizeof(INPUT_FORMAT);
    }

    return true;
}

void TestPerformanceFunction(
    const INPUT_FORMAT* const input,
    const dataInfo* const inputInfo,
    OUTPUT_FORMAT* const output,
    const dataInfo* const outputInfo,
    const ImageParam* const param,
    const int count,
    CALLBACK_TYPE callback,
    const string title
    ) {

    TIME_FORMAT sum = 0;
    for (size_t i = 0; i < count; i++)
    {
        sum += callback(input, inputInfo, output, outputInfo, param);
    }
    TIME_FORMAT aveSum = sum / (TIME_FORMAT)count;

    stringstream stm;
    stm << title << ": " << aveSum << " ms\n";
    cout << stm.str();
}