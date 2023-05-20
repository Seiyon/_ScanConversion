
#include "stdafx.h"
#include "function.cuh"


__global__ void ScanConversionKernel(
    const INPUT_FORMAT* const input,
    const dataInfo inputInfo,
    OUTPUT_FORMAT* const output,
    const dataInfo outputInfo,
    const ImageParam param
    )
{
    const int widthIdx = threadIdx.x + blockDim.x * blockIdx.x;
    const int heightIdx = threadIdx.y + blockDim.y * blockIdx.y;

    //boundary condition of data length
    if (widthIdx < outputInfo.iWidth && heightIdx < outputInfo.iHeigth)
    {
        //xIdx for leteral directrion, base location is center of the element
        const float xLength = heightIdx * param.fGridStepM;
        //zIdx for axial index. match to width index
        const float zLength = widthIdx * param.fGridStepM;
        
        const float xIdx = xLength / param.fLeteralStepM;
        const float zIdx = zLength / param.fAxialStepM;

        const int xIdxFloor = (int)xIdx;
        const int zIdxFloor = (int)zIdx;
        const float xIdxFrac = xIdx - xIdxFloor;
        const float zIdxFrac = zIdx - zIdxFloor;

        //calculate 4 index
        const int left_up_index = xIdxFloor * inputInfo.iWidth + zIdxFloor;
        const int right_up_index = left_up_index + 1;
        const int left_down_index = left_up_index + inputInfo.iWidth;
        const int right_down_index = left_down_index + 1;

        float left_up    = 0;
        float right_up   = 0;
        float left_down  = 0;
        float right_down = 0;

        if (zIdxFloor < inputInfo.iWidth - 1 && xIdxFloor < inputInfo.iHeigth - 1) {
            left_up = input[left_up_index] * (1 - xIdxFrac) * (1 - zIdxFrac);
            right_up = input[right_up_index] * (1 - xIdxFrac) * zIdxFrac;
            left_down = input[left_down_index] * xIdxFrac * (1 - zIdxFrac);
            right_down = input[right_down_index] * xIdxFrac * zIdxFrac;
        }
        else if (zIdxFloor == inputInfo.iWidth - 1) {
            left_up = input[left_up_index] * (1 - xIdxFrac) * (1 - zIdxFrac);
            //right_up = 0;
            left_down = input[left_down_index] * xIdxFrac * (1 - zIdxFrac);
            //right_down = 0;
        }
        else if (xIdxFloor == inputInfo.iHeigth - 1) {
            left_up = input[left_up_index] * (1 - xIdxFrac) * (1 - zIdxFrac);
            right_up = input[right_up_index] * (1 - xIdxFrac) * zIdxFrac;
            //left_down = 0;
            //right_down = 0;
        }

        //const int offset_index_plot_center = (outputInfo.iHeigth - inputInfo.iHeigth) / 2;
        output[heightIdx * outputInfo.iWidth + widthIdx] = left_up + right_up + left_down + right_down;

    }

}

__global__ void ScanConversionKernelTexture(
    const cudaTextureObject_t const input, 
    const dataInfo inputInfo, 
    OUTPUT_FORMAT* const output, 
    const dataInfo outputInfo, 
    const ImageParam param)
{
    const int widthIdx = threadIdx.x + blockDim.x * blockIdx.x;
    const int heightIdx = threadIdx.y + blockDim.y * blockIdx.y;

    //boundary condition of data length
    if (widthIdx < outputInfo.iWidth && heightIdx < outputInfo.iHeigth)
    {
        //xIdx for leteral directrion, base location is center of the element
        const float xLength = heightIdx * param.fGridStepM;
        //zIdx for axial index. match to width index
        const float zLength = widthIdx * param.fGridStepM;

        const float xIdx = xLength / param.fLeteralStepM;
        const float zIdx = zLength / param.fAxialStepM;
        const int xIdxFloor = (int)xIdx;
        const int zIdxFloor = (int)zIdx;

        //const int offset_index_plot_center = (outputInfo.iHeigth - inputInfo.iHeigth) / 2;
        // read from texture and write to global memory
        if (zIdxFloor < inputInfo.iWidth - 1 && xIdxFloor < inputInfo.iHeigth - 1) {
            output[heightIdx * outputInfo.iWidth + widthIdx] = tex2D<INPUT_FORMAT>(input, zIdx, xIdx);
        }
    }
}


void ExcuteScanConversionKernel(
    const dim3* grid,
    const dim3* block,
    const INPUT_FORMAT* const input,
    const dataInfo inputInfo,
    OUTPUT_FORMAT* const output,
    const dataInfo outputInfo,
    const ImageParam param
)
{
    ScanConversionKernel << < *grid, *block >> > (input, inputInfo, output, outputInfo, param);
}

void ExcuteScanConversionKernelTexture(
    const dim3* grid,
    const dim3* block,
    const cudaTextureObject_t const input,
    const dataInfo inputInfo,
    OUTPUT_FORMAT* const output,
    const dataInfo outputInfo,
    const ImageParam param
) {
    ScanConversionKernelTexture << < *grid, *block >> > (input, inputInfo, output, outputInfo, param);
}
