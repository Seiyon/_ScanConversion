
#include "stdafx.h"
#include "kernel.cuh"


__global__ void ScanConversionKernel(
    const INPUT_FORMAT* const input,
    const dataInfo const inputInfo,
    OUTPUT_FORMAT* const output,
    const dataInfo const outputInfo,
    const ImageParam const param
    )
{
    const int widthIdx = threadIdx.x + blockDim.x * blockIdx.x;
    const int heightIdx = threadIdx.y + blockDim.y * blockIdx.y;

    if (param.fGridStepM < 0) return; // for miss calcaculation

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

// Helper function for using CUDA to add vectors in parallel.
TIME_FORMAT ScanConversion(const INPUT_FORMAT* const input, const dataInfo* const inputInfo, OUTPUT_FORMAT* const output, const dataInfo* const outputInfo, const ImageParam* const param)
{
    cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        throw std::exception("cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
    }

    //define auto memory free pointer
    printf("input");
    std::shared_ptr<INPUT_FORMAT> gInput((INPUT_FORMAT*)AllocCudaMem(inputInfo), cudaFree);
    printf("output");
    std::shared_ptr<OUTPUT_FORMAT> gOutput((OUTPUT_FORMAT*)AllocCudaMem(outputInfo), cudaFree);

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(gInput.get(), input, GetTotalSize(inputInfo), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        throw std::exception("cudaMemcpy failed!");
    }

    dim3 block(512,1); //thread x, y count
    dim3 grid((outputInfo->iWidth - 1)/ block.x + 1 , (outputInfo->iHeigth - 1) / block.y + 1); // block x, y count
    // minus 1 in molecule and plus one to optimize thread index calculation
    //function

    std::chrono::system_clock::time_point startTime = std::chrono::system_clock::now();
    ScanConversionKernel <<< grid, block >>> (gInput.get(), *inputInfo, gOutput.get(),*outputInfo, *param);
    std::chrono::duration<TIME_FORMAT, std::micro> elapsedTime = std::chrono::system_clock::now() - startTime;

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        std::stringstream stm;
        stm << "ScanConversionKernel launch failed: " << cudaGetErrorString(cudaStatus) << "\n";
        throw std::exception(stm.str().c_str());
    }

    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        std::stringstream stm;
        stm << "cudaDeviceSynchronize returned error code " << (int)cudaStatus << " after launching addKernel!\n";
        throw std::exception(stm.str().c_str());
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(output, gOutput.get(), GetTotalSize(outputInfo), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        throw std::exception("cudaMemcpy failed!");
    }

    return elapsedTime.count(); // SUCCESS status return
}

void* AllocCudaMem(const dataInfo* const inputInfo)
{
    cudaError_t cudaStatus;

    void* temp = nullptr;
    cudaStatus = cudaMalloc((void**)&temp, GetTotalSize(inputInfo));
    if (cudaStatus != cudaSuccess) {
        cudaFree(temp);
        throw std::exception("CUDA malloc error");
    }

    return temp;
}

int GetTotalSize(const dataInfo* const info)
{
    if(info == nullptr) throw std::exception("info is null");

    //data size calculation could be exceed data format(32bit signed). need to carefully use data type which is returned
    printf("get buff size: %d(bytes)\n", info->iHeigth * info->iWidth * info->iUnitDataSize);

    return info->iHeigth * info->iWidth * info->iUnitDataSize;
}
