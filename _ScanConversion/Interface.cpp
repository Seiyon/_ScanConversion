#include "stdafx.h"
#include "interface.h"
#include "function.cuh"

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
    std::shared_ptr<INPUT_FORMAT> gInput((INPUT_FORMAT*)AllocCudaMem(inputInfo), cudaFree);
    std::shared_ptr<OUTPUT_FORMAT> gOutput((OUTPUT_FORMAT*)AllocCudaMem(outputInfo), cudaFree);

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(gInput.get(), input, GetTotalSize(inputInfo), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        throw std::exception("cudaMemcpy failed!");
    }

    dim3 block(512, 1); //thread x, y count
    dim3 grid((outputInfo->iWidth - 1) / block.x + 1, (outputInfo->iHeigth - 1) / block.y + 1); // block x, y count
    // minus 1 in molecule and plus one to optimize thread index calculation
    //function

    StopWatchInterface* timer = NULL;
    sdkCreateTimer(&timer);
    sdkStartTimer(&timer);
    ExcuteScanConversionKernel(&grid, &block, gInput.get(), *inputInfo, gOutput.get(), *outputInfo, *param);

    // Check for any errors launching the kernel
    /*cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        std::stringstream stm;
        stm << "ScanConversionKernel launch failed: " << cudaGetErrorString(cudaStatus) << "\n";
        throw std::exception(stm.str().c_str());
    }*/

    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        std::stringstream stm;
        stm << "cudaDeviceSynchronize returned error code " << (int)cudaStatus << " after launching addKernel!\n";
        throw std::exception(stm.str().c_str());
    }
    sdkStopTimer(&timer);

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(output, gOutput.get(), GetTotalSize(outputInfo), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        throw std::exception("cudaMemcpy failed!");
    }

    return sdkGetTimerValue(&timer);
}

TIME_FORMAT ScanConversionTexture(const INPUT_FORMAT* const input, const dataInfo* const inputInfo, OUTPUT_FORMAT* const output, const dataInfo* const outputInfo, const ImageParam* const param)
{
    cudaError_t cudaStatus;

    checkCudaErrors(cudaSetDevice(0));

    cudaChannelFormatDesc channelDesc = 
        cudaCreateChannelDesc(8 * sizeof(INPUT_FORMAT), 0, 0, 0, cudaChannelFormatKindFloat);
    std::shared_ptr<cudaArray> gCuArray(AllocCudaArray(inputInfo, &channelDesc), cudaFreeArray);
    checkCudaErrors(cudaMemcpyToArray(gCuArray.get(), 0, 0, input, GetTotalSize(inputInfo), cudaMemcpyHostToDevice));

    cudaResourceDesc textRes;
    memset(&textRes, 0, sizeof(cudaResourceDesc));

    // resource option
    textRes.resType = cudaResourceTypeArray;
    textRes.res.array.array = gCuArray.get();

    cudaTextureDesc textDesc;
    memset(&textDesc, 0, sizeof(cudaTextureDesc));

    // descriptor option
    textDesc.normalizedCoords = false;
    textDesc.filterMode = cudaFilterModeLinear;
    textDesc.addressMode[0] = cudaAddressModeWrap;
    textDesc.addressMode[1] = cudaAddressModeWrap;
    textDesc.readMode = cudaReadModeElementType;

    std::shared_ptr<cudaTextureObject_t> gInput(CreateTextureObject(&textRes, &textDesc), DeleteTextureObject);
    std::shared_ptr<OUTPUT_FORMAT> gOutput((OUTPUT_FORMAT*)AllocCudaMem(outputInfo), cudaFree);

    dim3 block(512, 1);
    dim3 grid((outputInfo->iWidth - 1) / block.x + 1, (outputInfo->iHeigth - 1) / block.y + 1);

    StopWatchInterface* timer = NULL;
    sdkCreateTimer(&timer);
    sdkStartTimer(&timer);
    ExcuteScanConversionKernelTexture(&grid, &block, *gInput.get(), *inputInfo, gOutput.get(), *outputInfo, *param);
    

    //checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
    sdkStopTimer(&timer);

    checkCudaErrors(cudaMemcpy(output, gOutput.get(), GetTotalSize(outputInfo), cudaMemcpyDeviceToHost));

    return sdkGetTimerValue(&timer);
}


int GetTotalSize(const dataInfo* const info)
{
    if (info == nullptr) throw std::exception("info is null");

    //data size calculation could be exceed data format(32bit signed). need to carefully use data type which is returned
    //printf("buff size: %d(bytes)\n", info->iHeigth * info->iWidth * info->iUnitDataSize);

    return info->iHeigth * info->iWidth * info->iUnitDataSize;
}

void DeleteTextureObject(cudaTextureObject_t* obj) {
    checkCudaErrors(cudaDestroyTextureObject(*obj));
}

cudaTextureObject_t* CreateTextureObject(const cudaResourceDesc* const textRes, const cudaTextureDesc* const textDesc) {

    cudaTextureObject_t* obj = new cudaTextureObject_t();
    checkCudaErrors(cudaCreateTextureObject(obj, textRes, textDesc, NULL));
    return obj;
}

cudaArray* AllocCudaArray(const dataInfo* const info, const cudaChannelFormatDesc* const channelDesc) {

    cudaArray* cuArray;
    checkCudaErrors(cudaMallocArray(&cuArray, channelDesc, info->iWidth, info->iHeigth));
    return cuArray;
}

void* AllocCudaMem(const dataInfo* const inputInfo)
{
    cudaError_t cudaStatus;
    void* temp = nullptr;
    checkCudaErrors(cudaMalloc((void**)&temp, GetTotalSize(inputInfo)));
    return temp;
}