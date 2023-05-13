
#include "stdafx.h"
#include "kernel.cuh"


__global__ void ScanConversionKernel(
    const char* const input, 
    char* const output
    )
{
    int i = threadIdx.x;
}

// Helper function for using CUDA to add vectors in parallel.
cudaError_t ScanConversion(const char* const input, dataInfo* inputInfo, char* const output, dataInfo* outputInfo)
{
    cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        throw std::logic_error("cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
    }

    //define auto memory free pointer
    std::shared_ptr<char> gInput(AllocCudaMem(inputInfo), cudaFree);
    std::shared_ptr<char> gOutput(AllocCudaMem(outputInfo), cudaFree);

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(gInput.get(), input, GetTotalSize(inputInfo), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        throw std::logic_error("cudaMemcpy failed!");
    }

    dim3 block(512,1,1); //thread x, y, z count
    dim3 grid(inputInfo->width / block.x , inputInfo->heigth / block.y , 1 / block.z); // block x, y, z count
    //function

    ScanConversionKernel <<< block, grid >>> (gInput.get(), gOutput.get());

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        std::stringstream stm;
        stm << "addKernel launch failed: " << cudaGetErrorString(cudaStatus) << "\n";
        throw std::logic_error(stm.str());
    }

    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        std::stringstream stm;
        stm << "cudaDeviceSynchronize returned error code " << (int)cudaStatus << " after launching addKernel!\n";
        throw std::logic_error(stm.str());
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(output, gOutput.get(), GetTotalSize(outputInfo), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        throw std::logic_error("cudaMemcpy failed!");
    }

    return cudaStatus; // SUCCESS status return
}

char* AllocCudaMem(dataInfo* inputInfo)
{
    cudaError_t cudaStatus;

    char* temp;
    cudaStatus = cudaMalloc((void**)&temp, GetTotalSize(inputInfo));
    if (cudaStatus != cudaSuccess) {
        cudaFree(temp);
        throw std::logic_error("CUDA malloc error");
    }

    return temp;
}

int GetTotalSize(dataInfo* info)
{
    if(info == nullptr) throw std::logic_error("info is null");

    //data size calculation could be exceed data format(32bit signed). need to carefully use data type which is returned

    return info->heigth * info->unitDataSize * info->unitDataSize;
}
