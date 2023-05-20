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
    printf("input");
    std::shared_ptr<INPUT_FORMAT> gInput((INPUT_FORMAT*)AllocCudaMem(inputInfo), cudaFree);
    printf("output");
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

    std::chrono::system_clock::time_point startTime = std::chrono::system_clock::now();
    ExcuteScanConversionKernel(&grid, &block, gInput.get(), *inputInfo, gOutput.get(), *outputInfo, *param);

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
    std::chrono::duration<TIME_FORMAT, std::micro> elapsedTime = std::chrono::system_clock::now() - startTime;

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(output, gOutput.get(), GetTotalSize(outputInfo), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        throw std::exception("cudaMemcpy failed!");
    }

    return elapsedTime.count();
}

TIME_FORMAT ScanConversionTexture(const INPUT_FORMAT* const input, const dataInfo* const inputInfo, OUTPUT_FORMAT* const output, const dataInfo* const outputInfo, const ImageParam* const param)
{
    cudaError_t cudaStatus;

    checkCudaErrors(cudaSetDevice(0));

    std::shared_ptr<INPUT_FORMAT> gInput((INPUT_FORMAT*)AllocCudaMem(inputInfo), cudaFree);
    std::shared_ptr<OUTPUT_FORMAT> gOutput((OUTPUT_FORMAT*)AllocCudaMem(outputInfo), cudaFree);

    checkCudaErrors(cudaMemcpy(gInput.get(), input, GetTotalSize(inputInfo), cudaMemcpyHostToDevice));

    dim3 block(512, 1);
    dim3 grid((outputInfo->iWidth - 1) / block.x + 1, (outputInfo->iHeigth - 1) / block.y + 1);

    std::chrono::system_clock::time_point startTime = std::chrono::system_clock::now();
    ExcuteScanConversionKernel(&grid, &block, gInput.get(), *inputInfo, gOutput.get(), *outputInfo, *param);

    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
    std::chrono::duration<TIME_FORMAT, std::micro> elapsedTime = std::chrono::system_clock::now() - startTime;

    checkCudaErrors(cudaMemcpy(output, gOutput.get(), GetTotalSize(outputInfo), cudaMemcpyDeviceToHost));

    return elapsedTime.count();
}

void* AllocCudaMem(const dataInfo* const inputInfo)
{
    cudaError_t cudaStatus;
    void* temp = nullptr;
    checkCudaErrors(cudaMalloc((void**)&temp, GetTotalSize(inputInfo)));
    return temp;
}

int GetTotalSize(const dataInfo* const info)
{
    if (info == nullptr) throw std::exception("info is null");

    //data size calculation could be exceed data format(32bit signed). need to carefully use data type which is returned
    printf("buff size: %d(bytes)\n", info->iHeigth * info->iWidth * info->iUnitDataSize);

    return info->iHeigth * info->iWidth * info->iUnitDataSize;
}
