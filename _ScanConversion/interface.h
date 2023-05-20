#pragma once


TIME_FORMAT ScanConversion(
	const INPUT_FORMAT* const input,
	const dataInfo* const inputInfo,
	OUTPUT_FORMAT* const output,
	const dataInfo* const outputInfo,
	const ImageParam* const param);

TIME_FORMAT ScanConversionTexture(
	const INPUT_FORMAT* const input,
	const dataInfo* const inputInfo,
	OUTPUT_FORMAT* const output,
	const dataInfo* const outputInfo,
	const ImageParam* const param);


int GetTotalSize(const dataInfo* const info);

void DeleteTextureObject(cudaTextureObject_t* obj);

cudaTextureObject_t* CreateTextureObject(const cudaResourceDesc* const textRes, const cudaTextureDesc* const textDesc);

cudaArray* AllocCudaArray(const dataInfo* const info, const cudaChannelFormatDesc* const channelDesc);

void* AllocCudaMem(const dataInfo* const inputInfo);
