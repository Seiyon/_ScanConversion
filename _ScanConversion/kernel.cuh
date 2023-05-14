

TIME_FORMAT ScanConversion(
	const INPUT_FORMAT* const input,
	const dataInfo* const inputInfo, 
	OUTPUT_FORMAT* const output,
	const dataInfo* const outputInfo, 
	const ImageParam* const param);

void* AllocCudaMem(const dataInfo* const inputInfo);

int GetTotalSize(const dataInfo* const info);