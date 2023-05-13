

cudaError_t ScanConversion(const char* const input, dataInfo* inputInfo, char* const output, dataInfo* outputInfo);

char* AllocCudaMem(dataInfo* inputInfo);

int GetTotalSize(dataInfo* info);