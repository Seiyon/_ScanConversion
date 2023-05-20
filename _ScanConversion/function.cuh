


extern "C" __global__ void ScanConversionKernel(
    const INPUT_FORMAT* const input,
    const dataInfo inputInfo,
    OUTPUT_FORMAT* const output,
    const dataInfo outputInfo,
    const ImageParam param
);

extern "C" __global__ void ScanConversionKernelTexture(
    const cudaTextureObject_t const input,
    const dataInfo inputInfo,
    OUTPUT_FORMAT* const output,
    const dataInfo outputInfo,
    const ImageParam param
);

void ExcuteScanConversionKernel(
    const dim3* grid,
    const dim3* block,
    const INPUT_FORMAT* const input,
    const dataInfo inputInfo,
    OUTPUT_FORMAT* const output,
    const dataInfo outputInfo,
    const ImageParam param
);

void ExcuteScanConversionKernelTexture(
    const dim3* grid,
    const dim3* block,
    const cudaTextureObject_t const input,
    const dataInfo inputInfo,
    OUTPUT_FORMAT* const output,
    const dataInfo outputInfo,
    const ImageParam param
);
