


void ExcuteScanConversionKernel(
    const dim3* grid,
    const dim3* block,
    const INPUT_FORMAT* const input,
    const dataInfo inputInfo,
    OUTPUT_FORMAT* const output,
    const dataInfo outputInfo,
    const ImageParam param
);

__global__ void ScanConversionKernel(
    const INPUT_FORMAT* const input,
    const dataInfo inputInfo,
    OUTPUT_FORMAT* const output,
    const dataInfo outputInfo,
    const ImageParam param
);
