# _ScanConversion   

2023.05.12 update   
texture memory 를 사용한 scan conversion을 추가했습니다.   
특징은 일반적으로  global memory보다 빠르지만, read only이고, 그래픽 카드에 따라 compute capability가 달라 사용가능한 메모리도 다른 것으로 알고있습니다.   
그래서 사용가능한 메모리를 cuda smples에 있는 deviceQuery 솔루션으로 확인해 보는게 좋습니다.   

예시,   
Device 0: "NVIDIA GeForce RTX 4090"   
  CUDA Driver Version:                           12.1   
  CUDA Capability Major/Minor version number:    8.9   
  Total amount of global memory:                 24564 MBytes (25756696576 bytes)   
  (128) Multiprocessors, (128) CUDA Cores/MP:     16384 CUDA Cores   
  GPU Max Clock rate:                            2535 MHz (2.54 GHz)   
  Memory Clock rate:                             10501 Mhz   
  Memory Bus Width:                              384-bit   
  L2 Cache Size:                                 75497472 bytes   
  Max Texture Dimension Sizes                    1D=(131072) 2D=(131072, 65536) 3D=(16384, 16384, 16384)   
  Maximum Layered 1D Texture Size, (num) layers  1D=(32768), 2048 layers   
  Maximum Layered 2D Texture Size, (num) layers  2D=(32768, 32768), 2048 layers   
  Total amount of constant memory:               65536 bytes   
  Total amount of shared memory per block:       49152 bytes   
  Total number of registers available per block: 65536   
  Warp size:                                     32   
  Maximum number of threads per multiprocessor:  1536   
  Maximum number of threads per block:           1024   
  Max dimension size of a thread block (x,y,z): (1024, 1024, 64)   
  Max dimension size of a grid size (x,y,z):    (2147483647, 65535, 65535)   
  Texture alignment:                             512 bytes   
  Maximum memory pitch:                          2147483647 bytes   
  Concurrent copy and kernel execution:          Yes with 1 copy engine(s)   
  Run time limit on kernels:                     Yes   
  Integrated GPU sharing Host Memory:            No   
  Support host page-locked memory mapping:       Yes   
  Concurrent kernel execution:                   Yes   
  Alignment requirement for Surfaces:            Yes   
  Device has ECC support:                        Disabled   
  CUDA Device Driver Mode (TCC or WDDM):         WDDM (Windows Display Driver Model)   
  Device supports Unified Addressing (UVA):      Yes   
  Device supports Managed Memory:                Yes   
  Device supports Compute Preemption:            Yes   
  Supports Cooperative Kernel Launch:            Yes   
  Supports MultiDevice Co-op Kernel Launch:      No   
  Device PCI Domain ID / Bus ID / location ID:   0 / 1 / 0   
  Compute Mode:   
     < Default (multiple host threads can use ::cudaSetDevice() with device simultaneously) >   
Result = PASS   
   
   
현재 구현된 코드로는 각 input의 width와 height의 크기를 몇 천 단위로 늘리게 되면 오히려 global memory로 구현한게 몇 배 빨라집니다.   
디버깅을 해보았으나, 제가 파악하기론 texture memory buffer 크기가 커지면 불리해지는 점이 있다고밖엔 생각을 못하고 있습니다.   
혹시 의견이 있으시다면 제게 알려주세요.   

seiyun Lee   






