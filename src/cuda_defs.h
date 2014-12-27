#include <cstdio>
#ifndef MM_CUDA_DEFS
#define MM_CUDA_DEFS

#ifdef __CUDACC__
  #define __HOST__ __host__
  #define __DEVICE__ __device__
  #define __HOST_DEVICE__ __host__ __device__
#else
  #define __HOST__ 
  #define __DEVICE__ 
  #define __HOST_DEVICE__
  #define __host__
  #define __device__
  #define __global__ 
#endif

#define MM_BLOCK_DIM 16

#ifdef __CUDACC__
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

#ifdef CURAND_H_

#endif
#endif



typedef float Real_t;

namespace mm_ray {
  enum Virtual_Type_Val {
    MATERIAL,
    PHONG_MAT,
    SPHERE,
    POINTLIGHT,
    AMBIENT_MAT,
    NOT_VT
  };
}


#endif
