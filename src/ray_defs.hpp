#include <cstdio>
#include <Vecky.hpp>


#ifndef MM_CUDA_DEFS
#define MM_CUDA_DEFS

using namespace Vecky;


#ifndef __CUDACC__
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

#define curand_check(err) { check_curand(err, __FILE__, __LINE__); }

void check_curand(curandStatus_t err, const char* file, int line){
  switch(err) {
  case CURAND_STATUS_SUCCESS:
    //All good
    return ;
  case CURAND_STATUS_VERSION_MISMATCH:
    fprintf(stderr, "Header file and linked library version do not match\n");
    break;
  case CURAND_STATUS_NOT_INITIALIZED:
    fprintf(stderr, "Generator not initialized\n");
    break;
  case CURAND_STATUS_ALLOCATION_FAILED:
    fprintf(stderr, "Memory allocation failed\n");
    break;
  case CURAND_STATUS_TYPE_ERROR:
    fprintf(stderr, "Generator is wrong type\n");
    break;
  case CURAND_STATUS_OUT_OF_RANGE:
    fprintf(stderr, "Argument out of range\n");
    break;
  case CURAND_STATUS_LENGTH_NOT_MULTIPLE:
    fprintf(stderr, "Length requested is not a multple of dimension\n");
    break;
  case CURAND_STATUS_LAUNCH_FAILURE:
    fprintf(stderr, "Kernel launch failure\n");
    break;
  case CURAND_STATUS_PREEXISTING_FAILURE:
    fprintf(stderr, "Preexisting failure on library entry\n");
    break;
  case CURAND_STATUS_INITIALIZATION_FAILED:
    fprintf(stderr, "Initialization of CUDA failed\n");
    break;
  case CURAND_STATUS_ARCH_MISMATCH:
    fprintf(stderr, "Architecture mismatch, GPU does not support requested feature\n");
    break;
  case CURAND_STATUS_DOUBLE_PRECISION_REQUIRED:
    fprintf(stderr, "The GPU does not support double precision\n");
    break;
  case CURAND_STATUS_INTERNAL_ERROR:
    fprintf(stderr, "Internal library error\n");
    break;
  }
  fprintf(stderr, "%s, %d\n", file, line);
}

#endif
#endif


typedef float Real_t;
typedef VecN<Real_t, 4> Vec4;
typedef VecN<Real_t, 3> Vec3;
typedef VecN<Real_t, 2> Vec2;

namespace mm_ray {
  enum Virtual_Type_Val {
    MATERIAL,
    PHONG_MAT,
    SPHERE,
    POINTLIGHT,
    TRIANGLE,
    NOT_VT
  };
}


#endif
