#ifndef MM_CUDA_DEFS
#define MM_CUDA_DEFS
#include <cstdio>
#include <cstdlib>

#include <Vecky.hpp>
#include "Managed.hpp"


#ifndef __CUDACC__
  #define __host__
  #define __device__
  #define __global__ 
#endif

using namespace Vecky;

#define MM_BLOCK_DIM 16


typedef float Real_t;
typedef VecN<Real_t, 4> Vec4;
typedef VecN<Real_t, 3> Vec3;
typedef VecN<Real_t, 2> Vec2;

#endif
