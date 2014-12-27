#include "cuda_defs.h"
#include "VectorMath.hpp"

#ifndef MM_RAY_BASE
#define MM_RAY_BASE

enum RayType { RAYBASE };

class Ray {
  //This member lets us implement virtual functions without virtual functions
  RayType ray_t;
public:
  Vec3 direc;
  Vec3 origin;

  __host__ __device__ Ray(Vec3& direc, Vec3& origin) : direc(direc), origin(origin){
    ray_t = RAYBASE;
  }

};
#endif
