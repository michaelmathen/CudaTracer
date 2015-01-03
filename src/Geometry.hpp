#include "ray_defs.hpp"
#include "Material.hpp"
#include "Ray.hpp"
#include "Hit.hpp"

#include "SceneAllocator.hpp"

#ifndef MM_GEOMETRY
#define MM_GEOMETRY

namespace mm_ray {

  class Geometry {

  public:

    //This cannot do anything!!!!
    //Our serialization relies on this type of default
    //constructor
    
    __host__ __device__  bool intersectBox(Vec3& l, Vec3& u) {
      return false;
    }
    __host__ __device__ virtual void intersectRay(Ray& ray, Hit& prop) = 0;

    __host__ __device__ virtual Vec3 getLight() {
      Vec3 light = 0;
      return light;
    }

    __host__ __device__ virtual s_ptr<Material> getMaterial() = 0;
      
    __host__ __device__ virtual bool isLight() {
      return false;
    }

    /*
      This gets the rough center of the object. 
     */
    __host__ __device__ inline virtual Vec3 getCenter() = 0;
    
  };
}

#endif
