#include "Geometry.hpp"


#ifndef MM_POINT_LIGHT
#define MM_POINT_LIGHT

namespace mm_ray {
  class PointLight : public Geometry {

    Vec3 location;
    Vec3 illumination;
  public:

    __host__ __device__ PointLight(){}
  
    PointLight(Vec3 illumination, Vec3 location) : location(location),
						   illumination(illumination)
    {}

    __host__ __device__ virtual void intersectRay(Ray& ray, Hit& prop){
      //You don't hit a point
    }

    __host__ __device__ virtual bool isLight(){
      return true;
    }
    
    __host__ __device__  virtual Vec3 getCenter() {
      return location;
    }

    __host__ __device__ virtual Vec3 getLight() {
      return illumination;
    }
    
    static const Virtual_Type_Val type_id = POINTLIGHT;
  };
}
#endif
