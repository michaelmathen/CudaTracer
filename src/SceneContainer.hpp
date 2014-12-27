#include <algorithm>
#include <iostream>
#include "cuda_defs.h"
#include "VectorMath.hpp"
#include "SceneObjects.hpp"
#include "Geometry.hpp"
#include "Sphere.hpp"


#ifndef MM_SCENE_CONTAINER
#define MM_SCENE_CONTAINER
namespace mm_ray {

class SceneContainer {
  
  s_ptr<s_ptr<Geometry> > geometry_buffer;
  s_ptr<s_ptr<Geometry> > light_sources;
  
  //int material_length;
  int geom_length;
  int light_length;
  
public:
  SceneContainer(){
  }


  void insertGeometry(s_ptr<s_ptr<Geometry> > geom, int geom_length);
  
  ~SceneContainer(){
  }


  void initialize(){}

  __host__ __device__ void intersect(Ray& ray, Hit& prop){
    Hit tmp;
    prop.distance = INFINITY;
    for (int i = 0; i < geom_length; i++){
      geometry_buffer[i]->intersectRay(ray, tmp);
      if (tmp.distance < prop.distance){
	prop = tmp;
      }
    }
  }

  __host__ __device__ inline int getLightNumber(){
    return light_length;
  }

  
  __host__ __device__ inline s_ptr<Geometry> getLight(int i){
    return light_sources[i];
  }
  

  
};

}
#endif
