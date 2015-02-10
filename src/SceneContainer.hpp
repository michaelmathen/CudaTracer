#include <algorithm>
#include <iostream>
#include <memory>
#include "ray_defs.hpp"
#include "Accelerator.hpp"
#include "Geometry.hpp"

#ifndef MM_SCENE_CONTAINER
#define MM_SCENE_CONTAINER
namespace mm_ray {

  
  class SceneContainer : public Accelerator<SceneContainer> {
  
    Geometry** geometry_buffer;

    unsigned geom_length;
  public:
    
    SceneContainer(std::vector<Geometry*>& geom) :
      Accelerator<SceneContainer>(geom)
    {
      geom_length = geom.size();
      geometry_buffer = (Geometry**)Cuda_Malloc(geom.size() * sizeof(Geometry*));
      
      for (unsigned i = 0; i < geom.size(); i++){
	geometry_buffer[i] = geom[i];
      }
    }
    
    ~SceneContainer(){
      Cuda_Free(geometry_buffer);
    }
    
    __host__ __device__ void Intersect(Ray const& ray, Hit& prop) const {
      Hit tmp;
      prop.distance = INFINITY;
      for (int i = 0; i < geom_length; i++){
	geometry_buffer[i]->Intersect_Ray(ray, tmp);
	if (tmp.distance < prop.distance){
	  prop = tmp;
	}
      }
    }
  };
}
#endif
