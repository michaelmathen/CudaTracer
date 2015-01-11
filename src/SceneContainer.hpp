#include <algorithm>
#include <iostream>
#include <memory>
#include "ray_defs.hpp"
#include "Geometry.hpp"
#include "Managed.hpp"

#ifndef MM_SCENE_CONTAINER
#define MM_SCENE_CONTAINER
namespace mm_ray {

  
  class SceneContainer : public Managed {
  
    Geometry** geometry_buffer;
    Geometry** light_sources;
    
    unsigned geom_length;
    unsigned light_length;
    unsigned material_length;
    
  public:
    
    SceneContainer(){}
    ~SceneContainer();
    
    static SceneContainer* Build_Accelerator(std::vector<Geometry*>&);
    
    __host__ __device__ void intersect(Ray const& ray, Hit& prop) const {
      Hit tmp;
      prop.distance = INFINITY;
      for (int i = 0; i < geom_length; i++){
	geometry_buffer[i]->intersectRay(ray, tmp);
	if (tmp.distance < prop.distance){
	  prop = tmp;
	}
      }
      
    }
    
    __host__ __device__ inline int getLightNumber() const {
      return light_length;
    }
    
    __host__ __device__ inline Geometry const* getLight(int i) const {
      return light_sources[i];
    }
  };

}
#endif
