#include <vector>
#include "ray_defs.hpp"
#include "Managed.hpp"
#include "Geometry.hpp"

#ifndef MM_ACCELERATOR
#define MM_ACCELERATOR
namespace mm_ray {
  class Ray;
  class Hit;
  
  template <typename T>
  class Accelerator : public Managed {

    Geometry** light_sources;
    unsigned light_length;
  public:

    ~Accelerator(){
      Cuda_Free(light_sources);
      static_cast<T*>(this)->~T();
    }
    
    Accelerator(std::vector<Geometry*>& geom){
      light_length = 0;
      for (unsigned i = 0; i < geom.size(); i++){
	if (geom[i]->isLight())
	  light_length++;
      }
      light_sources = (Geometry**)Cuda_Malloc(sizeof(Geometry*) * light_length);
      int light_curr = 0;
      for (unsigned i = 0; i < geom.size(); i++){
	if (geom[i]->isLight()) {
	  light_sources[light_curr] = geom[i];
	  light_curr++;
	}
      }
    }
    
    __host__ __device__ void Intersect(Ray const& ray, Hit& prop) const {
      static_cast<T*>(this)->Intersect(ray, prop);
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
