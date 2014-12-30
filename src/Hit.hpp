#include "SceneObjects.hpp"
#include "SceneAllocator.hpp"
#include "Material.hpp"
#include "ray_defs.hpp"

#ifndef MM_HIT
#define MM_HIT
namespace mm_ray {
  class Hit {
  public:
    __host__ __device__ Hit(){
      hit = false;
      distance = INFINITY;
    }
    
    bool hit;
    Real_t distance;
    Vec3 normal;
    s_ptr<Material> material;
    Vec3 hit_location;
  };
}
#endif
