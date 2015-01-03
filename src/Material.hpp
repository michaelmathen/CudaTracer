#include "ray_defs.hpp"


#ifndef MM_MATERIAL
#define MM_MATERIAL

namespace mm_ray {
  class Material {
  public:
    
    __host__ __device__ virtual bool isLight() {
      return false;
    }

    static const Virtual_Type_Val type_id = MATERIAL;
  };
}
#endif
