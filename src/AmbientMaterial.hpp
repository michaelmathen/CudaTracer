#include "ray_defs.hpp"
#include "Material.hpp"

#ifndef MM_AMBIENT_MATERIAL
#define MM_AMBIENT_MATERIAL

namespace mm_ray {
  class AmbientMaterial : public Material {
  public:
    Vec3 color; 
    AmbientMaterial(Vec3& color) : color(color){
    }

    __host__ __device__ AmbientMaterial(){}

#ifdef __CUDACC__
    Material* serialize(){
      AmbientMaterial* dev_mat;
      cudaMalloc(&dev_mat, sizeof(AmbientMaterial));
      cudaMemcpy(dev_mat, this, sizeof(AmbientMaterial), cudaMemcpyHostToDevice);
      return dev_mat;
    }
#endif

    static const Virtual_Type_Val type_id = AMBIENT_MAT;
  };
}
#endif
