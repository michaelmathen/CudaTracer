#include "Material.hpp"

#ifndef MM_PHONG
#define MM_PHONG
namespace mm_ray {

  class PhongMaterial : public Material {
  public:
    Vec3 color;
    Real_t spec_light;
    Real_t diff_light;
    Real_t amb_light;
    Real_t shine;
    
    __host__ __device__ PhongMaterial(){}
    
    PhongMaterial(Vec3& color,
		  Real_t spec_light, Real_t diff_light,
		  Real_t amb_light, Real_t shine) : color(color),
						    spec_light(spec_light),
						    diff_light(diff_light),
						    amb_light(amb_light),
						    shine(shine){
    }
    
    static const Virtual_Type_Val type_id = PHONG_MAT;
  };
}


#endif
