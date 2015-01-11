#include "rapidjson/document.h"
#include "ray_defs.hpp"
#include <memory>

#ifndef MM_MATERIAL
#define MM_MATERIAL

namespace mm_ray {

  enum MaterialType {
    PHONG_MAT
  };
    
  class Material : public Managed {
    const MaterialType material_type;
  public:

    __host__ __device__ inline Material(MaterialType mat_type);
    
    __host__ __device__  inline bool isLight() const;


  };

  class PhongMaterial : public Material {
  public:
    Vec3 color;
    Real_t spec_light;
    Real_t diff_light;
    Real_t amb_light;
    Real_t shine;
    
    __host__ __device__ PhongMaterial() : Material(PHONG_MAT) {}
    
    PhongMaterial(Vec3& color,
		  Real_t spec_light, Real_t diff_light,
		  Real_t amb_light, Real_t shine) : Material(PHONG_MAT),
						    color(color),
						    spec_light(spec_light),
						    diff_light(diff_light),
						    amb_light(amb_light),
						    shine(shine)
    {}

    __host__ __device__  bool isLight() const {
      return false;
    }

  };

  
  inline Material::Material(MaterialType mat) :
    material_type(mat)
  {}
    
  __host__ __device__ inline bool Material::isLight() const {
      return false;
    }
  
  class Scene; 

  struct MaterialBuilder {
    virtual Material* operator()(rapidjson::Value&, Scene const& scene_data) = 0;
  };

  struct PhongMaterialBuilder {
    virtual Material* operator()(rapidjson::Value&, Scene const&);
  };
  
}
#endif
