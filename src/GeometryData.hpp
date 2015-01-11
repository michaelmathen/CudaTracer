#include <vector>
#include <string>
#include "rapidjson/document.h"
#include "Transform.hpp"
#include "ray_defs.hpp"
#include "Scene.hpp"

#ifndef MM_GEOMETRY_DATA
#define MM_GEOMETRY_DATA

namespace mm_ray {


  struct Tri_vert : public Managed {
    int x;
    int y;
    int z;
  };

  class Triangle;
  class Geometry;
  class Material;

  class TriangleMesh : public Managed {
    Vec3* triangle_vertices;
    Tri_vert* vertex_indices;
   

    friend class Triangle;
  public:
    Material* material;
    unsigned int number_of_triangles;

    __device__ __host__ TriangleMesh(){}
    std::vector<Geometry*> refine();
#ifndef __CUDACC__
    void setMaterial(Material* mat);
    void parseObj(std::string const& fname, Transform const& tranform);

#endif 
  };


  
  
  struct GeometryBuilder {
    virtual void operator() (rapidjson::Value& sphere_obj,
			     Scene const& scn,
			     std::vector<Material*>& materials, 
			     std::vector<std::string>& material_names, 
			     std::vector<Geometry*>& geom_ptrs,
			     std::vector<Managed*>& geom_data) = 0;
  };

  struct SphereBuilder : public GeometryBuilder {
    virtual void operator()(rapidjson::Value& sphere_obj,
			    Scene const& scn,
			    std::vector<Material*>& materials, 
			    std::vector<std::string>& material_names, 
			    std::vector<Geometry*>& geom_ptrs,
			    std::vector<Managed*>& geom_data);
  
  };

  struct TriangleMeshBuilder : public GeometryBuilder {
    virtual void operator()(rapidjson::Value& sphere_obj,
			    Scene const& scn,
			    std::vector<Material*>& materials, 
			    std::vector<std::string>& material_names, 
			    std::vector<Geometry*>& geom_ptrs,
			    std::vector<Managed*>& geom_data);
  
  };

  struct PointBuilder : public GeometryBuilder {
    virtual void operator()(rapidjson::Value& sphere_obj,
			    Scene const& scn,
			    std::vector<Material*>& materials, 
			    std::vector<std::string>& material_names, 
			    std::vector<Geometry*>& geom_ptrs,
			    std::vector<Managed*>& geom_data);
  
  };

}

#endif

