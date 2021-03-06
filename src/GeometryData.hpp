#include <vector>
#include <string>

#ifndef __CUDACC__
#include "rapidjson/document.h"
#endif

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

}
#endif

