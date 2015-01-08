#include <vector>
#include <string>
#include "Material.hpp"
#include "Transform.hpp"
#include "ray_defs.hpp"


#ifndef MM_TRIANGLE_MESH
#define MM_TRIANGLE_MESH

namespace mm_ray {

  struct Tri_vert {
    int x;
    int y;
    int z;
  };
  class Triangle;
  
  class TriangleMesh {
    Vec3* triangle_vertices;
    Tri_vert* vertex_indices;
   

    friend class Triangle;
  public:
    Material* material;
    unsigned int number_of_triangles;

    __device__ __host__ TriangleMesh(){}

    
#ifndef __CUDACC__
    void setMaterial(Material* mat);
    void parseObj(std::string const& fname, Transform const& tranform);

#endif 
  };
  std::vector<Geometry*> refine(TriangleMesh*);
}

#endif
