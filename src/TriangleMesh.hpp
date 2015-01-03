#include <vector>
#include <string>
#include "Material.hpp"
#include "Geometry.hpp"
#include "Transform.hpp"
#include "ray_defs.hpp"
#include "SceneAllocator.hpp"

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
    s_ptr<Vec3> triangle_vertices;
    s_ptr<Tri_vert> vertex_indices;
   

    friend class Triangle;
  public:
    s_ptr<Material> material;
    unsigned int number_of_triangles;

    __device__ __host__ TriangleMesh(){}

    
#ifndef __CUDACC__
    void setMaterial(s_ptr<Material> mat);
    void parseObj(std::string const& fname, Transform const& tranform);

#endif 
  };
  std::vector<s_ptr<Geometry> > refine(s_ptr<TriangleMesh>);
}

#endif
