#include <vector>
#include <string>
#include "SceneAllocator.hpp"
#include "Material.hpp"
#include "ray_defs.hpp"

#ifndef MM_TRIANGLE_MESH
#define MM_TRIANGLE_MESH
namespace mm_ray {

  class Triangle;
  
  class TriangleMesh {
    
  public:

    s_ptr<Vec3> triangle_vertices;
    s_ptr<int[3]> vertex_indices;
    unsigned int number_of_triangles;
    s_ptr<Material> material;

    TriangleMesh(){}

    TriangleMesh(std::vector<Vec3> const& vertices, std::vector<int[3]> const& v_indices, s_ptr<Material> mat) :
      material(mat)
    {
      triangle_vertices = scene_alloc<Vec3>(vertices.size());
      vertex_indices = scene_alloc<int[3]>(v_indices.size());
      
      for (unsigned int i = 0; i < vertices.size(); i++){
	triangle_vertices[i] = vertices[i];
      }

      for (unsigned int i = 0; i < v_indices.size(); i++){
	vertex_indices[i][0] = v_indices[i][0];
	vertex_indices[i][1] = v_indices[i][1];
	vertex_indices[i][2] = v_indices[i][2];
      }
    }

#ifndef __CUDACC__
    TriangleMesh(std::string const& fname, s_ptr<Material> material_type);
    std::vector<s_ptr<Triangle> >  refine();
#endif 


  };
}

#endif
