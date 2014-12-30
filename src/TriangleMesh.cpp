#include "SceneAllocator.hpp"
#include "TriangleMesh.hpp"
#include "Triangle.hpp"
#include "ray_defs.hpp"

namespace mm_ray {
  using namespace std;

  TriangleMesh::TriangleMesh(std::string const& fname, s_ptr<Material> material){
    
  }
  
  vector<s_ptr<Triangle> > TriangleMesh::refine(){
    /*
      Break the triangle mesh into sub triangles.
    */
    vector<s_ptr<Triangle> > triangle_arr;
    triangle_arr.resize(number_of_triangles);
    
    for (unsigned int i = 0; i < number_of_triangles; i++){
      triangle_arr[i] = scene_alloc<Triangle>(Triangle(get_offset<TriangleMesh>(this), i));
    }
    return triangle_arr;
  }
}
