#include "TriangleMesh.hpp"
#include "Material.hpp"
#include "SceneAllocator.hpp"
#include "ray_defs.hpp"
#include "Geometry.hpp"

#ifndef MM_TRIANGLE
#define MM_TRIANGLE
namespace mm_ray {

  class Triangle : public Geometry {

    //Pointer to the trianlge mesh that this triange is from
    s_ptr<TriangleMesh> mesh_ptr;
    //Index of a triangle
    unsigned int triangle_index;
  public:    


    __host__ __device__ Triangle() {}

    
    
    Triangle(s_ptr<TriangleMesh> const& triangle_mesh, unsigned int tri_ix) :
      mesh_ptr(triangle_mesh),
      triangle_index(tri_ix) {}

    /*
    __host__ __device__ virtual void intersectRay(Ray& ray, Hit& prop) {
      Tri_vert& tri = mesh_ptr->vertex_indices[triangle_index];
      s_ptr<Vec3> vertices = mesh_ptr->triangle_vertices;
      Vec3 v1 = vertices[tri.x];
      Vec3 v2 = vertices[tri.y];
      Vec3 v3 = vertices[tri.z];
      
    }
    */
    __host__ __device__ virtual void intersectRay(Ray& ray, Hit& prop) {
      //Todo this is pretty inefficient
      Tri_vert& tri = mesh_ptr->vertex_indices[triangle_index];
      s_ptr<Vec3> vertices = mesh_ptr->triangle_vertices;
      Vec3 v1 = vertices[tri.x];
      Vec3 v2 = vertices[tri.y];
      Vec3 v3 = vertices[tri.z];
      
      Vec3 normal = cross(v2 - v1, v3 - v1);
      normal = normal / mag(normal);

      Real_t t = dot( v1 - ray.origin, normal)  / dot(ray.direc, normal);
      if (t <= 0.0) {
	prop.distance = INFINITY;
	return;
      }
      
      Vec3 x = ray.direc * t + ray.origin;
      
      if (dot(cross(v2 - v1, x - v1), normal) >= 0 &&
	  dot(cross(v3 - v2, x - v2), normal) >= 0 &&
	  dot(cross(v1 - v3, x - v3), normal) >= 0) {
	Real_t tmp_val = dot(normal, ray.direc);
	prop.normal = -tmp_val / (Real_t)fabs(tmp_val) * normal;
	prop.distance = t;
	//prop.normal = normal;
	prop.material = mesh_ptr->material;
	prop.hit_location = x;
	return ;
      } else {
	prop.distance = INFINITY;
	return ;
      }
    }

    __host__ __device__ inline virtual Vec3 getCenter() {
      
      Vec3 v1 = mesh_ptr->triangle_vertices[mesh_ptr->vertex_indices[triangle_index].x];
      Vec3 v2 = mesh_ptr->triangle_vertices[mesh_ptr->vertex_indices[triangle_index].y];
      Vec3 v3 = mesh_ptr->triangle_vertices[mesh_ptr->vertex_indices[triangle_index].z];
      return Vec3((v1 + v2 + v3) / (Real_t)3.0);
    }

    __host__ __device__ virtual s_ptr<Material> getMaterial(){
      return mesh_ptr->material;
    }
    
    const static Virtual_Type_Val type_id = TRIANGLE;
  };
}

#endif
