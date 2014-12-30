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
    
    __host__ __device__ static inline void triangle_intersection(Vec3& v1,
								 Vec3& v2,
								 Vec3& v3,
								 Ray  ray,
								 Hit& prop){
      //Cast a ray at one of the voxels
      //The distance to a point on the plane
      //The point in space where we intersect the plane
      
      Real_t dist_to_intersect = dot(v1 - ray.origin, ray.direc);
      Vec3 intersection_pnt = dist_to_intersect * ray.direc + ray.origin;
      Vec3 normal = cross(v2 - v1, v3 - v1);
      
      //Now time to check if the point is inside of the triangle
      if ((dot(cross(v2 - v1, intersection_pnt - v1), normal)  >= 0) &&
	  (dot(cross(v3 - v2, intersection_pnt - v2), normal) >= 0) &&
	  (dot(cross(v1 - v3, intersection_pnt - v3), normal) >= 0)) {
	prop.hit = true;
	prop.distance = dist_to_intersect;
	prop.normal = normal;
      }
    }
    
  public:

    Triangle() {}
    
    Triangle(s_ptr<TriangleMesh> const& triangle_mesh, unsigned int tri_ix) :
      mesh_ptr(triangle_mesh),
      triangle_index(tri_ix) {}
    
    
    __host__ __device__ virtual void intersectRay(Ray& ray, Hit& prop) {
      triangle_intersection(mesh_ptr->triangle_vertices[mesh_ptr->vertex_indices[triangle_index][0]],
			    mesh_ptr->triangle_vertices[mesh_ptr->vertex_indices[triangle_index][1]],
			    mesh_ptr->triangle_vertices[mesh_ptr->vertex_indices[triangle_index][2]],
			    ray,
			    prop);
      if (prop.hit)
	prop.material = mesh_ptr->material;
    }

    __host__ __device__ inline virtual Vec3 getCenter() {
      
      Vec3 v1 = mesh_ptr->triangle_vertices[mesh_ptr->vertex_indices[triangle_index][0]];
      Vec3 v2 = mesh_ptr->triangle_vertices[mesh_ptr->vertex_indices[triangle_index][1]];
      Vec3 v3 = mesh_ptr->triangle_vertices[mesh_ptr->vertex_indices[triangle_index][2]];
      return Vec3((v1 + v2 + v3) / (Real_t)3.0);
    }

    __host__ __device__ virtual s_ptr<Material> getMaterial(){
      return mesh_ptr->material;
    }
    
    const static Virtual_Type_Val type_id = TRIANGLE;
  };
}

#endif
