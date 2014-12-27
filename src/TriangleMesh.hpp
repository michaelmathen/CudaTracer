#include "cuda_defs.h"
#include "SceneAllocator.hpp"
#include "Geometry.hpp"

namespace mm_ray {
  class TriangleMesh : public Geometry {

    s_ptr<Vec3> triangle_vertices;
    s_ptr<int[3]> vertex_indices;
    int number_of_triangles;

    __host__ __device__ static inline void triangle_intersection(Vec3& v1, Vec3& v2, Vec3& v3, Ray& ray, Hit& prop){
      //First find out where the ray intersects the triangle plane.
      Vec3 plane_normal = cross(v2 - v1, v3 - v1);
      //Cast a ray at one of the voxels
      Vec3 ray_cst = v1 - ray.origin;
      //The distance to a point on the plane
      Real_t plane_distance = dot(ray_cst, ray.direc);
      //The point in space where we intersect the plane
      Vec3 intersection_pnt = plane_distance * ray.direc + ray.origin;

      //Now time to check if the point is inside of the triangle
      prop.hit = false;
      prop.hit &= 
    }
    
  public:  
    TriangleMesh(){}

    __host__ __device__ virtual void intersectRay(Ray& ray, Hit& prop) {
      for (int i = 0; i < number_of_triangles; i++){
	Vec3 v1 = triangle_vertices[vertex_indices[i][0]];
	Vec3 v2 = triangle_vertices[vertex_indices[i][1]];
	Vec3 v3 = triangle_vertices[vertex_indices[i][2]];
	TriangleMesh::triangle_intersection(v1, v2, v3, ray, prop);
      }
    }

  };
}
