#include "ray_defs.hpp"

#include "Ray.hpp"
#include "Hit.hpp"
#include "Geometry.hpp"


#ifndef MM_SPHERE
#define MM_SPHERE


namespace mm_ray {
  class Sphere : public Geometry {
    Vec3 center;
    Real_t radius;
    s_ptr<Material> material;
  
  public:

    __host__ __device__ Sphere(){}
    
    __host__ __device__ Sphere(Vec3& center, Real_t radius, s_ptr<Material> mat) :
      center(center), radius(radius), material(mat) {

    }

    __host__ __device__ virtual s_ptr<Material> getMaterial(){
      return material;
    }
    
    __host__ __device__ inline  virtual bool intersectBox(Vec3& l,
							 Vec3& u){
      Real_t r2 = radius * radius;
      Real_t dmin = 0;
      for(int i = 0; i < 3; i++) {
	if (center[i] < l[i])
	  dmin += (center[i] - l[i]) * (center[i] - l[i]);
	else if (center[i] > u[i])
	  dmin += (center[i] - u[i]) * (center[i] - u[i]);
      }
      return dmin <= r2;
    }

    __host__ __device__ inline virtual void intersectRay(Ray& ray, Hit& prop){

      //Cast a ray from the ray origin to the sphere center
      Vec3 L = center - ray.origin;

      //Project that ray onto the light direction
      Real_t tca = dot(L, ray.direc);

      prop.distance = INFINITY;
      
      //If the projection is behind the ray origin then we don't intersect
      if (tca < 0) {
	return ;
      }

      //Calculate the distance squared from sphere center to point on the
      //ray perpendicular to the sphere center
      Real_t d2 = dot(L, L) - tca * tca;
      //Check that the point is inside of the sphere
      if (d2 > radius * radius) {
	return ;
      }
      
      prop.distance = tca - sqrt(radius * radius - d2);
  
      Vec3 vtmp = prop.distance * ray.direc;
      prop.hit_location = vtmp + ray.origin;
      Vec3 normal = prop.hit_location - center;
      
      prop.normal = normal * (1 / mag(normal));
      prop.material = material;
    }

    __host__ __device__ inline virtual Vec3 getCenter() {
      return center;
    }
      
    static const Virtual_Type_Val type_id = SPHERE;
  
  };
}
#endif
