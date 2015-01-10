#include "ray_defs.hpp"
#include "Material.hpp"
#include "Ray.hpp"

#include "Hit.hpp"
#include "GeometryData.hpp"

//#include "SceneAllocator.hpp"

#ifndef MM_GEOMETRY
#define MM_GEOMETRY

namespace mm_ray {

  enum Geom_t {
    SPHERE,
    TRIANGLE,
    POINT_LIGHT
  };

  class Geometry : public Managed{
    const Geom_t geometry_type;
  public:
    Geometry(Geom_t geom_t);
    __host__ __device__ void intersectRay(Ray const& ray, Hit& prop) const;
    __host__ __device__ Vec3 getLight() const;
    __host__ __device__ Material const* getMaterial() const;
    __host__ __device__ bool isLight() const;
    __host__ __device__ Vec3 getCenter() const;
  };


  class Sphere : public Geometry {
    Vec3 center;
    Real_t radius;
    Material const* material;
  
  public:

    __host__ __device__ Sphere() :
      Geometry(SPHERE)
    {}
    
    __host__ __device__ Sphere(Vec3 const& center, Real_t radius, Material const* mat) :
      Geometry(SPHERE),
      center(center),
      radius(radius),
      material(mat)
    {}

    __host__ __device__ inline Material const* getMaterial() const {
      return material;
    }
    
    __host__ __device__ inline bool intersectBox(Vec3& l,
						 Vec3& u) const {
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

    __host__ __device__ inline void intersectRay(Ray const& ray, Hit& prop) const {
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

    __host__ __device__ inline Vec3 getCenter() const {
      return center;
    }
    
    __host__ __device__ inline bool isLight() const {
      return false;
    }

    __host__ __device__ inline Vec3 getLight() const {
      Vec3 v1 = 0;
      return v1;
    }
      
  };
  

  class Triangle : public Geometry {

    //Pointer to the trianlge mesh that this triange is from
    TriangleMesh const* mesh_ptr;
    //Index of a triangle
    unsigned int triangle_index;
  public:    


      __host__ __device__ Triangle() :
	Geometry(TRIANGLE)
      {}

    
    
    Triangle(TriangleMesh const* triangle_mesh, unsigned int tri_ix) :
      Geometry(TRIANGLE),
      mesh_ptr(triangle_mesh),
      triangle_index(tri_ix) {}

    __host__ __device__ void inline intersectRay(Ray const& ray, Hit& prop) const {
      //Todo this is pretty inefficient
      Tri_vert& tri = mesh_ptr->vertex_indices[triangle_index];
      Vec3* vertices = mesh_ptr->triangle_vertices;
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

    __host__ __device__ inline Vec3 getCenter() const {
      
      Vec3 v1 = mesh_ptr->triangle_vertices[mesh_ptr->vertex_indices[triangle_index].x];
      Vec3 v2 = mesh_ptr->triangle_vertices[mesh_ptr->vertex_indices[triangle_index].y];
      Vec3 v3 = mesh_ptr->triangle_vertices[mesh_ptr->vertex_indices[triangle_index].z];
      return Vec3((v1 + v2 + v3) / (Real_t)3.0);
    }

    __host__ __device__ inline Material const* getMaterial() const {
      return mesh_ptr->material;
    }

    __host__ __device__ inline
    bool isLight() const {
      return false;
    }

    __host__ __device__ inline
    Vec3 getLight() const {
      Vec3 v1 = 0;
      return v1;
    }
  };

  class PointLight : public Geometry {

    Vec3 location;
    Vec3 illumination;
  public:

    __host__ __device__ PointLight() :
      Geometry(POINT_LIGHT)
    {}
  
    PointLight(Vec3 illumination, Vec3 location)
      :
      Geometry(POINT_LIGHT),
      location(location),
      illumination(illumination)
    {}

    __host__ __device__ void intersectRay(Ray const& ray, Hit& prop) const {
      //You don't hit a point
      (void)ray;
      (void)prop;
    }

    __host__ __device__  bool isLight() const {
      return true;
    }
    
    __host__ __device__ Vec3 getCenter() const {
      return location;
    }

    __host__ __device__ Vec3 getLight() const {
      return illumination;
    }

    __host__ __device__ Material const* getMaterial() const {
      //We don't have a material... Hopefully who ever calls this realizes it.
      return NULL;
    }
    
  };


  /*
    Now the definitions of the geometry class which is just a big case table 
    so that we can have virtual inheritance
   */

  inline Geometry::Geometry(Geom_t geom_t)
      : geometry_type(geom_t)
  {}
    
    /*
      This is a bit of a hack. We get virtual methods without virtual methods.
    */
  __host__ __device__ inline void Geometry::intersectRay(Ray const& ray, Hit& prop) const {
    switch (this->geometry_type) {		       
    case(SPHERE):			
      static_cast<Sphere const*>(this)->intersectRay(ray, prop);
      break;
    case(TRIANGLE):			  
      static_cast<Triangle const*>(this)->intersectRay(ray, prop);  
      break;
    case(POINT_LIGHT):		         
      static_cast<PointLight const*>(this)->intersectRay(ray, prop);	
      break;
    }    
  }
  __host__ __device__ inline Vec3 Geometry::getLight() const {
    switch (this->geometry_type) {		       
    case(SPHERE):			
      return static_cast<Sphere const*>(this)->getLight();
    case(TRIANGLE):			  
      return static_cast<Triangle const*>(this)->getLight();
    case(POINT_LIGHT):		   
      return static_cast<PointLight const*>(this)->getLight();
    }
    Vec3 v1 = (Real_t)0.;
    return v1;
  }

  __host__ __device__ inline Material const* Geometry::getMaterial() const{
    switch (this->geometry_type) {		       
    case(SPHERE):			
      return static_cast<Sphere const*>(this)->getMaterial();
    case(TRIANGLE):			  
      return static_cast<Triangle const*>(this)->getMaterial();
    case(POINT_LIGHT):		   
      return static_cast<PointLight const*>(this)->getMaterial();
    }
    return NULL;
  }
    
  __host__ __device__ inline bool Geometry::isLight() const {
    switch (this->geometry_type) {
    case(SPHERE):{
      Sphere const* tmp = static_cast<Sphere const*>(this);
      return tmp->isLight();
    }case(TRIANGLE):{
       Triangle const* tmp = static_cast<Triangle const*>(this);
       return tmp->isLight();
    }case(POINT_LIGHT):{	
       PointLight const* tmp = static_cast<PointLight const*>(this);
       return tmp->isLight();
     }
    }
    return false;
  }

    /*
      This gets the rough center of the object. 
     */
  __host__ __device__ inline Vec3 Geometry::getCenter() const {
    switch (this->geometry_type) {
    case(SPHERE):			
      return static_cast<Sphere const*>(this)->getCenter();
    case(TRIANGLE):			  
      return static_cast<Triangle const*>(this)->getCenter();
    case(POINT_LIGHT):		   
      return static_cast<PointLight const*>(this)->getCenter();
    }
    Vec3 v1 = (Real_t)0.;
    return v1;
  }
}

#endif
