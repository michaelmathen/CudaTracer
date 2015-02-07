#include <cstdio>
#include <vector>
#include "SceneContainer.hpp"
#include "Ray.hpp"
#include "Hit.hpp"
#include "ray_defs.hpp"
#include "Renderer.hpp"

#ifndef  MM_PHONG_RENDERER
#define MM_PHONG_RENDERER

namespace mm_ray {

  template<typename Accel>
  struct PhongRenderer {
  public:
    template<typename Accel>
    Vec3 operator()(Scene const& scene,
		    Accel const& objects,
		    Ray const& initial_ray){
      Hit prop;
      objects->Intersect(initial_ray, prop);

      if (prop.distance < INFINITY) {
	PhongMaterial const* pmat = static_cast<PhongMaterial const*>(prop.material);

	//Draw a ray to each light
	Vec3 pixel_color = pmat->color * pmat->amb_light;
	//printf("prop.distance = %f\n", pmat->color[1]);
    
	for (int i = 0; i < objects.getLightNumber(); i++){
	  //We only support point lights so this will not be accurate for area lights
	  Geometry const* light_source = objects.getLight(i);
      
	  //Get a ray going from our center to the light source
	  Vec3 ctmp = light_source->getCenter();
	  Vec3 new_ray = ctmp - prop.hit_location;

	  Real_t length_to_light = mag(new_ray);
      
	  new_ray = new_ray / length_to_light;
      
	  //Shoot ray towards the light source and see if we hit before the light
	  Hit shadow_prop;
      
	  //Vec3 new_ray_origin = prop.hit_location;
	  Vec3 new_ray_origin = prop.hit_location + prop.normal * 1e-6f;

	  Ray shadow_ray(new_ray, new_ray_origin);
	  objects->Intersect(shadow_ray, shadow_prop);
     
	  Real_t diff = pmat->diff_light;
	  Real_t spec = pmat->spec_light;
	  Real_t shine = pmat->shine;

	  Vec3 light_contr = (diff * max(dot(new_ray, prop.normal),0.f) * pmat->color + 
			      spec * pow(max(dot((new_ray - ray.direc) / 2.0f, prop.normal), 0.f), shine) * pmat->color);

	  pixel_color += (Real_t)(shadow_prop.distance > length_to_light) * light_contr * light_source->getLight();
	}
	return pixel_color;
    }
  };
}
#endif 
