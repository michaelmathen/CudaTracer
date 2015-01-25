#include <vector>
#include "BVHTreeSimple.hpp"
#include "HostRenderer.hpp"

namespace mm_ray {

  using namespace std;

  template<typename Accel>
  void render_pixel(Scene const* scene, 
		    Accel const* objects,
		    int px_image,
		    int py_image,
		    vector<Real_t>& pixel_out){
    
    //Index into the image buffer
    int pix_ix = (py_image * scene->output[0] + px_image) * 3;

    //Normalized into the viewport 
    Real_t norm_i = ((px_image / (Real_t)scene->output[0]) - .5) * scene->viewport[0];
    Real_t norm_j = ((py_image / (Real_t)scene->output[1]) - .5) * scene->viewport[1];

    Vec3 direc = norm_i * scene->cam_right + norm_j * scene->cam_up + scene->cam_dir;

    //Normalize ray
    direc = direc / mag(direc);

    Ray ray(direc, scene->cam_loc);

    //Run our ray tracing algorithm

    Hit prop;
    objects->Intersect(ray, prop);

    if (prop.distance < INFINITY) {
      PhongMaterial const* pmat = static_cast<PhongMaterial const*>(prop.material);

      //Draw a ray to each light
      Vec3 pixel_color = pmat->color * pmat->amb_light;
      //printf("prop.distance = %f\n", pmat->color[1]);
    
      for (int i = 0; i < objects->getLightNumber(); i++){
	//We only support point lights so this will not be accurate for area lights
	Geometry const* light_source = objects->getLight(i);
      
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
    
      pixel_out[pix_ix] += pixel_color[0];
      pixel_out[pix_ix + 1] += pixel_color[1];
      pixel_out[pix_ix + 2] += pixel_color[2];
    } 

  }

  template<typename Accel> 
  HostRenderer<Accel>::HostRenderer(Scene const* scn, Accel const* acc) : 
    Renderer<Accel>(scn, acc){}
  
  template<typename Accel>
  void HostRenderer<Accel>::Render(){
    this->output_buffer.resize(this->host_scene->output[0] * this->host_scene->output[1] * 3);
    for (int i = 0; i < this->host_scene->output[0]; i++){
      for (int j = 0; j < this->host_scene->output[1]; j++){
	render_pixel<Accel>(this->host_scene,
			    this->host_accel,
			    i,
			    j,
			    this->output_buffer);
				  
      }
    }
  }
  template class HostRenderer<SceneContainer>;
  template class HostRenderer<BVHTreeSimple>;

}
