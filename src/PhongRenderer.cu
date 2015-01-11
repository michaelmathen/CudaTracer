#include <cstdio>
#include <vector>
#include <curand_kernel.h>

#include "SceneContainer.hpp"
#include "Geometry.hpp"
#include "Material.hpp"
#include "Ray.hpp"
#include "Hit.hpp"
#include "PhongRenderer.hpp"
#include "ray_defs.hpp"

using namespace std;


namespace mm_ray {
  
template<typename Accel>
__global__
void render_pixel_phong(Scene const* scene,
			Accel const* objects,
			int scene_lower_x,
			int scene_lower_y,
			int scene_width,
			int chunk_width,
			Real_t* pixel_out,
			float* random_values,
			int sample_run){
  
  //Copy over our memory space
  //so that now all of our pointers should work like they did on the host
  
  /* 
     Renders a single pixel in the image
   */

  //This is the current location in the output image block
  int pixel_block_x = threadIdx.x + blockDim.x * blockIdx.x;
  int pixel_block_y = threadIdx.y + blockDim.y * blockIdx.y;
  //the index into the random numbers
  int rand_index = (pixel_block_y * chunk_width + pixel_block_x) * 2;
  float rand_x = random_values[rand_index];
  float rand_y = random_values[rand_index + 1];

  //Index into the image itself
  int px_image = pixel_block_x + scene_lower_x;
  int py_image = pixel_block_y + scene_lower_y;

  //Index into the image buffer
  int pix_ix = (py_image * scene_width + px_image) * 3;

  //Normalized into the viewport 
  Real_t norm_i = (((px_image + rand_x) / (Real_t)scene->output[0]) - .5) * scene->viewport[0];
  Real_t norm_j = (((py_image + rand_y) / (Real_t)scene->output[1]) - .5) * scene->viewport[1];

  Vec3 direc = norm_i * scene->cam_right + norm_j * scene->cam_up + scene->cam_dir;

  //Normalize ray
  direc = direc / mag(direc);

  Ray ray(direc, scene->cam_loc);

  //Run our ray tracing algorithm

  Hit prop;
  objects->intersect(ray, prop);

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
      objects->intersect(shadow_ray, shadow_prop);
     
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

__global__
void zeroMemory(Real_t* pixel_mem) {
  int px = threadIdx.x + blockDim.x * blockIdx.x;
  pixel_mem[px] = 0;
}

__global__
void average_samples(Real_t* pixel_mem, int samples) {
  int px = threadIdx.x + blockDim.x * blockIdx.x;
  pixel_mem[px] = pixel_mem[px] / samples;
}


  
template<typename Accel>
PhongRenderer<Accel>::PhongRenderer(Scene const* scene, Accel const* accel) 
  : Renderer<Accel>(scene, accel) {}

template<typename Accel>
void PhongRenderer<Accel>::Render(){

  int image_size_x;
  int image_size_y;
  if (this->host_scene->output[0] % 16 == 0)
    image_size_x = this->host_scene->output[0];
  else
    image_size_x = (this->host_scene->output[0] / 16 + 1) * 16;

  if (this->host_scene->output[1] % 16 == 0)
    image_size_y = this->host_scene->output[1];
  else
    image_size_y = (this->host_scene->output[1] / 16 + 1) * 16;
  

  Real_t* device_pixel_buffer;
  cudaMalloc(&device_pixel_buffer, image_size_x * image_size_y * 3 * sizeof(Real_t));

  curandGenerator_t sobol_generator;
  curand_check(curandCreateGenerator(&sobol_generator, CURAND_RNG_PSEUDO_DEFAULT));

  int rbx = this->host_scene->render_block_x;
  int rby = this->host_scene->render_block_y;

  float* random_values;
  cudaMalloc(&random_values,
	     sizeof(float) * rbx * rby * 16 * 16 * 2);

  curand_check(curandSetPseudoRandomGeneratorSeed(sobol_generator, (unsigned long long)time(NULL)));
    
  zeroMemory<<<image_size_x * image_size_y * 3, 1>>>(device_pixel_buffer);
  
  gpuErrchk(cudaThreadSynchronize());

  //A block is a 16 x 16 chunk
  dim3 block(16, 16);

  //A grid is a bunch of blocks in a chunk
  int chunk_x = image_size_x % rbx == 0 ?  image_size_x / (rbx * 16) : image_size_x / (rbx * 16) + 1;
  int chunk_y = image_size_y % rby == 0 ?  image_size_y / (rby * 16) : image_size_y / (rby * 16) + 1;
  
  int sample_num = this->host_scene->samples;
  for (int i = 0; i < chunk_y; i++){
    int tmp_y = min(image_size_y / 16, rby * (i + 1));
    int grid_y_size = tmp_y - rby * i;
    
    for (int j = 0; j < chunk_x; j++){
      int tmp_x = min(image_size_x / 16, rbx * (j + 1));
      int grid_x_size = tmp_x - rbx * j;
      dim3 grid(grid_x_size, grid_y_size);
      //printf("grid_x_size= %d grid_y_size %d\n", grid_x_size, grid_y_size);
      for (int k = 0; k < sample_num; k++){
	
	curand_check(curandGenerateUniform(sobol_generator,
					   random_values,
					   grid_x_size * grid_y_size * 16 * 16 * 2));

	render_pixel_phong<Accel><<<grid, block>>>(this->host_scene,
						   this->host_accel,
						   16 * rbx * j,
						   16 * rby * i,
						   image_size_x,
						   grid_x_size,
						   device_pixel_buffer,
						   random_values,
						   i);
      }
    }
  }
  gpuErrchk(cudaThreadSynchronize());
  average_samples<<<image_size_x * image_size_y * 3, 1>>>(device_pixel_buffer, sample_num);
  gpuErrchk(cudaThreadSynchronize());

  curand_check(curandDestroyGenerator(sobol_generator));
  gpuErrchk(cudaFree(random_values));
  
  //Copy the entire buffer to a temporary buffer
  int buffer_size = image_size_x * image_size_y * 3;
  vector<Real_t> tmp_buffer;
  tmp_buffer.resize(buffer_size);
  
  cudaMemcpy(&*(tmp_buffer.begin()), device_pixel_buffer,
	     sizeof(Real_t) * buffer_size,
	     cudaMemcpyDeviceToHost);

  
  this->output_buffer.resize(this->host_scene->output[1] * this->host_scene->output[0] * 3);
  
  for (int i = 0; i < this->host_scene->output[1]; i++){
    for (int j = 0; j < this->host_scene->output[0] * 3; j++){
      this->output_buffer[i * this->host_scene->output[0] * 3 + j] = tmp_buffer[i * image_size_x * 3 + j];
    }
  }
  
  cudaFree(device_pixel_buffer);
}

  template <typename Accel>
  Renderer<Accel>* PhongBuilder<Accel>::operator()(rapidjson::Value& , 
						   Scene const* scn,
						   Accel const* accel,
						   std::vector<Geometry*>& geom) const {
    return new PhongRenderer<Accel>(scn, accel);
  }

  template class PhongRenderer<SceneContainer>;
  template class PhongBuilder<SceneContainer>;
}

