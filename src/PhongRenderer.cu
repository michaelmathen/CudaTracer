#include <cstdio>
#include <vector>
#include <curand_kernel.h>

#include "SceneContainer.hpp"
#include "SceneObjects.hpp"
#include "Ray.hpp"
#include "Hit.hpp"
#include "PhongRenderer.hpp"
#include "ray_defs.hpp"

using namespace std;


namespace mm_ray {
  
template<typename Accel>
__global__
void render_pixel_phong(Scene scene,
			Accel objects,
			void* dev_memory,
			int scene_x,
			int scene_y,
			Real_t* pixel_out,
			float* random_values,
			int sample_run){
  
  //Copy over our memory space
  //so that now all of our pointers should work like they did on the host
  device_buffer = (char*)dev_memory;
  
  /* 
     Renders a single pixel in the image
   */

  int px_i = threadIdx.x + blockDim.x * blockIdx.x;
  int py_i = threadIdx.y + blockDim.y * blockIdx.y;

  int pix_ix = (py_i * scene_x + px_i);
  
  Real_t px_f = (px_i + random_values[pix_ix]) +  (1.0 / (Real_t)scene.samples) * sample_run;
  Real_t py_f = (py_i + random_values[pix_ix]) +  (1.0 / (Real_t)scene.samples) * sample_run;

  pix_ix = pix_ix * 3;
  
  Real_t norm_i = ((px_f / (Real_t)scene.output[0]) - .5) * scene.viewport[0];
  Real_t norm_j = ((py_f / (Real_t)scene.output[1]) - .5) * scene.viewport[1];

  Vec3 direc= norm_i * scene.cam_right + norm_j * scene.cam_up + scene.cam_dir;

  //Normalize ray
  direc = direc / mag(direc);

  Ray ray(direc, scene.cam_loc);

  //Run our ray tracing algorithm

  Hit prop;
  objects.intersect(ray, prop);

  if (prop.hit) {
    s_ptr<PhongMaterial> pmat = static_pointer_cast<PhongMaterial, Material>(prop.material);


    //Draw a ray to each light
    Vec3 pixel_color = pmat->color * pmat->amb_light;

    for (int i = 0; i < objects.getLightNumber(); i++){
      //We only support point lights so this will not be accurate for area lights
      s_ptr<Geometry> light_source = objects.getLight(i);
      
      //Get a ray going from our center to the light source
      Vec3 ctmp = light_source->getCenter();
      Vec3 new_ray = ctmp - prop.hit_location;

      Real_t length_to_light = mag(new_ray);
      
      new_ray = new_ray / length_to_light;
      
      //Shoot ray towards the light source and see if we hit before the light
      Hit shadow_prop;
      
      Vec3 new_ray_origin = prop.hit_location;
      new_ray_origin += (prop.normal * 1e-6f);
      
      Ray shadow_ray(new_ray, new_ray_origin);
      objects.intersect(shadow_ray, shadow_prop);
     
      Real_t diff = pmat->diff_light;
      Real_t spec = pmat->spec_light;
      Real_t shine = pmat->shine;
      Vec3 refl_dir = 2 * dot(new_ray, prop.normal) * prop.normal;
      refl_dir -= new_ray;
      //Now calculate the distance to the light source
      
      Vec3 light_contr = diff * dot(new_ray, prop.normal) * pixel_color;
      light_contr += spec * pow(-dot(refl_dir, direc), shine) * pixel_color;
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


  
template<typename Accelerator>
PhongRenderer<Accelerator>::PhongRenderer(Scene const& scene, Accelerator const& accel) : Renderer<Accelerator>(scene, accel) {}

template<typename Accelerator>
void PhongRenderer<Accelerator>::Render(){
  
  int image_size_x = (this->host_scene.output[0] / 16 + 1) * 16;
  int image_size_y = (this->host_scene.output[1] / 16 + 1) * 16;


  //A block is a 16 x 16 chunk
  dim3 block(16, 16);
  //A grid is a bunch of blocks in a chunk of our image
  dim3 grid(image_size_x / 16, image_size_y / 16);

  Real_t* device_pixel_buffer;
  cudaMalloc(&device_pixel_buffer, image_size_x * image_size_y * 3 * sizeof(Real_t));

  //Since we don't use pointers anywhere we can just serialize this one object
  //and everthing should work realy nice on the gpu
  void* dev_Memory = serialize_scene_alloc();

  curandGenerator_t sobol_generator;
  curand_check(curandCreateGenerator(&sobol_generator, CURAND_RNG_PSEUDO_DEFAULT));

  float* random_values;
  cudaMalloc(&random_values, sizeof(float) * image_size_x * image_size_y);
  //curandDirectionVectors32_t direc_vec; 
  //curandGetDirectionVector32(&direc_vec, CURAND_DIRECTION_VECTORS_32_JOEKUO6);
  curand_check(curandSetPseudoRandomGeneratorSeed(sobol_generator, (unsigned long long)time(NULL)));
    
  zeroMemory<<<image_size_x * image_size_y * 3, 1>>>(device_pixel_buffer);
  

  gpuErrchk(cudaThreadSynchronize());  
  //Finally run the raytracing kernel

  for (int i = 0; i < this->host_scene.samples; i++){
    curand_check(curandGenerateUniform(sobol_generator, random_values, image_size_x * image_size_y));
    render_pixel_phong<Accelerator><<<grid, block>>>(this->host_scene,
						     this->host_accel,
						     dev_Memory,
						     image_size_x,
						     image_size_y,
						     device_pixel_buffer,
						     random_values,
						     i);
  }
  gpuErrchk(cudaThreadSynchronize());
  average_samples<<<image_size_x * image_size_y * 3, 1>>>(device_pixel_buffer, this->host_scene.samples);
  gpuErrchk(cudaThreadSynchronize());

  curand_check(curandDestroyGenerator(sobol_generator));
  cudaFree(random_values);
  cudaError_t error = cudaGetLastError();
  if(error != cudaSuccess)
  {
    // print the CUDA error message
    printf("CUDA error: %s\n", cudaGetErrorString(error));
  }

  
  //Copy the entire buffer to a temporary buffer
  int buffer_size = image_size_x * image_size_y * 3;
  vector<Real_t> tmp_buffer;
  tmp_buffer.resize(buffer_size);
  
  cudaMemcpy(&*(tmp_buffer.begin()), device_pixel_buffer,
	     sizeof(Real_t) * buffer_size,
	     cudaMemcpyDeviceToHost);

  
  this->output_buffer.resize(this->host_scene.output[1] * this->host_scene.output[0] * 3);
  
  for (int i = 0; i < this->host_scene.output[1]; i++){
    for (int j = 0; j < this->host_scene.output[0] * 3; j++){
      this->output_buffer[i * this->host_scene.output[0] * 3 + j] = tmp_buffer[i * image_size_x * 3 + j];
    }
  }
  
  cudaFree(device_pixel_buffer);
}

  template<typename Accelerator>
  PhongRenderer<Accelerator>::~PhongRenderer(){
    //Todo delete the memory associated with the device pointers
  }

  template class PhongRenderer<SceneContainer>;
}

