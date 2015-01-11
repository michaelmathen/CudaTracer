#include <cstdio>
#include <vector>
#include <cmath>
#include "SceneContainer.hpp"
#include "Ray.hpp"
#include "Hit.hpp"
#include "ray_defs.hpp"

#include "DistanceRenderer.hpp"

using namespace std;


namespace mm_ray {

  template<typename Accel>
  __global__
  void render_pixel(Scene const& scene,
		    Accel const& objects,
		    int scene_x,
		    int scene_y,
		    Real_t* pixel_out){
  
    /* 
       Renders a single pixel in the image
    */
    int px = threadIdx.x + blockDim.x * blockIdx.x;
    int py = threadIdx.y + blockDim.y * blockIdx.y;
  
    Real_t norm_i = ((px / (Real_t)scene.output[0]) - .5) * scene.viewport[0];
    Real_t norm_j = ((py / (Real_t)scene.output[1]) - .5) * scene.viewport[1];
  
    Vec3 direc;
    direc = norm_i * scene.cam_right + norm_j * scene.cam_up + scene.cam_dir;

    //Normalize ray
    direc = direc / mag(direc);

    Ray ray(direc, scene.cam_loc);

    //Run our ray tracing algorithm

    Hit prop;
    objects.intersect(ray, prop);

    int pix_ix = (py * scene_x + px) * 3;

    //Use this mapping to map 0 to infinity to 0 to 1
    Real_t distance = 1 / (1 + prop.distance);
    pixel_out[pix_ix] = distance;
    pixel_out[pix_ix + 1] = distance;
    pixel_out[pix_ix + 2] = distance;

  }


  template<typename Accelerator>
  void DistanceRenderer<Accelerator>::Render(){
  
    int image_size_x = this->host_scene.output[0];
    int image_size_y = this->host_scene.output[1];


    //A block is a 16 x 16 chunk
    dim3 block;
    //A grid is a bunch of blocks in a chunk of our image
    dim3 grid(image_size_x, image_size_y);

    Real_t* device_pixel_buffer;
    cudaMalloc(&device_pixel_buffer, image_size_x * image_size_y * 3 * sizeof(Real_t));

    //Finally run the raytracing kernel
    render_pixel<Accelerator><<<grid, block>>>(this->host_scene,
					       this->host_accel,
					       image_size_x,
					       image_size_y,
					       device_pixel_buffer);

    cudaThreadSynchronize();

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


  template<typename Accel>
  Renderer<Accel>*
  DistanceBuilder<Accel>::operator()(rapidjson::Value& val_obj, 
				     Scene const& scn,
				     Accel const& accelerator,
				     std::vector<Geometry*>& geometry) const {
    return new DistanceRenderer<Accel>(scn, accelerator);
  }

  template class DistanceBuilder<SceneContainer>;
  template class DistanceRenderer<SceneContainer>;
}

