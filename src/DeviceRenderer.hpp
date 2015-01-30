#include <cstdio>
#include <vector>
#include <curand_kernel.h>

#ifndef __CUDACC__
#include "rapidjson/document.h"
#endif 
#include "SceneContainer.hpp"
#include "Ray.hpp"
#include "Hit.hpp"
#include "ray_defs.hpp"
#include "Renderer.hpp"



#ifndef  MM_PHONG_RENDERER
#define MM_PHONG_RENDERER

namespace mm_ray {

  template<typename Accel, typename RenderFunction>
  __global__
  void Device_Render_Pixel(Scene const& scene, 
			   Accel const& objects, 
			   int region_offset_x,
			   int region_offset_y,
			   curandState* globalState,
			   RenderFunction& render_f,
			   Vec3* output_buffer){

    //Declare a shared block that is the size of blockDim
    extern __shared__ Vec3 pixel_samples[];

    int thread_id_x = threadIdx.x + blockDim.x * blockIdx.x;

    int thread_id = threadIdx.y * blockDim.x * gridDim.x + thread_id_x;
    
    int pixel_offset_x = threadIdx.x + region_offset_x;
    int pixel_offset_y = threadIdx.y + region_offset_y;

    float rand_x = curand_uniform( &globalState[thread_id] );
    float rand_y = curand_uniform( &globalState[thread_id] );

    Real_t norm_x = ((pixel_offset_x + rand_x) / scene.output[0] - .5) * scene.viewport[0];
    Real_t norm_y = ((pixel_offset_y + rand_y) / scene.output[1] - .5) * scene.viewport[1];
    
    render_f(scene, 
	     objects, 
	     norm_x, 
	     norm_y, 
	     &pixel_samples[blockIdx.x])
    
    __syncthreads();
    //Sum the other dimension
    if (blockIdx.x == 0) {
      Vec3 accum = 0;
      for (int i = 0; i < blockDim.y; i++)
	accum += pixel_samples[i];
      output_buffer[(pixel_offset_y * scene.output[0] + pixel_offset_x) * 3] = accum / (Real_t)blockDim.x;
    }
  }

  __global__ void Setup_Generator ( curandState * state, unsigned long seed ){
    int id = threadIdx.x + blockDim.x * blockIdx.x;
    curand_init ( seed, id, 0, &state[id] );
  }

  template <typename Accel, typename RenderFunction>  
  void Device_Renderer(Scene const& scn, Accel const& acc, Vec3* results, RenderFunction& render){
    
    //Allocate memory for the output image
    Vec3* dev_mem;
    cudaMalloc((void**)&dev_mem, scn.output[0] * scn.output[1] * sizeof(Vec3));

    //Setup the random number generator for the pixel sampling
    curandState* random_state;
    cudaMalloc((void**)&dev_state, 
	       scn.render_block_y * scn.render_block_x * scn.samples * sizeof(curandState));
    Setup_Generator<<<scn.samples, scn.render_block_y * scn.render_block_x>>>(random_state,
									      time(NULL));
    dim3 block(scn.samples);
    int chunk_x = scn.output[0] / scn.render_block_x + scn.output[0] % scn.render_block_x != 0;
    int chunk_y = scn.output[1] / scn.render_block_y + scn.output[1] % scn.render_block_y != 0;
    for (int j = 0; j < chunk_y; j++){
      int grid_y_size = min((j + 1) * scn.render_block_y, scn.output[1]) - j * scn.render_block_y;
      for (int i = 0; i < chunk_x; i++){
	int grid_x_size = min((i + 1) * scn.render_block_x, scn.output[0]) - i * scn.render_block_x;
	dim3 grid(grid_x_size, grid_y_size);

	Device_Render_Pixel<<<block, grid, scn.samples * sizeof(Vec3)>>>(scn, acc, 
									 i * scn.render_block_x,
									 i * scn.render_block_y,
									 random_state,
									 render,
									 dev_mem);
      }
    }
    gpuErrchk(cudaThreadSynchronize());
    gpuErrchk(cudaMemcpy(results, dev_mem, sizeof(Vec3) * scn.output[0] * scn.output[1], 
			 cudaMemcpyDeviceToHost));
    cudaFree(dev_mem);
    cudaFree(random_state);
  }

}
#endif 



