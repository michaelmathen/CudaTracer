#include "SceneAllocator.hpp"

#include "SceneObjects.hpp"


namespace mm_ray {
  using namespace std;
  
  unsigned int scene_buff_size;
  unsigned int curr_buff;
  char* host_buffer;
  vector<Virtual_Type_Val> virtual_types;
  vector<unsigned int> virtual_indices;
  
#ifdef __CUDACC__
  //This is the memory buffer for all the scene object data 
  __device__ char* device_buffer;
#endif

  
  
  __host__ __device__ void classMap(char* buffer, Virtual_Type_Val el, unsigned int index) {
    /*
      This kernel function fixes the vtable in the classes that we copied over
     */
    switch (el){
    case MATERIAL:
      new(buffer + index) Material();
      break;
    case PHONG_MAT:
      new(buffer + index) PhongMaterial();
      break;
    case POINTLIGHT:
      new(buffer + index) PointLight();
      break;
    case SPHERE:
      new(buffer + index) Sphere();
      break;
    case AMBIENT_MAT:
      new(buffer + index) AmbientMaterial();
      break;
    case NOT_VT:
      break;
    }
  }
  
  __global__
  void initMemory(char* buffer, Virtual_Type_Val* types, unsigned int* indices) {
    
    int ix = threadIdx.x + blockDim.x * blockIdx.x;
    classMap(buffer, types[ix], indices[ix]);
    
  }

  void expand_scene_buffer(){
    //everytime we increase the buffer size by 1.5 times
    scene_buff_size = scene_buff_size + scene_buff_size;
    char* tmp_buffer = (char*)malloc(scene_buff_size);
    memcpy(tmp_buffer, host_buffer, curr_buff);
    free(host_buffer);
    host_buffer = tmp_buffer;
  }

  
  void* serialize_scene_alloc(){
    /*
      This method copies the device buffer over to gpu memory 
      and sets up the dev_scene to point to it. 
      Then we have to fix the vtables of all the virtual classes that we passed over.
      The classMap kernel uses placement new to fix the vtable pointer.

      Now all you have to do is pass the dev_scene to the
	cuda kernel as an argument.
    */
    char* dev_buffer;
    cudaMalloc(&dev_buffer, curr_buff);
    cudaMemcpy(dev_buffer, host_buffer, curr_buff, cudaMemcpyHostToDevice);

    Virtual_Type_Val* dev_types;
    cudaMalloc(&dev_types, sizeof(Virtual_Type_Val) * virtual_types.size() );
    cudaMemcpy(dev_types, &*virtual_types.begin(),
	       sizeof(Virtual_Type_Val) * virtual_types.size(),
	       cudaMemcpyHostToDevice);

    unsigned int* dev_indices;
    cudaMalloc(&dev_indices, sizeof(unsigned int) * virtual_indices.size());
    cudaMemcpy(dev_indices, &*virtual_indices.begin(),
	       sizeof(unsigned int) * virtual_indices.size(),
	       cudaMemcpyHostToDevice);

    //Fix our vtables
    dim3 grid(virtual_indices.size());
    dim3 block;
    cout << "Number of Virtual objects=" << grid.x  << endl;
    initMemory<<<grid, block>>>(dev_buffer, dev_types, dev_indices);

    cudaThreadSynchronize();
    
    cudaError_t error = cudaGetLastError();
    if(error != cudaSuccess){
      // print the CUDA error message
      printf("CUDA error: %s\n", cudaGetErrorString(error));
    }
    cudaFree(dev_types);
    cudaFree(dev_indices);
    return dev_buffer;
  }

}
