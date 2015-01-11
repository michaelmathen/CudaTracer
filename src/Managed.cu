#include "Managed.hpp"

namespace mm_ray {
  using namespace std;
  
  
  
  void* Managed::operator new(std::size_t count) throw(std::bad_alloc){
    void* dev_ptr;
    gpuErrchk(cudaMallocManaged((void**)&dev_ptr, count, cudaMemAttachGlobal));
    return dev_ptr;
  }
  
  void* Managed::operator new[](std::size_t count) throw(std::bad_alloc){
    void* dev_ptr;
    gpuErrchk(cudaMallocManaged((void**)&dev_ptr, count, cudaMemAttachGlobal));
    return dev_ptr;
  }

  void Managed::operator delete(void* ptr) throw() {
    gpuErrchk(cudaFree(ptr));
  }
  
  void Managed::operator delete[](void* ptr) throw() {
    gpuErrchk(cudaFree(ptr));
  }

  void* Cuda_Malloc(size_t d_size){
    //A convenience function that
    void* dev_ptr;
     gpuErrchk(cudaMallocManaged(&dev_ptr, d_size, cudaMemAttachGlobal));
    //Call a placement copy constructor
     return dev_ptr;
  }
 
  void Cuda_Free(void* mem){ 
    gpuErrchk(cudaFree(mem));
  }
}
