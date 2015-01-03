
#include <cstdlib>
#include <iostream>
#include <vector>

#ifndef __CUDACC__

#include <boost/type_traits/is_polymorphic.hpp>
#include <boost/utility/enable_if.hpp>

#endif

#define SERIALIZE void 
#include "ray_defs.hpp"

#ifndef MM_SCENE_ALLOCATOR
#define MM_SCENE_ALLOCATOR


/*
This is meant for making moving the scene data to the gpu easier. Instead of
scene_allocating the individual memory pointers we just construct everthing in 
a giant buffer and then cudaMscene_alloc and cudaMemcpy everything over 
to the gpu. We use integers instead of pointers to reference items
inside of the memory buffer. 
 */
namespace mm_ray {

  template<int N>
  struct marker_id {
    static int const value = N;
  };
  
  template<typename T>
  struct marker_type {
    typedef T type;
  };

  using namespace std;


  extern unsigned int scene_buff_size;
  extern unsigned int curr_buff;
  extern char* host_buffer;

  extern vector<Virtual_Type_Val> virtual_types;
  extern vector<unsigned int> virtual_indices;

#ifdef __CUDACC__
  //This is the memory buffer for all the scene object data 
  __device__ extern char* device_buffer;
#endif
  
  //128 bytes
#define SCENE_INITIAL_SIZE 128

  //This is used for serialization
  //Every scene object class gets a enum value.
  //This allows the virtual function pointer to be recreated
  //on the gpu by the initMemory kernel in SceneAllocator
  

  
  template<typename T>
  struct s_ptr {

    /*
      It should be the same size as a integer, but we get 
      some nice type safety.
      Also we got pointer operations.
    */
  public:
    unsigned int index;
    __host__ __device__ s_ptr() {}
    
    __host__ __device__ s_ptr(unsigned int index) : index(index)
    {}

    //Define some operations that can be done on our pointers
    template<typename A, typename B>
    friend s_ptr<A> static_pointer_cast(s_ptr<B> r);

    template<typename A, typename B>
    friend s_ptr<A> dynamic_pointer_cast(s_ptr<B> r);

    template<typename A, typename B>
    friend s_ptr<A> pointer_cast(s_ptr<B> r);

    //Some operations so we can treat them like normal pointers
    __host__ __device__ inline T& operator*(){
#ifdef __CUDA_ARCH__
      return *reinterpret_cast<T*>(device_buffer + index);
#else
      return *reinterpret_cast<T*>(host_buffer + index);
#endif
    }
  
    __host__ __device__ inline T* operator->(){
#ifdef __CUDA_ARCH__
      return reinterpret_cast<T*>(device_buffer + index);
#else
      return reinterpret_cast<T*>(host_buffer + index);
#endif
    }
  
    __host__ __device__ inline T& operator[](unsigned int i){
      //This is a scary line of code...
#ifdef __CUDA_ARCH__
      return *reinterpret_cast<T*>(device_buffer + index + sizeof(T) * i);
#else
      return *reinterpret_cast<T*>(host_buffer + index + sizeof(T) * i);
#endif    
    }
  
  };

  template<typename U, typename T>
  __host__ __device__ inline s_ptr<U> static_pointer_cast(s_ptr<T> r){
    //This is here to check if the static cast can occur.
    //It should throw an error if the static_cast is not possible
    T* ptr_1 = NULL;
    (void)static_cast<U*>(ptr_1);
    return s_ptr<U>(r.index);
  }

  template<typename U, typename T>
  __host__ __device__ inline s_ptr<U> dynamic_pointer_cast(s_ptr<T> r){
    //This is here to check if a dynamic cast can occur.
    //It should throw an error if the dynamic_cast is not possible
    T* ptr_1 = NULL;
    (void)dynamic_cast<U*>(ptr_1);
    return s_ptr<U>(r.index);
  }

  template<typename U, typename T>
  __host__ __device__ inline s_ptr<U> pointer_cast(s_ptr<T> r){
    //This is here to check if the type can be cast
    //It should throw an error if the cast is not possible
    T* ptr_1 = NULL;
    (void)(U*)ptr_1;
    return s_ptr<U>(r.index);
  }

  inline void initialize_scene_alloc(){
    scene_buff_size = SCENE_INITIAL_SIZE;
    curr_buff = 0;
    host_buffer = (char*)malloc(SCENE_INITIAL_SIZE);

  }

  inline void free_scene_scene_alloc(){
    free(host_buffer);
    virtual_types.clear();
    virtual_indices.clear();
  }

  
  void expand_scene_buffer();


#ifndef __CUDACC__
  template <class T>                                                                                                                     
  typename boost::enable_if_c<boost::is_polymorphic<T>::value, Virtual_Type_Val>::type                                                                      
  foo() {                                                                                                                             
    return T::type_id;                                                                 
  }                                                                                                                                      
  
  template <class T>                                                                                                                     
  typename boost::disable_if_c<boost::is_polymorphic<T>::value, Virtual_Type_Val>::type                                                                     
  foo() {                                                                                                                                
    return NOT_VT;                                                                                                                         
  }                                                                                                                                      

  
  template<typename T>
  void addToTypeList(int index){
    foo<T>();
    if (foo<T>() != NOT_VT) {
      virtual_types.push_back(foo<T>());
      virtual_indices.push_back(index);
    }
  }


#ifndef __CUDACC__
  template<typename T>
  unsigned int get_starting_index(int buffer_start){
    /*
      Ensures that the object starts at the right
      byte boundary by padding the begining.
      This is to deal with alignment issues.
    */
    if (buffer_start % alignof(T) == 0)
      return buffer_start;
    return (buffer_start / alignof(T) + 1) * alignof(T);
  }
#endif   
  template<typename T>
  inline s_ptr<T> scene_alloc_internal(){
    /*
      Scene_Allocate the value into our slab
    */

    //Calculates the begining of the buffer index
    unsigned int tmp_start = get_starting_index<T>(curr_buff);
    while (sizeof(T) + tmp_start >= scene_buff_size) {
      expand_scene_buffer();
    }
    
    curr_buff = sizeof(T) + tmp_start;
    addToTypeList<T>(tmp_start);

    return s_ptr<T>(tmp_start);
  }

  template<typename T>
  inline s_ptr<T> scene_alloc(const T& val){
    //Allocate the memory block
    s_ptr<T> start = scene_alloc_internal<T>();
    new(host_buffer + start.index) T(val);
    return start;
  }

  
  template<typename T>
  inline s_ptr<T> scene_alloc(){
    //Allocate the memory block
    s_ptr<T> start = scene_alloc_internal<T>();
    new(host_buffer + start.index) T();
    return start;
  }
  
  template<typename T>
  inline s_ptr<T> scene_alloc(unsigned int n){
    /*
	Scene_Allocate the value into our slab
    */
    if (n == 0){
      //Return a pointer that will segfault if we derefference
      return s_ptr<T>(UINT_MAX);
    }
    
    //Get the correct allignment for the first element
    unsigned int alligned_start = get_starting_index<T>(curr_buff);
    
    while (sizeof(T) * n + alligned_start >= scene_buff_size) {
      expand_scene_buffer();
    }
    for (unsigned int i = 0; i < n; i++){
      new(host_buffer + alligned_start + i * sizeof(T)) T();
    }
    curr_buff = alligned_start + n * sizeof(T);
    return s_ptr<T>(alligned_start);
  }
#endif  
  
#ifdef __CUDACC__
  void* serialize_scene_alloc();
#endif
  




}

#endif
