#include "Renderer.hpp"


namespace mm_ray {

  template<typename Accelerator>
  Renderer<Accelerator>::Renderer(Scene const& scn, Accelerator const& acc) :
    host_scene(scn),
    host_accel(acc) {			
    output_buffer.resize(host_scene.output[0] * host_scene.output[1] * 3);
  }
  
  
  template<typename Accelerator>
  Renderer<Accelerator>::~Renderer(){
  }
  
  template<typename Accelerator>
  vector<Real_t> Renderer<Accelerator>::getImage(){
    return output_buffer;
  }

  template class Renderer<SceneContainer>;
}

