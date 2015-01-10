#include <vector>
#include "ray_defs.hpp"
#include "Scene.hpp"
#include "SceneContainer.hpp"

/*
  The renderer base class. All of the different renderers inherit from this one. 
  The rendering class renders a chunk of an image using either the device or host 
  code.
 */

#ifndef MM_RENDERER
#define MM_RENDERER
namespace mm_ray {
  template<typename Accelerator>
  class Renderer {
    
  protected:
    Scene host_scene;
    Accelerator const* host_accel;
    
    std::vector<Real_t> output_buffer;
    
  public:
    
    Renderer(Scene const& scn, Accelerator const* acc);
    ~Renderer();
    virtual std::vector<Real_t> getImage();
    
    virtual void Render() = 0;
    
  };
}
#endif
