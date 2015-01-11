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
  template <typename Accel>
  class Renderer {
    
  protected:
    Scene host_scene;
    Accel host_accel;
    std::vector<Real_t> output_buffer;
    
  public:
    
    Renderer(Scene const& scn, Accel const& accel);
    ~Renderer();

    virtual std::vector<Real_t> getImage();
    virtual void Render() = 0;
    
  };
  
  class Geometry;
  
  template<typename Accel>
  struct RendererBuilder {
    virtual Renderer<Accel>* operator()(rapidjson::Value&, 
					Scene const&,
					Accel const&,
					std::vector<Geometry*>&) const =0;
  };
}
#endif
