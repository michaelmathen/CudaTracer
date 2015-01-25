#ifndef __CUDACC__
#include "rapidjson/document.h"
#endif 

#include "Renderer.hpp"


#ifndef __HOST_RENDERER__
#define __HOST_RENDERER__
namespace mm_ray {

  template<typename Accel>
  class HostRenderer : public Renderer<Accel> {
  public:
    HostRenderer(Scene const* scn, Accel const* acc);
    virtual void Render();
  };

  #ifndef __CUDACC__
  template<typename Accel>
  struct HostBuilder : public RendererBuilder<Accel> {
    virtual Renderer<Accel>* operator()(rapidjson::Value&, 
					Scene const* scn,
					Accel const* accel,
					std::vector<Geometry*>&) const{
      return new HostRenderer<Accel>(scn, accel);
    }
  };
#endif 

}
#endif 
