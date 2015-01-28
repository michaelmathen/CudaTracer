#include <cstdio>
#include <vector>

#ifndef __CUDACC__
#include "rapidjson/document.h"
#endif 

#include "SceneContainer.hpp"
#include "Ray.hpp"
#include "Hit.hpp"
#include "ray_defs.hpp"
#include "Renderer.hpp"

namespace mm_ray {

  template<typename Accel>
  class DistanceRenderer : public Renderer<Accel> {
  public:

    DistanceRenderer(Scene const* scn, Accel const* acc) :
      Renderer<Accel>(scn, acc) 
    {}
    
    virtual void Render();
  };

#ifndef __CUDACC__  
  template<typename Accel>
  struct DistanceBuilder : public RendererBuilder<Accel> {
    virtual Renderer<Accel>* operator()(rapidjson::Value& val_obj, 
					Scene const* scn,
					Accel const* accelerator,
					std::vector<Geometry*>& geometry) const {
      return new DistanceRenderer<Accel>(scn, accelerator);
    }
  };
#endif
  
}
