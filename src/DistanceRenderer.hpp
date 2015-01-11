#include <cstdio>
#include <vector>

#include "rapidjson/document.h"
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
  
  template<typename Accel>
  struct DistanceBuilder : public RendererBuilder<Accel> {
    virtual Renderer<Accel>* operator()(rapidjson::Value&, 
					Scene const*,
					Accel const*,
					std::vector<Geometry*>&) const;
  };
  
}
