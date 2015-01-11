#include <cstdio>
#include <vector>

#include "SceneContainer.hpp"
#include "Ray.hpp"
#include "Hit.hpp"
#include "ray_defs.hpp"
#include "Renderer.hpp"

#ifndef  MM_PHONG_RENDERER
#define MM_PHONG_RENDERER

namespace mm_ray {

  template<typename Accel>
  class PhongRenderer : public Renderer<Accel> {
  protected:
  public:

    PhongRenderer(Scene const& scn, Accel const& acc);
    
    virtual void Render();

  };

  template<typename Accel>
  struct PhongBuilder : public RendererBuilder<Accel> {
    virtual Renderer<Accel>* operator()(rapidjson::Value&, 
					Scene const&,
					Accel const&,
					std::vector<Geometry*>&) const;
  };

}
#endif 
