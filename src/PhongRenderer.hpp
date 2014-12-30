#include <cstdio>
#include <vector>

#include "SceneContainer.hpp"
#include "SceneObjects.hpp"
#include "Ray.hpp"
#include "Hit.hpp"
#include "ray_defs.hpp"
#include "Renderer.hpp"

#ifndef  MM_PHONG_RENDERER
#define MM_PHONG_RENDERER

namespace mm_ray {

  template<typename Accelerator>
  class PhongRenderer : public Renderer<Accelerator> {
  protected:
    
  public:

    PhongRenderer(Scene const& scn, Accelerator const& acc);
    
    virtual ~PhongRenderer();
    virtual void Render();

  };
}
#endif 
