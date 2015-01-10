#include <cstdio>
#include <vector>

#include "SceneContainer.hpp"
#include "Ray.hpp"
#include "Hit.hpp"
#include "ray_defs.hpp"
#include "Renderer.hpp"

namespace mm_ray {

  template<typename Accelerator>
  class DistanceRenderer : public Renderer<Accelerator> {
  protected:
    
  public:

    DistanceRenderer(Scene const& scn, Accelerator const* acc);
    
    virtual ~DistanceRenderer();
    virtual void Render();

  };
}
