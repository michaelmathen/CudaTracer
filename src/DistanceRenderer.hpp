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

#ifndef MM_DISTANCE_RENDER
#define MM_DISTANCE_RENDER
namespace mm_ray {

  struct DistanceRenderer{
    template<typename Accel>
    Vec3 operator()(Scene const& scene,
		    Accel const& objects,
		    Ray const& initial_ray){
      Hit prop;
      objects->Intersect(initial_ray, prop);
      Vec3 distance = 1 / (1 + prop.distance);
      return distance;
    }
  };
}
#endif
