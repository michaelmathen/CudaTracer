#include <cstdio>
#include <vector>
#include "Ray.hpp"
#include "Hit.hpp"
#include "Accelerator.hpp"
#include "Scene.hpp"
#include "ray_defs.hpp"

#ifndef MM_DISTANCE_RENDER
#define MM_DISTANCE_RENDER
namespace mm_ray {

  struct DistanceRenderer{
    template<typename T>
    Vec3 operator()(Scene const& scene,
		    Accelerator<T> const& objects,
		    Ray const& initial_ray){
      Hit prop;
      objects->Intersect(initial_ray, prop);
      Vec3 distance = 1 / (1 + prop.distance);
      return distance;
    }
  };
}
#endif
