#include <algorithm>

#include "rapidjson/document.h"
#include "rapidjson/filereadstream.h"
#include "rapidjson/error/error.h"
#include "rapidjson/error/en.h"

#include "ParsingException.hpp"
#include "Transform.hpp"
#include "ray_defs.hpp"
#include "SceneContainer.hpp"

namespace mm_ray {
  using namespace std;
  
  SceneContainer::~SceneContainer(){
    Cuda_Free(geometry_buffer);
    Cuda_Free(light_sources);
  }

  SceneContainer* SceneContainer::Build_Accelerator(std::vector<Geometry*>& geom){
    auto scn_ptr = new SceneContainer();
    scn_ptr->geom_length = geom.size();
    scn_ptr->geometry_buffer = (Geometry**)Cuda_Malloc(geom.size() * sizeof(Geometry*));
    
    int number_of_lights = 0;
    for (unsigned i = 0; i < geom.size(); i++){
      if (geom[i]->isLight())
	number_of_lights++;
      
      scn_ptr->geometry_buffer[i] = geom[i];
    }

    //Now copy all the lights into the light buffer
    scn_ptr->light_sources = (Geometry**)Cuda_Malloc(sizeof(Geometry*) * number_of_lights);
    for (int i = 0, light_curr = 0; i < scn_ptr->geom_length; i++){
      if (scn_ptr->geometry_buffer[i]->isLight()) {
	scn_ptr->light_sources[light_curr] = scn_ptr->geometry_buffer[i];
	light_curr++;
      }
    }
    scn_ptr->light_length = number_of_lights;
    
  }
}


