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
    //Properly destroys everything
    Cuda_Free(geometry_buffer);
    Cuda_Free(light_sources);
  }
  
  void SceneContainer::Insert_Geometry(vector<Geometry*>& geom){

    //geometry_buffer = geom;
    this->geom_length = geom.size();
    this->geometry_buffer = (Geometry**)Cuda_Malloc(geom.size() * sizeof(Geometry*));
    
    int number_of_lights = 0;
    for (int i = 0; i < geom.size(); i++){
      if (geom[i]->isLight())
	number_of_lights++;
      
      geometry_buffer[i] = geom[i];
    }

    //Now copy all the lights into the light buffer
    light_sources = (Geometry**)Cuda_Malloc(sizeof(Geometry*) * number_of_lights);
    for (int i = 0, light_curr = 0; i < geom_length; i++){
      if (geometry_buffer[i]->isLight()) {
	light_sources[light_curr] = geometry_buffer[i];
	light_curr++;
      }
    }
    this->light_length = number_of_lights;
  }
}


