#include "SceneContainer.hpp"

namespace mm_ray {
  void SceneContainer::insertGeometry(s_ptr<s_ptr<Geometry> > geom, int geom_length){

    geometry_buffer = geom;
    cout << geom_length << endl;
    this->geom_length = geom_length;
    
    int number_of_lights = 0;
    for (int i = 0; i < geom_length; i++){
      /*
      cerr << "Geom index " << geometry_buffer.index << endl;
      cerr << "sptr index " << geometry_buffer[i].index << endl;
      
      s_ptr<Geometry> gem = geometry_buffer[i];
      Sphere* g = (Sphere*)(host_buffer + gem.index);
      cerr << g->isLight() << endl;
      */
      if (geometry_buffer[i]->isLight())
	number_of_lights++;
      
      geometry_buffer[i] = geom[i];
    }

    //Now copy all the lights into the light buffer
    light_sources = scene_alloc<s_ptr<Geometry> >(number_of_lights);
    for (int i = 0, light_curr = 0; i < geom_length; i++){
      if (geometry_buffer[i]->isLight()) {
	light_sources[light_curr] = geometry_buffer[i];
	light_curr++;
      }
    }
    
    this->light_length = number_of_lights;
    this->geom_length = geom_length;
  }
}
