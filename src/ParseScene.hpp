#include <string>
#include <memory>

#include "SceneContainer.hpp"
#include "Scene.hpp"
#include "Renderer.hpp"

#ifndef MM_PARSE_SCENE
#define MM_PARSE_SCENE
namespace mm_ray{

  void parse_file(std::string& fname, 
		  Scene& scn, 
		  std::shared_ptr<SceneContainer> container, 
		  Renderer<SceneContainer>** renderer,
		  mm_ray::SceneContainerHost&);
}

#endif 






