#include <string>
#include <memory>
#include "SceneContainer.hpp"
#include "Scene.hpp"
#include "Renderer.hpp"

#ifndef MM_PARSE_SCENE
#define MM_PARSE_SCENE
template<typename Accelerator>
extern void parse_file(std::string& fname, Scene& scn, Accelerator& container, std::shared_ptr<mm_ray::Renderer<Accelerator> >& renderer);

#endif 






