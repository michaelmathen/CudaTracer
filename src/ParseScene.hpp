#include <string>
#include <map>
#include <memory>
//#include <boost/shared_ptr.hpp>

#include "Scene.hpp"
#include "SceneContainer.hpp"
#include "Renderer.hpp"
//#include "GeometryBuil

#ifndef MM_PARSE_SCENE
#define MM_PARSE_SCENE

namespace mm_ray{
  class ParseScene {

    //This allocates of the data allocated for geometry,
    //materials, and other managed objects

    std::map<std::string, std::shared_ptr<GeometryBuilder> > geometry_builders;
    std::map<std::string, std::shared_ptr<MaterialBuilder> > material_builders;

    Scene* scene_data;
    std::string accel_name;

    //This is the actual data for the geometry objects
    //We have this because of meshes and such that require
    //having tons of lightweight objects that are basically
    //just pointers
    std::vector<Managed*> geometry_data;

    //The Geometry that has been created with new should also have 
    //a pointer in geometry_data
    std::vector<Geometry*> geometry_ptrs;
    std::vector<Material*> material_ptrs;

    std::vector<std::string> material_names;

    void Parse_Scene(rapidjson::Value&);
    void Parse_Material(rapidjson::Value&);
    void Parse_Geometry(rapidjson::Value&);

    template<typename Accel, typename RenderF, typename DeviceF>
    void Launch_Kernel(Vec3* results){

    template<typename DeviceF>
    void Parse_Launch_Renderer(Vec3* results);

    template<typename RenderF, typename DeviceF>
    void Parse_Launch_Accelerator(Vec3* results);

    rapidjson::Value& root;
    
  public:
    /*
      Reads all of the data from file and allocates memory for 
      all of them
     */
    ParseScene(std::string& fname);
    
    //Register different creation functions that handle parsing this 
    //objects data

    //Materials will be parsed after scene since they are used for geometry data
    void Register_Material(std::string const&, std::shared_ptr<MaterialBuilder>);

    //Then we parse geometry data
     void Register_Geometry(std::string const&, std::shared_ptr<GeometryBuilder>);
    
    void Parse(std::string& fname);
    
    void Run_Renderer();

    Scene const& Get_Scene(){
      return *scene_data;
    }
    
    
    Renderer<Accel>& Get_Renderer(){
      return *renderer;
    }
    
    ~SceneParser();

  };
  
  Transform Parse_Transform(rapidjson::Value&);
}

#endif 






