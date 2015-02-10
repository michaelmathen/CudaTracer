#include <string>
#include <map>
#include <memory>
#include "Scene.hpp"
#include "GeometryData.hpp"
#include "Material.hpp"

#include "rapidjson/document.h"

#ifndef MM_PARSE_SCENE
#define MM_PARSE_SCENE

namespace mm_ray{

  typedef Material*(Material_t)(rapidjson::Value&, Scene const&);

  typedef void(Geometry_t)(rapidjson::Value&, Scene const&,
			   std::vector<Material*>&, std::vector<std::string>&, 
			   std::vector<Geometry*>&, std::vector<Managed*>&);
  
  class ParseScene {

    //This allocates of the data allocated for geometry,
    //materials, and other managed objects

    std::map<std::string, Geometry_t> geometry_builders;
    std::map<std::string, Material_t> material_builders;

    Scene* scene_data;
    std::string accel_name;

    //This is the actual data for the geometry objects
    //We have this because of meshes and such that require
    //having tons of lightweight objects that are basically
    //just pointers
    std::vector<Managed*> geometry_data;

    rapidjson::Value root;
    
    //The Geometry that has been created with new should also have 
    //a pointer in geometry_data
    std::vector<Geometry*> geometry_ptrs;
    std::vector<Material*> material_ptrs;

    std::vector<std::string> material_names;

    void Parse_Scene(rapidjson::Value&);
    void Parse_Material(rapidjson::Value&);
    void Parse_Geometry(rapidjson::Value&);

    template<typename T, typename RenderF, typename DeviceF>
    void Launch_Kernel(Vec3* results);

    template<typename RenderF, typename DeviceF>
    void Parse_Launch_Accelerator(Vec3* results);

    template<typename DeviceF>
    void Parse_Launch_Renderer(Vec3* results);

  public:
    /*
      Reads all of the data from file and allocates memory for 
      all of them
     */
    ParseScene(){
    }

    //Register different creation functions that handle parsing this 
    //objects data

    //Materials will be parsed after scene since they are used for geometry data
    void Register_Material(std::string const&, Material_t);

    //Then we parse geometry data
     void Register_Geometry(std::string const&, Geometry_t);
    
    void Parse(std::string& fname);
    
    void Run_Renderer(int* width, int* height, std::vector<char>& output);

    Scene const& Get_Scene(){
      return *scene_data;
    }
    
    ~ParseScene();

  };
  
  Transform Parse_Transform(rapidjson::Value&);
}

#endif 






