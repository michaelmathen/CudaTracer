#include <cstdlib>
#include <fstream>
#include <vector>
#include <iostream>
#include <memory>
//#include <boost/shared_ptr.hpp>

#include "rapidjson/document.h"
#include "rapidjson/filereadstream.h"
#include "rapidjson/error/error.h"
#include "rapidjson/error/en.h"
#include "ray_defs.hpp"
#include "Renderer.hpp"
#include "PhongRenderer.hpp"
#include "DistanceRenderer.hpp"
#include "SceneContainer.hpp"
#include "ParsingException.hpp"
#include "BVHTreeSimple.hpp"
#include "Transform.hpp"
#include "GeometryData.hpp"
#include "ParseScene.hpp"


using namespace std;
namespace mm_ray {

  SceneParser::~SceneParser(){
    delete scene_data;
    for (auto el : geometry_data)
      delete el;
    geometry_data.clear();
    for (auto el : material_ptrs)
      delete el;
    material_ptrs.clear();
  }

  void SceneParser::Register_Material(std::string const& type, 
					     shared_ptr<MaterialBuilder> build){
    material_builders[type] = build;
  }

  void SceneParser::Register_Geometry(std::string const& type, 
					     shared_ptr<GeometryBuilder> build){
    geometry_builders[type] = build;
  }


  ParseScene(std::string& fname){
    FILE *fb = fopen(fname.c_str(), "r");
    
    if (fb == NULL){
      parse_err << "File " << fname  << " does not seem to exist\n";
      throw &parse_err;
    }
    char readBuffer[65536];
    rapidjson::FileReadStream is(fb, readBuffer, sizeof(readBuffer));
    root_doc.ParseStream(is);
    
    if (root_doc.HasParseError()){
      auto ok = root_doc.GetParseError();
      parse_err << "JSON parse error: "  << rapidjson::GetParseError_En(ok);
      parse_err << "Error offset is: " << root_doc.GetErrorOffset();
      throw &parse_err;
    }
    auto& root = dynamic_cast<rapidjson::Value&>(root_doc);
    try {
      Parse_Scene(root);
      Parse_Material(root);
      Parse_Geometry(root);
    } catch (ParsingException* e){
      *e << "\n";
      *e << "Parsing error in file: " << fname << "\n";
      throw e;
    }
    fclose(fb);

  }


  void SceneParser::Parse_Scene(rapidjson::Value& root){
    Vec2 viewport;
    
    auto& viewport_size = parse_err.AssertGetMember(root, "viewport_size");
    viewport[0] = parse_err.get<double>(viewport_size, 0);
    viewport[1] = parse_err.get<double>(viewport_size, 1);
    
    int output[2];
    auto& output_json = parse_err.get(root, "output");
    output[0] = parse_err.get<int>(output_json, 0);
    output[1] = parse_err.get<int>(output_json, 1);
    
    Vec3 cam_loc;
    Vec3 cam_dir;
    Vec3 cam_up;
    auto& camera = parse_err.get(root, "camera");
    for (int i = 0; i < 3; i++){
      cam_loc[i] = parse_err.get<double>(parse_err.get(camera, "cam_loc"), i);
      cam_dir[i] = parse_err.get<double>(parse_err.get(camera, "cam_dir"), i);
      cam_up[i] =  parse_err.get<double>(parse_err.get(camera, "cam_up"), i);
    }
    int samples = parse_err.get<int>(root, "samples");

    auto& render_block = parse_err.get(root, "render_block");
    scene_data = new Scene(viewport, output[0], output[1], cam_loc, cam_dir, cam_up, samples);
    scene_data->render_block_x = parse_err.get<int>(render_block, 0);
    scene_data->render_block_y = parse_err.get<int>(render_block, 1);
  }


  Transform Parse_Transform(rapidjson::Value& transform_obj){
    Real_t scale = parse_err.get<Real_t>(transform_obj, "scale");
    Real_t rotation = parse_err.get<Real_t>(parse_err.get(transform_obj, "rotate"), "angle");
    auto& axis_array = parse_err.get(parse_err.get(transform_obj, "rotate"), "axis");
    Vec3 axis;
    axis[0] = parse_err.get<Real_t>(axis_array, 0);
    axis[1] = parse_err.get<Real_t>(axis_array, 1);
    axis[2] = parse_err.get<Real_t>(axis_array, 2);
    
    auto& translate_arr = parse_err.get(transform_obj, "translate");
    Vec3 translate;
    translate[0] = parse_err.get<Real_t>(translate_arr, 0);
    translate[1] = parse_err.get<Real_t>(translate_arr, 1);
    translate[2] = parse_err.get<Real_t>(translate_arr, 2);
    
    return Transform(scale, rotation, axis, translate);
  }

  /*
    The below two functions are here because I didn't want to write out 
    all of the nested if statements so instead I let c++ templates 
    come to the rescue.
   */
  template<typename RenderF, typename DeviceF>
  void Parse_Launch_Accelerator(rapidjson::Value& root, Vec3* results){
    rapidjson::Value& accelerator_tag = parse_err.get(root, "accelerator");
    string accelerator_name = parse_err.get<string>(accelerator_tag, "type");
    
    if (accelerator_name == "Simple"){
      
    } else if (accelerator_name == "Distance"){
      
    } else {
      parse_err << "Unrecognized render function name " << render_name << "\n";
    }
  }

  template<typename DeviceF>
  void Parse_Launch_Renderer(rapidjson::Value& root, Vec3* results){
    rapidjson::Value& renderer_tag = parse_err.get(root, "renderer");
    string render_name = parse_err.get<string>(renderer_tag, "type");
    if (render_name == "Phong"){
      Parse_Launch_Accelerator<PhongFunc, DeviceF>(root, results);
    } else if (render_name == "Distance"){
      Parse_Launch_Accelerator<DistFunc, DeviceF>(root, results);
    } else {
      parse_err << "Unrecognized render function name " << render_name << "\n";
    }
  }

  vector<Vec3> SceneParser::Run_Renderer(){
    /*
      This function figures out which renderer to run based
      on the information in the scene file.
      We can have multiple kinds of render functions such as device,
      serial host, threaded host,... We can also have multiple kinds
      of accelerators such as BVH trees, or grids. We can also have 
      multiple different kinds of ray tracing algorithms defined 
      on a per pixel basis such as path tracing, distance, whitting,...
     */
    rapidjson::Value& device_tag = parse_err.get(root, "device");
    string device_name = parse_err.get<string>(device_tag, "type");
    vector<Vec3> output_buffer;
    output_buffer.resize(scene_data->output[0] * scene_data->output[1]);

    if (device_name == "CUDA"){
      Parse_Launch_Renderer<DeviceRenderer>(root, &*output_buffer.begin());
    } else if (device_name == "HostSerial"){
      
    } else {
      parse_err << "Unrecognized render tag: render_name \n";
      throw parse_err;
    }
    Build_Accelerator(geometry_ptrs);
    renderer = it->second->operator()(renderer_tag, scene_data,
				      accelerator, geometry_ptrs);
    
  }

  void SceneParser::Parse_Material(rapidjson::Value& root){
    auto& material = parse_err.get(root, "materials");
    parse_err.checkType(material, ValueType::VAL_ARRAY);
    for (unsigned i = 0; i < material.Size(); i++){
      auto& curr_matt = parse_err.get(material, i);
      auto type_string = parse_err.get<string>(curr_matt, "type");
      auto it = material_builders.find(type_string);
      if (it == material_builders.end()){
	parse_err << "Unrecognized material type " << type_string << "\n";
	throw &parse_err;
      }
      material_names.push_back(parse_err.get<string>(curr_matt, "name"));
      material_ptrs.push_back(it->second->operator()(curr_matt, *scene_data));
    }
  }

  void SceneParser::Parse_Geometry(rapidjson::Value& root){
    auto& objects = parse_err.get(root, "geometry");
    for (int i = 0; i < objects.Size(); i++){
      auto& geom_obj = parse_err.get(objects, i);
      auto type_string = parse_err.get<string>(geom_obj, "type");
      auto it = geometry_builders.find(type_string);
      if (it == geometry_builders.end()){
	parse_err << "Unrecognized geometry type " << type_string << "\n";
	throw &parse_err;
      }
      it->second->operator()(geom_obj,
			     *scene_data,
			     material_ptrs, 
			     material_names, 
			     geometry_ptrs, 
			     geometry_data);
    }
  }

  template class SceneParser<SceneContainer>;
  template class SceneParser<BVHTreeSimple>;
}
