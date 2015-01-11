#include <cstdlib>
#include <fstream>
#include <vector>
#include <iostream>
#include <memory>
#include <boost/shared_ptr.hpp>

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
#include "Transform.hpp"
#include "GeometryData.hpp"
#include "ParseScene.hpp"


using namespace std;
namespace mm_ray {

  template<typename Accel>
  void SceneParser<Accel>::Register_Material(std::string const& type, 
      boost::shared_ptr<MaterialBuilder> build){
          material_builders[type] = build;
  }

  template<typename Accel>
  void SceneParser<Accel>::Register_Geometry(std::string const& type, 
				      boost::shared_ptr<GeometryBuilder> build){
    geometry_builders[type] = build;
  }

  template<typename Accel>
  void SceneParser<Accel>::Register_Renderer(std::string const& type, 
				      boost::shared_ptr<RendererBuilder<Accel> > build){
    render_builders[type] = build;
  }


  template<typename Accel>
  void SceneParser<Accel>::Parse_Scene(rapidjson::Value& root){
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
    scene_data = Scene(viewport, output[0], output[1], cam_loc, cam_dir, cam_up, samples);
    scene_data.render_block_x = parse_err.get<int>(render_block, 0);
    scene_data.render_block_y = parse_err.get<int>(render_block, 1);
  }

  template<typename Accel>
  void SceneParser<Accel>::Parse(string& fname){
    FILE *fb = fopen(fname.c_str(), "r");
    
    if (fb == NULL){
      parse_err << "File " << fname  << " does not seem to exist\n";
      throw &parse_err;
    }
    char readBuffer[65536];
    rapidjson::FileReadStream is(fb, readBuffer, sizeof(readBuffer));
    rapidjson::Document root_doc;
    
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
      Parse_Geometry(root);
      Parse_Renderer(root);
    } catch (ParsingException* e){
      *e << "\n";
      *e << "Parsing error in file: " << fname << "\n";
      throw e;
    }
    fclose(fb);
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

  template<typename Accel>
  void SceneParser<Accel>::Parse_Renderer(rapidjson::Value& root){
    //There can only be one scene declared at a time
    auto& renderer_tag = parse_err.get(root, "renderer");
    auto render_name = parse_err.get<string>(renderer_tag, "type");
    auto it = render_builders.find(render_name);
    if (it == render_builders.end()){
      parse_err << "Unrecognized renderer type " << render_name << "\n";
      throw &parse_err;
    }
    accelerator.Insert_Geometry(geometry_ptrs);
    renderer = it->second->operator()(renderer_tag, scene_data,
				      accelerator, geometry_ptrs);
  }

  template<typename Accel>
  void SceneParser<Accel>::Parse_Material(rapidjson::Value& root){
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
      material_ptrs.push_back(it->second->operator()(curr_matt, scene_data));
    }
  }

  template<typename Accel>
  void SceneParser<Accel>::Parse_Geometry(rapidjson::Value& root){
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
			     scene_data, 
			     material_ptrs, 
			     material_names, 
			     geometry_ptrs, 
			     geometry_data);
    }
  }

  template class SceneParser<SceneContainer>;
}
