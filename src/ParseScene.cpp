#include <cstdlib>
#include <fstream>
#include <vector>
#include <iostream>
#include <memory>

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
void parse_materials(rapidjson::Value& root,
		     vector<Material*>& materials,
		     vector<string>& material_names,
		     SceneContainerHost&);

void parse_scene(rapidjson::Value& root, Scene& scn){
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
  Scene lcl(viewport, output[0], output[1], cam_loc, cam_dir, cam_up, samples);
  auto& render_block = parse_err.get(root, "render_block");
  
  lcl.render_block_x = parse_err.get<int>(render_block, 0);
  lcl.render_block_y = parse_err.get<int>(render_block, 1);
  lcl.print();
  scn = lcl;
}


void parse_geometry(rapidjson::Value&,
		    vector<Material*>&,
		    vector<string>&,
		    vector<Geometry*>&, 
		    SceneContainerHost&);


template<typename Accelerator>
void parse_render_tag(rapidjson::Value& root, 
		      Scene const& scn, 
		      shared_ptr<Accelerator> acc,
		      Renderer<Accelerator>** renderer){
  /*
    This parses the renderer tag allowing us to choose 
    the renderer of the current scene
   */
  auto& renderer_tag = parse_err.get(root, "renderer");
  auto render_name = parse_err.get<string>(renderer_tag, "type");

  if (render_name == "phong"){
    *renderer = new PhongRenderer<Accelerator>(scn, acc.get());
  } else if (render_name == "distance") {
    *renderer = new DistanceRenderer<Accelerator>(scn, acc.get());
  } else {
    parse_err << "Unknown renderer type \"" << render_name << "\"\n";
    throw &parse_err;
  }
}

void parse_file(string& fname, 
		Scene& scn,
		shared_ptr<SceneContainer> container,
		Renderer<SceneContainer>** renderer,
		SceneContainerHost& managed_data){
  
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
  vector<string> mat_names;
  try {
    vector<Material*> mat;
    vector<Geometry*> geometry_objs;
    parse_materials(root, mat, mat_names, managed_data);
    parse_scene(root, scn);

    parse_geometry(root, mat, mat_names, geometry_objs, managed_data);
    //cout << geometry_objs[0]->isLight() << endl;
    container->Insert_Geometry(geometry_objs);

    parse_render_tag<SceneContainer>(root, scn, container, renderer);
  } catch (ParsingException* e){
    *e << "\n";
    *e << "Parsing error in file: " << fname << "\n";
    throw e;
  }
  fclose(fb);
}

Transform parse_transform(rapidjson::Value& transform_obj){
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

TriangleMesh* parse_mesh(rapidjson::Value& mesh_obj,
			 vector<Material*> & materials,
			 vector<string>& material_names){
  
  Transform tran = parse_transform(parse_err.get(mesh_obj, "transform"));
  auto fname = parse_err.get<string>(mesh_obj, "name");

  TriangleMesh tr_mesh;
  tr_mesh.parseObj(fname, tran);

  auto material_name = parse_err.get<string>(mesh_obj, "material");
  auto it = find(material_names.begin(), material_names.end(), material_name);
  int mat_index = it - material_names.begin();

  tr_mesh.setMaterial(materials[mat_index]);

  return new TriangleMesh(tr_mesh);
}


Geometry* parse_sphere(rapidjson::Value& sphere_obj,
			     vector<Material*>& materials,
			     vector<string>& material_names){


  

  auto mat_name = parse_err.get<string>(sphere_obj, "material");
  auto it = find(material_names.begin(), material_names.end(), mat_name);

  int mat_index = it - material_names.begin();
  cout << mat_index << endl;
  Material* sphere_mat = materials[mat_index];

  Vec3 center;
  auto& center_vals = parse_err.get(sphere_obj, "center");
  for (auto i = 0; i < center_vals.Size(); i++)
    center[i] = parse_err.get<double>(center_vals, i);
  
  auto radius = parse_err.get<double>(sphere_obj, "radius");
  return new Sphere(center, radius, sphere_mat);
}

Geometry*  parse_point_light(rapidjson::Value& point_obj){
  Vec3 loc;
  Vec3 illum;
  auto& center = parse_err.get(point_obj, "center");
  loc[0] = parse_err.get<Real_t>(center, 0);
  loc[1] = parse_err.get<Real_t>(center, 1);
  loc[2] = parse_err.get<Real_t>(center, 2);

  auto& illumination = parse_err.get(point_obj, "illumination");
  illum[0] = parse_err.get<Real_t>(illumination, 0);
  illum[1] = parse_err.get<Real_t>(illumination, 1);
  illum[2] = parse_err.get<Real_t>(illumination, 2);

  return new PointLight(illum, loc);
}

Material* parse_phong_material(rapidjson::Value& material) {
  
  Real_t spec_light = parse_err.get<Real_t>(material, "specular");
  Real_t diff_light = parse_err.get<Real_t>(material, "diffuse");
  Real_t amb_light = parse_err.get<Real_t>(material, "ambient");
  Real_t shine = parse_err.get<Real_t>(material, "shine");

  unsigned  int color_val;
  stringstream ss;
  ss << hex << parse_err.get<string>(material, "color");
  ss >> color_val;

  Vec3 color;
  color[0] = ((color_val >> 16) & 255) / 255.0;  
  color[1] = ((color_val >> 8) & 255) / 255.0;
  color[2] = (color_val & 255) / 255.0;

  return new PhongMaterial(color, spec_light, diff_light, amb_light, shine);
}



void parse_materials(rapidjson::Value& root,
		     vector<Material*>& materials,
		     vector<string>& material_names,
		     SceneContainerHost& managed_data){

  /*
    Materials are kept track of using their names.
   */
  if (!root.HasMember("materials")){
    cout << "No materials tag using default material for everything" << endl;
    return ;
  }
  auto& material = parse_err.get(root, "materials");
  parse_err.checkType(material, ValueType::VAL_ARRAY);
  for (int i = 0; i < material.Size(); i++){
    auto& curr_matt = parse_err.get(material, i);

    material_names.push_back(parse_err.get<string>(curr_matt, "name"));

    if (parse_err.get<string>(curr_matt, "type") == "phong"){
      auto tmp = parse_phong_material(curr_matt);
      managed_data.push_back(auto_ptr<Managed>(tmp));
      materials.push_back(tmp);
    }
  }
}

void parse_geometry(rapidjson::Value& root,
		    vector<Material*>& materials,
		    vector<string>& material_names,
		    vector<Geometry*>& geometry, 
		    SceneContainerHost& scene_data){

  auto& objects = parse_err.get(root, "geometry");


  for (int i = 0; i < objects.Size(); i++){
    if (parse_err.get<string>(objects[i], "type") == "sphere"){
       auto tmp = parse_sphere(objects[i], materials, material_names);
      geometry.push_back(tmp);
      scene_data.push_back(auto_ptr<Managed>(tmp));
    } else if (parse_err.get<string>(objects[i], "type") == "point_light"){
      auto tmp = parse_point_light(objects[i]);
      geometry.push_back(tmp);
      scene_data.push_back(auto_ptr<Managed>(tmp));
    } else if (parse_err.get<string>(objects[i], "type") == "mesh"){
      //Parse the mesh object
      TriangleMesh* mesh = parse_mesh(objects[i], materials, material_names);
      scene_data.push_back(auto_ptr<Managed>(mesh));
      //Now we break it appart
      auto triangle_ptrs = refine(mesh);
      geometry.insert(geometry.end(), triangle_ptrs.begin(), triangle_ptrs.end());
    }
  }
}
}
