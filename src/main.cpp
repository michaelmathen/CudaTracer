//extern "C" {
//#include "bmpWrapper.h"
//}
#include <iostream>
#include <fstream>
#include <thread>
#include <algorithm>
#include <vector>
#include <memory>
#include "BVHTreeSimple.hpp"
#include "ray_defs.hpp"
#include "ParseScene.hpp"
#include "SceneContainer.hpp"
#include "ParsingException.hpp"
#include "DistanceRenderer.hpp"
#include "PhongRenderer.hpp"

using namespace std;
using namespace mm_ray;


void writePPM(int dimx, int dimy, const char* file_name, char* image_data){
  FILE *fp = fopen(file_name, "wb");
  if (fp == NULL){
    perror("File failed to open: ");
    exit(EXIT_FAILURE);
  }
  fprintf(fp, "P6\n %s\n %d\n %d\n %d\n", "", dimx, dimy, 255);
  for (int j = 0; j < dimy; ++j){
    for (int i = 0; i < dimx; ++i){
      static unsigned char color[3];
      color[0] = image_data[(j * dimx + i) * 3 + 0];
      color[1] = image_data[(j * dimx + i) * 3 + 1];
      color[2] = image_data[(j * dimx + i) * 3 + 2];
      if (fwrite(color, 1, 3, fp) != 3) {
	perror("Error in writing the file");
	exit(EXIT_FAILURE);
      }
    }
  }
  if (fclose(fp)){
    perror("Error in closing file");
    exit(EXIT_FAILURE);
  }
}


int main(int argc, char* argv[]){

  
  if (argc < 2){
    cout << "Need an input file" << endl;
    return 1;
  }


  Scene host_scene;
  
  typedef BVHTreeSimple Accel;
  
  //Contains all of the managed cuda data so that it can be deleted at the end
  //Uses a vector of unique vectors so they should be freed when the vector goes out of 
  //scope
  //SceneParser<SceneContainer> scn_parser;
  ParseScene scn_parser;
  string fname(argv[1]);
  try {
    //Register the supported materials, geometries, and accelerators

    auto phong_builder = [](rapidjson::Value& material,
			    Scene const& scene_data){
      (void) scene_data;
      
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
    };

    auto sphere_builder = [] (rapidjson::Value& sphere_obj,
			      Scene const& scn,
			      vector<Material*>& materials, 
			      vector<string>& material_names, 
			      vector<Geometry*>& geom_ptrs,
			      vector<Managed*>& geom_data){
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
      auto tmp = new Sphere(center, radius, sphere_mat);
      geom_data.push_back(tmp);
      geom_ptrs.push_back(tmp);
    };

    auto point_builder = [](rapidjson::Value& point_obj,
			    Scene const& scn,
			    vector<Material*>& materials, 
			    vector<string>& material_names, 
			    vector<Geometry*>& geom_ptrs,
			    vector<Managed*>& geom_data){
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
      
      auto tmp = new PointLight(illum, loc);
      geom_data.push_back(tmp);
      geom_ptrs.push_back(tmp);
    };

    auto triangle_builder = [] (rapidjson::Value& mesh_obj,
				Scene const& scn,
				vector<Material*>& materials, 
				vector<string>& material_names, 
				vector<Geometry*>& geom_ptrs,
				vector<Managed*>& geom_data){
  
      Transform tran = Parse_Transform(parse_err.get(mesh_obj, "transform"));
      auto fname = parse_err.get<string>(mesh_obj, "name");
      
      TriangleMesh tr_mesh;
      tr_mesh.parseObj(fname, tran);
      
      auto material_name = parse_err.get<string>(mesh_obj, "material");
      auto it = find(material_names.begin(), material_names.end(), material_name);
      int mat_index = it - material_names.begin();
      
      tr_mesh.setMaterial(materials[mat_index]);
      
      auto tmp = new TriangleMesh(tr_mesh);
      geom_data.push_back(dynamic_cast<TriangleMesh*>(tmp));
      auto triangles = tmp->refine();
      geom_ptrs.insert(geom_ptrs.end(), triangles.begin(), triangles.end());
    };
    
    scn_parser.Register_Material("phong", phong_builder);
    scn_parser.Register_Geometry("sphere", sphere_builder);
    scn_parser.Register_Geometry("point_light", point_builder);
    scn_parser.Register_Geometry("mesh", triangle_builder);
    
    //Now that all supported functionality is registered we can parse the file
    scn_parser.Parse(fname);
    cout << "Finished parsing file" << endl;
    int width;
    int height;
    vector<char> image;
    scn_parser.Run_Renderer(&width, &height, image);
    
    cout << "Writing rendered image" << endl;
    writePPM(width, height, argv[2], &*image.begin());
  } catch (ParsingException* e) {
    cout << "ParsingException" << endl;
    cout << e->what() << endl;
    e->clear();
    return 1;
  } catch (...) {
    cout << "Something really bad happened " << endl;
    return 1;
  }

  return 0;
}
