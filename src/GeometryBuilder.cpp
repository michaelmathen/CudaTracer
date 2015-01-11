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
#include "ParseScene.hpp"
#include "ParsingException.hpp"
#include "Transform.hpp"
#include "GeometryData.hpp"

using namespace std;
namespace mm_ray {

  void SphereBuilder::operator()(rapidjson::Value& sphere_obj,
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
  }

  void PointBuilder::operator()(rapidjson::Value& point_obj,
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
  }


  void TriangleMeshBuilder::operator()(rapidjson::Value& mesh_obj,
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
  }

}
