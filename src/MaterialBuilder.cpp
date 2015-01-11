#include <string>
#include <sstream>

#include "rapidjson/document.h"
#include "rapidjson/filereadstream.h"
#include "rapidjson/error/error.h"
#include "rapidjson/error/en.h"

#include "ParsingException.hpp"
#include "Material.hpp"

using namespace std;
namespace mm_ray {
  Material* PhongMaterialBuilder::operator()(rapidjson::Value& material, 
				     Scene const& scene_data){
    //(void) scene_data;
    
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
}
