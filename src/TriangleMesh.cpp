#include <fstream>
#include <iostream>
#include <vector>
#include <sstream>
#include <cstring>

#include <boost/algorithm/string/trim.hpp>
#include <boost/algorithm/string/predicate.hpp>
#include <boost/algorithm/string/split.hpp>
#include <boost/algorithm/string.hpp>
#include "ParsingException.hpp"

#include "GeometryData.hpp"
#include "Geometry.hpp"
#include "Transform.hpp"
#include "ray_defs.hpp"


namespace mm_ray {
  using namespace std;

  void TriangleMesh::setMaterial(Material* mat){
    this->material = mat;
  }

  
  vector<Geometry*> TriangleMesh::refine(){
    /*
      Break the triangle mesh into sub triangles.
    */
    vector<Geometry*> triangle_arr;
    triangle_arr.resize(this->number_of_triangles);
    cout << this->number_of_triangles << "number of triangles" << endl;

    for (unsigned int i = 0; i < this->number_of_triangles; i++){
      triangle_arr[i] = new Triangle(this, i);
    }
    return triangle_arr;
  }

  void TriangleMesh::parseObj(std::string const& fname, Transform const& transformation){

    vector<Vec3> vertices;
    vector<Tri_vert> triangles;

    ifstream obj_file(fname);
    string line_buff;
    while (getline(obj_file, line_buff)){
      //Remove comments
      auto comment_pos = line_buff.find_first_of("#");
      if (comment_pos != string::npos){
	//Grab everything before the comment
	line_buff = line_buff.substr(0, comment_pos - 1);
      }

      //Remove trailing and leading white spaces
      boost::trim(line_buff);
      
      //Ignore empty lines
      if (line_buff.length() == 0)
	continue;

      //Currently we aren't using the surface normals or the
      //texture coordinates
      if (boost::starts_with(line_buff, "vn"))
	  continue;

      if (boost::starts_with(line_buff, "vt"))
	continue;

      //The vertex coordinates
      if (boost::starts_with(line_buff, "v")){
	vector<string> els;
	boost::split(els, line_buff, boost::is_any_of(" \t"), boost::token_compress_on);
	
	if (els.size() != 4) {
	  parse_err << "Line is the wrong length:\n";
	  for (auto el : els)
	    parse_err << "\"" << el << "\" ";
	  parse_err << "\n";
	  parse_err << line_buff;
	  throw &parse_err;
	}
	Vec3 vertex;
	vertex[0] = stof(els[1]);
	vertex[1] = stof(els[2]);
	vertex[2] = stof(els[3]);
	//Now apply the transformation matrix
	vertices.push_back(transformation.applyTransform(vertex));
	continue;
      }

      if (boost::starts_with(line_buff, "f")){
	vector<string> els;
	boost::split(els, line_buff, boost::is_any_of(" \t"), boost::token_compress_on);

	//process each edge
	vector<int> polygon_vertices;
	for (unsigned int i = 1; i < els.size(); i++){
	  vector<string> indices;
	  boost::split(indices, els[i], boost::is_any_of("/"));
	  //Just want the first one now
	  int vertex_ind = atoi(indices[0].c_str());
	  if (vertex_ind < 0) {
	    //Relative to current polygon location
	    polygon_vertices.push_back(vertices.size() - vertex_ind);
	  } else {
	    //Relative to the beginning of the file
	    polygon_vertices.push_back(vertex_ind - 1);
	  }
	}
	//Break up the polygon into triangles
	for (unsigned int i = 1; i < polygon_vertices.size() - 1; i++) {
	  Tri_vert triangle;
	  triangle.x = polygon_vertices[0];
	  triangle.y = polygon_vertices[i];
	  triangle.z = polygon_vertices[i + 1];
	  triangles.push_back(triangle);
	}
	continue;
      }

      //Ignore materials right now
      if (boost::starts_with(line_buff, "usemtl"))
	continue;
      if (boost::starts_with(line_buff, "mtllib"))
	continue;
    }
    
    //Now copy this over to our object mesh
    
    triangle_vertices = (Vec3*)Cuda_Malloc(vertices.size() * sizeof(Vec3));
    vertex_indices = new Tri_vert[triangles.size()];
    
    for (unsigned int i = 0; i < vertices.size(); i++){
      triangle_vertices[i] = vertices[i];
    }
    
    for (unsigned int i = 0; i < triangles.size(); i++){
      vertex_indices[i] = triangles[i];
    }
    this->number_of_triangles = triangles.size();
  }
}
