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
#include "HostRenderer.hpp"
#include "PhongRenderer.hpp"

using namespace std;
using namespace mm_ray;

inline unsigned char pixel_map(Real_t val){
  return (unsigned char)( std::min(val * 255, 255.f));
}

void writePPM(int dimx, int dimy, const char* file_name, Real_t* image_data){
  FILE *fp = fopen(file_name, "wb");
  if (fp == NULL){
    perror("File failed to open: ");
    exit(EXIT_FAILURE);
  }
  fprintf(fp, "P6\n %s\n %d\n %d\n %d\n", "", dimx, dimy, 255);
  for (int j = 0; j < dimy; ++j){
    for (int i = 0; i < dimx; ++i){
      static unsigned char color[3];
      color[0] = pixel_map(image_data[(j * dimx + i) * 3 + 0]);
      color[1] = pixel_map(image_data[(j * dimx + i) * 3 + 1]);
      color[2] = pixel_map(image_data[(j * dimx + i) * 3 + 2]);
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
  SceneParser<Accel> scn_parser;
  string fname(argv[1]);
  try {
    //Register the supported materials, geometries, and accelerators
    
    auto tmp0 = shared_ptr<PhongMaterialBuilder>(new PhongMaterialBuilder());
    auto t0 = static_pointer_cast<MaterialBuilder, PhongMaterialBuilder>(tmp0);
    scn_parser.Register_Material("phong", t0);

    auto tmp1 = shared_ptr<SphereBuilder>(new SphereBuilder());
    auto t1 = dynamic_pointer_cast<GeometryBuilder, SphereBuilder>(tmp1);
    scn_parser.Register_Geometry("sphere", t1);

    auto tmp2 = shared_ptr<PointBuilder>(new PointBuilder());
    auto t2 = dynamic_pointer_cast<GeometryBuilder, PointBuilder>(tmp2);
    scn_parser.Register_Geometry("point_light", t2);

    auto tmp3 = shared_ptr<TriangleMeshBuilder>(new TriangleMeshBuilder());
    auto t3 = dynamic_pointer_cast<GeometryBuilder, TriangleMeshBuilder>(tmp3);
    scn_parser.Register_Geometry("mesh", t3);
    auto tmp4 = shared_ptr<DistanceBuilder<Accel> >(new DistanceBuilder<Accel>());
    auto t4 = dynamic_pointer_cast<RendererBuilder<Accel>, DistanceBuilder<Accel>>(tmp4);
    scn_parser.Register_Renderer("distance", t4);
    auto tmp5 = shared_ptr<PhongBuilder<Accel> >(new PhongBuilder<Accel>());
    auto t5 = dynamic_pointer_cast<RendererBuilder<Accel>, PhongBuilder<Accel>>(tmp5);
    scn_parser.Register_Renderer("phong", t5);

    auto tmp6 = shared_ptr<HostBuilder<Accel> >(new HostBuilder<Accel>());
    auto t6 = dynamic_pointer_cast<RendererBuilder<Accel>, HostBuilder<Accel>>(tmp6);
    scn_parser.Register_Renderer("host", t6);

    //Now that all supported functionality is registered we can parse the file
    scn_parser.Parse(fname);
    cout << "Finished parsing file" << endl;
  } catch (ParsingException* e) {
    cout << "ParsingException" << endl;
    cout << e->what() << endl;
    e->clear();
    return 1;
  } catch (...) {
    cout << "Something really bad happened " << endl;
    return 1;
  }
  auto& renderer = scn_parser.Get_Renderer();
  renderer.Render();
  vector<Real_t> image = renderer.getImage();
  auto& scn = scn_parser.Get_Scene();
  cout << "Writing rendered image" << endl;
  writePPM(scn.output[1], scn.output[0], argv[2], &*image.begin());
  return 0;
}
