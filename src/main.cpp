//extern "C" {
//#include "bmpWrapper.h"
//}
#include <iostream>
#include <fstream>
#include <thread>
#include <algorithm>
#include <vector>
#include <memory>
#include "ray_defs.hpp"
#include "ParseScene.hpp"
#include "SceneContainer.hpp"
#include "ParsingException.hpp"
#include "DistanceRenderer.hpp"
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

  Renderer<SceneContainer>* renderer;
  shared_ptr<SceneContainer> cont(new SceneContainer());
  SceneContainerHost hcontainer;
  
  string fname(argv[1]);
  try {
    parse_file(fname, host_scene, cont, &renderer, hcontainer);
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
  
  cont->initialize();
  renderer->Render();
  vector<Real_t> image = renderer->getImage();
  cout << "Writing rendered image" << endl;
  
  writePPM(host_scene.output[1], host_scene.output[0], argv[2], &*image.begin());
  //if (argc > 2){
  //int code = writeImage(argv[2], host_scene.output[0], host_scene.output[1], &*imagef.begin());
    /*switch(code){
    case BMP_CREATE_ERROR:
      cerr << "Failed to create output file" << endl;
      break;
    case BMP_SET_FAILED:
      cerr << "Something is wrong with the data buffer" << endl;
      break;
    case BMP_SAVE_ERROR:
      cerr << "Was not able to save the output file" << endl;
      break;
    case BMP_SUCCESS:
      break;
    };
  }
    */
  return 0;
}
