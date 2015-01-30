#include <cstring>
#include <iostream>

#include "ray_defs.hpp"
#include "Managed.hpp"

#ifndef MM_SCENE
#define MM_SCENE
namespace mm_ray {
class Scene : public Managed {
public:
  Vec3 cam_loc;
  Vec3 cam_dir;
  Vec3 cam_up;
  Vec3 cam_right;
  Vec2 viewport;
  int output[2];
  int samples;
  int render_block_x;
  int render_block_y;
  
  Scene(){}
  Scene(Vec2& view,
	int out_x,
	int out_y,
	Vec3& cam_loc,
	Vec3& cam_dir,
	Vec3& cam_up,
	int samples) : cam_loc(cam_loc),
		       cam_dir(cam_dir),
		       cam_up(cam_up) {
    
    this->samples = samples;
    output[0] = out_x;
    output[1] = out_y;

    //Now define the right vector and redefine up to make sure the up was defined ortho to
    //direction
    cam_right = cross(cam_dir, cam_up);
    cam_up = cross(cam_right, cam_dir);
    viewport[0] = view[0];
    viewport[1] = view[1];
  }

  void print(){
    std::cout << cam_loc[0] << " " << cam_loc[1] << " " << cam_loc[2] << std::endl;
    std::cout << cam_dir[0] << " " << cam_dir[1] << " " << cam_dir[2] << std::endl;
    std::cout << cam_up[0] << " " << cam_up[1] << " " << cam_up[2] << std::endl;
    std::cout << cam_right[0] << " " << cam_right[1] << " " << cam_right[2] << std::endl;
    std::cout << viewport[0] << " " << viewport[1] << std::endl;
    std::cout << output[0] <<  " " << output[1] << std::endl;
    std::cout << samples << std::endl;
  }
  /*
  Scene& operator=(Scene& scn){
    std::memcpy(this, &scn, sizeof(Scene));
    return *this;
  }
  */
  
};

}
#endif
