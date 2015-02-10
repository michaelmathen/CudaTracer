#include <vector>

#include "ray_defs.hpp"
#include "Accelerator.hpp"
#include "Geometry.hpp"


#ifndef MM_BVHTREE
#define MM_BVHTREE
namespace mm_ray{

  struct BVHNode : public AxisAlignedBox {
    BVHNode(Vec3 const& u, Vec3 const&l) :
      AxisAlignedBox(u, l)
    {};
    Geometry const* left;
    Geometry const* right;
  };
  
  class BVHTreeSimple : public Accelerator<BVHTreeSimple> {
    const Geometry* bvh_entries;
    int bvh_depth;

  public:

    BVHTreeSimple(std::vector<Geometry*>& geom);
    ~BVHTreeSimple();

    __host__ __device__ void Intersect(Ray const& ray, Hit& prop) const {
      //Check if we enter the scene
      typedef const Geometry* Gc_t;
      //Allocate memory for a stack equal to the depth of the tree.
      //this->bvh_depth
      Gc_t node_stack[32];
      int stack_ptr = -1;
      prop.distance = INFINITY;      
      Geometry const* curr_node = bvh_entries;
      
      while (true) {
	//Check to see if we are at a leaf
	if (curr_node->Geometry_Type() != AABB){
	  curr_node->Intersect_Ray(ray, prop);
	  if (prop.distance != INFINITY || stack_ptr < 0) 
	    return;
	  //printf("Got Here %d\n", stack_ptr);
	  curr_node = node_stack[stack_ptr];
	  --stack_ptr;
	  continue;
	}
	//We need to figure out which bounding box is closer
	Real_t dl = static_cast<BVHNode const*>(curr_node)->left->Intersect_Ray(ray);
	Real_t dr = static_cast<BVHNode const*>(curr_node)->right->Intersect_Ray(ray);
	//We didn't intersect anything
	if (dl == INFINITY && 
	    dr == INFINITY) {
	  if (stack_ptr < 0) 
	    return;
	  //Try the next pausible node up the tree
	  curr_node = node_stack[stack_ptr];
	  --stack_ptr;
	  continue;
	}
	//We only intersected one box
	if (dr == INFINITY){
	  curr_node = static_cast<BVHNode const*>(curr_node)->left;
	  continue;
	}
	if (dl == INFINITY){
	  curr_node = static_cast<BVHNode const*>(curr_node)->right;
	  continue;
	}
	//if we have gotten this far then we have intersected both boxes
	//find the nearest box and descend down its tree
	stack_ptr++;
	if (dl < dr) {
	  curr_node = static_cast<BVHNode const*>(curr_node)->left;
	  node_stack[stack_ptr] = static_cast<BVHNode const*>(curr_node)->right;
	} else {
	  node_stack[stack_ptr] = static_cast<BVHNode const*>(curr_node)->left;
	  curr_node = static_cast<BVHNode const*>(curr_node)->right;
	}
      }
    }
  };
}
#endif
