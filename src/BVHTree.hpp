#ifndef MM_BVHTREE
#define MM_BVHTREE
namespace mm_ray{

  class BVHEntry : Managed {
    Vec3 upper;
    Vec3 lower;

    BVHEntry(Vec3 const& ub, Vec3 const& lb) 
      : upper(ub), lower(lb)
    {}

    __host__ __device__ inline Real_t Intersect_Ray(Ray const& ray) const {
      
    }
    __host__ __device__ inline void Bounding_Box(Vec3& upper, Vec3& lower) const {
      lower = this->lower;
      upper = this->upper;
    }
  };

  class BVHTree : Managed {
    BVHEntry* bvh_entries;
    int bvh_length;
    int bvh_depth;
    Geometry* geometry_ptr;
    int geometry_length;

    Geometry* lights;
    int light_length;
  public:
    BVHTree() {}
    static BVHTree* Build_Accelerator(vector<Geometry*>& geom);
    
    __host__ __device__ void Intersect(Ray const& ray, Hit& prop) const {
      //Check if we enter the scene
      if (bvh_entries[0].Intersect_Ray(ray) == INFINITY)
	return;
      
      //Allocate memory for a stack equal to the depth of the tree.
      int stack_indices[bvh_depth + 1];
      int stack_ptr = 0;
      int i = 0;
      while (true) {
	
	Real_t dist1 = bvh_entries[2 * i + 1].Intersect_Ray(ray);
	Real_t dist2 = bvh_entries[2 * i + 2].Intersect_Ray(ray);
	if (dist1 != INFINITY){
	  
	}

	if (dist2 != INFINITY){
	  
	}

      }
    
    }
  };
}
#endif
