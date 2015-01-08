
#ifndef MM_BVHTREE
#define MM_BVHTREE
namespace mm_ray{
  class BVHTreeInit {
    std::vector<s_ptr<Geometry> > geometry;
  public:
    
    BVHTreeInit(){
    }
    
    void insertGeometry(std::vector<s_ptr<Geometry> > geom);
    
    void initialize();
  };
  
  class BVHTree {
    BVHTree() {
    }
  };
}
#endif
