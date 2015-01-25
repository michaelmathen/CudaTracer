#include <vector>
#include <algorithm>
#include <deque>
#include "SpaceCurve.hpp"
#include "BVHTreeSimple.hpp"

namespace mm_ray {
  using namespace std;

  BVHTreeSimple* BVHTreeSimple::Build_Accelerator(vector<Geometry*>& geom){
    /*
      To construct a BVH tree we use a really simple method 
      where we map all the elements to an index on a space
      curve and then construct the tree from the sorted 
      indices
     */
    vector<Vec3> centroids; 
    centroids.resize(geom.size());
    transform(geom.begin(), geom.end(), centroids.begin(), [](Geometry const* geom){
	return geom->getCenter();
      });
    
    Vec3 l = INFINITY;
    l = accumulate(centroids.begin(), centroids.end(), l, 
		   [](Vec3 const& v1, Vec3 const& v2){
		     return min(v1, v2);
		   });
    
    Vec3 u = -INFINITY;
    u= accumulate(centroids.begin(), centroids.end(), u, 
		  [](Vec3 const& v1, Vec3 const& v2){
		    return max(v1, v2);
		  });
    
    
    Vec3 r = (Real_t)2097152.0 / (u - l);
    typedef pair<unsigned long, Geometry const*> Geom_Pair;
    vector<Geom_Pair> ind_pair;
    ind_pair.resize(geom.size());
    transform(geom.begin(), geom.end(), ind_pair.begin(), 
	      [&r, &l](Geometry const* geom){
		Vec3 centroid = geom->getCenter();
		//scale the centroid into the range of 0 to 1.
		Vec3 tmp_scaled = (centroid - l) * r;
		unsigned v0 = (unsigned long)tmp_scaled[0];
		unsigned v1 = (unsigned long)tmp_scaled[1];
		unsigned v2 = (unsigned long)tmp_scaled[2];
		
		//21 * 3 = 63. We can use almost the entire unsigned long integer
		//and get the finest possible space filling curve
		return make_pair(Hilbert_Coord_To_Int_3D<21>(v0, v1, v2), geom);
	      });
    
    sort(ind_pair.begin(), ind_pair.end(), 
	 [](Geom_Pair const& l1, Geom_Pair const& l2){
	   return l1.first > l2.first;
	 });
    
    typedef pair<int, const Geometry*> node_t;
    deque<node_t> curr_nodes;

    
    for (unsigned int j = 0; j < geom.size(); j++)
      curr_nodes.push_back(make_pair(1, ind_pair[j].second));
    
    Vec3 u1, l1, u2, l2;      
    while (curr_nodes.size() != 1){
      auto n1 = curr_nodes.back();
      curr_nodes.pop_back();
      auto n2 =  curr_nodes.back();
      curr_nodes.pop_back();
      
      n1.second->Bounding_Box(u1, l1);
      n2.second->Bounding_Box(u2, l2);
      BVHNode* new_n = new BVHNode(max(u1, u2), min(l1, l2));
      //cout << new_n->upper[0] << " " << new_n->upper[1] << " " << new_n->upper[2] << endl;
      new_n->left = n1.second;
      new_n->right = n2.second;
      curr_nodes.push_front(make_pair(max(n1.first, n2.first) + 1, new_n));
    }
    
    int light_length = count_if(geom.begin(), geom.end(), 
				[](Geometry const* val){
				  return val->isLight();
				});
    const Geometry** lights = new Geometry const*[light_length];

    copy_if(geom.begin(), geom.end(), lights, 
	    [](Geometry const* val){
	      return val->isLight();
	    });

    BVHTreeSimple* tree = new BVHTreeSimple(curr_nodes.front().first, 
					    curr_nodes.front().second, 
					    light_length,
					    lights);
    return tree;
  }

  BVHTreeSimple::~BVHTreeSimple(){
    deque<Geometry const*> nodes;
    nodes.push_back(bvh_entries);
    while (nodes.size() != 0) {
      Geometry const* parent = nodes.back();
      nodes.pop_back();
      if (parent->Geometry_Type() == AABB) {
	nodes.push_back(static_cast<BVHNode const*>(parent)->left);
	nodes.push_back(static_cast<BVHNode const*>(parent)->right);
	delete parent;
      } 
    }
    delete lights;
  }
}
