#include <algorithm>
#include <thrust/device_vector.h>
#include "ray_defs.hpp"
#include "SpaceCurve.hpp"
#include "managed.hpp"

namespace mm_ray {
  using namespace std;


  struct Centroids_To_1D_Z_Order {
    Vec3 lower; //lower 
    Vec3 r; // r = scale / (upper - lower)  where upper and lower are the bounds of a bounding box.
            // We are scaling the values to a box of size scaleXscaleXscale

    Centroids_To_1D_Z_Order(Vec3 u, Vec3 l) :
      lower(l)
    {
      //This magic number below is 2 ^ 16 The zorder curves are using 2^16 2^16 and 2^16 for a total of 2^48
      r = (Real_t)65536.0 / (u - l);
    }
    
    __host__ __device__
    unsigned long operator() (Geometry const* geom){

      Vec3 centroid = geom->getCenter();
    
      //scale the centroid into the range of 0 to 1.
      Vec3 tmp_scaled = (centroid[ix] - lower) * r;
      unsigned long v0 = (unsigned long)tmp_scaled[0];
      unsigned long v1 = (unsigned long)tmp_scaled[1];
      unsigned long v2 = (unsigned long)tmp_scaled[2];
      return Z_Order_3D(v0, v1, v2);
    }
  };

  struct Centroids_To_1D_Hilbert {
    Vec3 lower; //lower 
    Vec3 r; // r = scale / (upper - lower)  where upper and lower are the bounds of a bounding box.
            // We are scaling the values to a box of size scaleXscaleXscale

    Centroids_To_1D_Hilbert(Vec3 u, Vec3 l, int scale) :
      lower(l)
    {
      r = (Real_t)2097152.0 / (u - l);
    }
    
    __host__ __device__
    unsigned long operator() (Geometry const* geom){
    
      Vec3 centroid = geom->getCenter();
      
      //scale the centroid into the range of 0 to 1.
      Vec3 tmp_scaled = (centroids[ix] - lower) * r;
      unsigned v0 = (unsigned long)tmp_scaled[0];
      unsigned v1 = (unsigned long)tmp_scaled[1];
      unsigned v2 = (unsigned long)tmp_scaled[2];
      
      //21 * 3 = 63. We can use almost the entire unsigned long integer
      //and get the finest possible space filling curve
      return Hilbert_Coord_To_Int_3D<21>(v0, v1, v2);
    }
  };

  __global__ void 
  FS_Layer_Reduce(BVHEntry* entry_data,
		  unsigned bvh_offset,
		  Geometry** geom_data,
		  unsigned geometry_length){
    unsigned entry_idx = threadIdx.x + blockDim.x * blockIdx.x;
    Vec3 lower1;
    Vec3 upper1;
    Vec3 lower2;
    Vec3 upper2;

    if (2 * entry_idx + 1 < geometry_length){
      geom_data[2 * entry_idx]->Bounding_Box(upper1, lower1);
      geom_data[2 * entry_idx + 1]->Bounding_Box(upper2, lower2);
      entry_data[bvh_offset + entry_idx] = BVHEntry(max(upper1, upper2), 
						    min(lower1, lower2));
    }  else if (2 * entry_idx < geometry_length) {
      geom_data[2 * entry_idx]->Bounding_Box(upper1, lower1);
      entry_data[bvh_offset + entry_idx] = BVHEntry(upper1, lower1);
    }
  }

  __global__ void 
  Layer_Reduce(BVHEntry* entry_data,
	       unsigned bvh_offset,
	       unsigned bvh_length){
    unsigned entry_idx = threadIdx.x + blockDim.x * blockIdx.x;

    Vec3 lower1;
    Vec3 upper1;
    Vec3 lower2;
    Vec3 upper2;

    unsigned r_idx = 2 * (bvh_offset + entry_idx) + 2; 
    unsigned l_idx = 2 * (bvh_offset + entry_idx) + 1; 

    if (l_idx >= bvh_length) {
      entry_data[l_idx].Bounding_Box(upper1, lower1);
      entry_data[r_idx].Bounding_Box(upper2, lower2);
      entry_data[bvh_offset + entry_idx] = BVHEntry(max(upper1, upper2), 
						    min(lower1, lower2));
    } 
  }
  
  BVHTree* BVHTree::Build_Accelerator(vector<Geometry*>& geom){
    Geometry** geometry_ptr = new Geometry*[geom.size()];
    copy(geom.begin(), geom.end(), geometry_ptr);
    
    Vec3 min_el_initial = (Real_t)INFINITY;
    Vec3 max_el_initial = (Real_t)-INFINITY;
    
    auto Get_Center = [](Geometry* g) {
      return g->getCenter();
    };
    
    thrust::device_vector<Vec3> centroids;
    transform(centroids.begin(), geometry_ptr, geometry_ptr + geom.size(), Get_Center);
    
    Vec3 lower = thrust::reduce(centroids.begin(),
				centroids.end(),
				min_el_initial,
				min);
    
    Vec3 upper = thrust::reduce(centroids.begin(),
				centroids.end(),
				max_el_initial,
				max);
    centroids.clear();
    
    thrust::device_vector<unsigned long> dist_space_curve;     
    thrust::transform(dist_space_curve.begin(), dist_space_curve.end(), 
		      geometry_ptr, geometry_ptr + geom.size(), 
		      centroids_To_1D_Hilbert(upper, lower));
    thrust::sort_by_key(dist_space_curve.begin(), 
			dist_space_curve.end(),
			geometry_ptr);
    
    dist_space_curve.clear();
    BVHEntry* bvh_entries = NULL;
    //Number of BVHEntries in the bottom level of the tree.
    unsigned base_nodes = geom.size() / 2 + geom.size() % 2;
    //Number of BVHEntries in total in the tree.
    unsigned total_nodes = 2 * base_nodes - 1;
    if (geom.size() > 0) {
      bvh_entries = new BVHEntry[total_nodes];
      //We want to pick the next biggest multiple of 32
      int block_size = base_nodes / 32 + (base_nodes % 32) != 0;
      FS_Layer_Reduce<<<block_size, 32>>>(bvh_entries, 
					  total_nodes - base_nodes, 
					  geometry_ptr,
					  geom.size());
       
      for (unsigned i = (total_nodes - base_nodes) >> 1; i != 0; i >>= 1){
	block_size = i / 32 + (i % 32) != 0;
	Layer_Reduce<<<block_size,32>>>(bvh_entries,
					i - 1,
					total_nodes);
      }
    }

    int light_length = thrust::count(geometry_ptr, 
				     geometry_ptr + geom.size(), 
				     [](Geometry* g){
				       return g->isLight();
				     });
    Geometry** lights = new Geometry*[light_length];
    thrust::copy_if(geometry_ptr, 
		    geometry_ptr + geom.size(),
		    lights,
		    [](Geometry* g){
		      return g->isLight();
		    });

    BVHTree* bvh_tree = new BVHTree();
    bvh_tree->bvh_entries = bvh_entries;
    bvh_tree->bvh_length = total_nodes;
    bvh_tree->geometry_ptr = geometry_ptr;
    bvh_tree->geometry_length = geom.size();
    bvh_tree->light_length = light_length;
    bvh_tree->lights = lights;
    return bvh_tree;
  }

}