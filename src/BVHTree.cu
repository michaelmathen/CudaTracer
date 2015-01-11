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


  BVHTree* BVHTree::Build_Accelerator(vector<Geometry*>& geom){
     thrust::device_vector<Geometry*> geometry_objects(geom.begin(), geom.end());

     
     Vec3 min_el_initial = (Real_t)INFINITY;
     Vec3 max_el_initial = (Real_t)-INFINITY;

     auto Get_Center = [](Geometry* geom) {
       return geom.getCenter();
     };

     thrust::device_vector<Vec3> centroids;
     transform(centroids.begin(), geometry.begin(), geometry.end(), Get_Center);

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
		       geom.begin(), geom.end(), 
		       centroids_To_1D_Hilbert(upper, lower));
     thrust::sort_by_key(dist_space_curve.begin(), 
			 dist_space_curve.end(),
			 geom.begin());

     dist_space_curve.clear();
     
     Geometry** geometry_data = thrust::raw_pointer_cast(&geom.begin());
     
     
  }

}
