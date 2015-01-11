#include <algorithm>
#include <thrust/device_vector.h>
#include "ray_defs.hpp"
#include "SpaceCurve.hpp"


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
    unsigned long operator() (Geometry* geom){

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
    unsigned long operator() (Geometry* geom){
    
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


  struct getCentroids {
    Vec3 operator()(Geometry* geom){
      return geom->getCenter();
    }
  }

    
  void BVHTreeInit::insertGeometry(vector<Geometry*>& geom){
    geometry.insert(geometry.end(), geom.begin(), geom.end());
  }

  void BVHTreeInit::initialize(){
    /*
      This builds the bvh tree using the LBVH construction method.
     */

    //Get the centroids from all of the geometry
    thrust::device_vector<Vec3> centroids;
    centroids.reserve(geometry.size());

    //Move all of the scene data over
    

    serialize_scene_alloc();
    
    getCentroids gcf;
    transform(centroids.begin(), geometry.begin(), geometry.end(), gcf);

    //Find a bounding box 
    Vec3 lower = thrust::reduce(centroids.begin(),
				centroids.end(),
				thrust::min_element<Vec3>());

    Vec3 upper = thrust::reduce(centroids.begin(),
				centroids.end(),
				thrust::max_element<Vec3>());
		   
    
    thrust::device_vector<unsigned long> length_into_curve;


#ifdef Z_ORDER
    int scale = 
    thrust::transform(length_into_curve.begin(),
		      length_into_curve.end(),
		      centriods.begin(),
		      centroids.end(),
		      centroids_To_1D_Z_Order(upper, lower));
#else
    int scale = 4194304;
    //For the scale we are going to use the finest space filling curve
    thrust::transform(length_into_curve.begin(),
		      length_into_curve.end(),
		      centriods.begin(),
		      centroids.end(),
		      centroids_To_1D_Hilbert(upper, lower));
#endif
    //Now we can sort the entire array
    thrust::sort(length_into_curve.begin(), length_into_curve.end());
    
  }
}
