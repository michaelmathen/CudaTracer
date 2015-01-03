#include <math.h>
#include "ray_defs.hpp"

#ifndef MM_TRANSFORM
#define MM_TRANSFORM
/*
Applies a tranformation matrix to vertex points
*/

namespace mm_ray {
  class Transform {
    Vec3 translation;
    Vec3 r_1;
    Vec3 r_2;
    Vec3 r_3;
    
  public:

    Transform(Real_t scale, Real_t angle, Vec3 const& rotation_axis, Vec3 const& tran)
      : translation(tran)
    {
      
      //Convert to radians
      Real_t theta = angle * M_PI / (Real_t)180.0;
      Real_t cos_t = (Real_t)cos(theta);
      Real_t sin_t = (Real_t)sin(theta);
      Vec3 norm_rot = rotation_axis / mag(rotation_axis);
      //Set everything to 0
      Vec3 tmp_r1;
      Vec3 tmp_r2;
      Vec3 tmp_r3;

      //The below computes the cross product matrix
      tmp_r1[0] = 0;
      tmp_r1[1] = -norm_rot[2];
      tmp_r1[2] = norm_rot[1];
      
      tmp_r2[0] = norm_rot[2];
      tmp_r2[1] = 0;
      tmp_r2[2] = -norm_rot[0];

      tmp_r3[0] = -norm_rot[1];
      tmp_r3[1] = norm_rot[0];
      tmp_r3[2] = 0;

      Vec3 tmp_I1;
      Vec3 tmp_I2;
      Vec3 tmp_I3;

      //The below computes the cross product matrix
      tmp_I1[0] = cos_t;
      tmp_I1[1] = 0;
      tmp_I1[2] = 0;
      
      tmp_I2[0] = 0;
      tmp_I2[1] = cos_t;
      tmp_I2[2] = 0;

      tmp_I3[0] = 0;
      tmp_I3[1] = 0;
      tmp_I3[2] = cos_t;

      //This code is computing the transformation matrix minus the translation part
      r_1 = (tmp_I1 + sin_t * tmp_r1 + (1 - cos_t) * norm_rot * norm_rot[0]) * scale;
      r_2 = (tmp_I2 + sin_t * tmp_r2 + (1 - cos_t) * norm_rot * norm_rot[1]) * scale;
      r_3 = (tmp_I3 + sin_t * tmp_r3 + (1 - cos_t) * norm_rot * norm_rot[2]) * scale;

    }
    
    inline Vec3 applyTransform(Vec3 const& point) const {
      Vec3 rotated_pnt;
      rotated_pnt[0] = dot(r_1, point);
      rotated_pnt[1] = dot(r_2, point);
      rotated_pnt[2] = dot(r_3, point);
      return Vec3(rotated_pnt + translation);
      //return point;
    }
  };
}

#endif
