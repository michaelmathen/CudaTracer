#include "ray_defs.hpp"

#ifndef MM_SPACE_CURVE
#define MM_SPACE_CURVE



template<unsigned bits>
__host__ __device__ 
unsigned long  Hilbert_Coord_To_Int_3D(unsigned x, unsigned y, unsigned z) {
  /*
    This code maps a 3d Hilburt coordinate to a distance along the curve
   */
  unsigned t;
  
  /* Inverse undo */
  for (unsigned Q = 1 << (bits - 1); Q > 1; Q >>= 1) {
    unsigned P = Q - 1;

    if (x & Q)
      x ^= P;  /* invert */

    if (y & Q)
      x ^= P;
    else  {
      t = (x ^ y) & P;
      x ^= t;
      y ^= t;
    }
    
    if (z & Q)
      x ^= P;
    else  {
      t = (x ^ z) & P;
      x ^= t;
      z ^= t;
    }
  }
  
  /* Gray encode (inverse of decode) */
  y ^= x;
  z ^= y;
  
  t = z;
  for (unsigned i = 1; i < bits; i <<= 1)
    z ^= z >> i;

  t ^= z;
  y ^= t;
  x ^= t;

  unsigned long output_distance = 0;
  for (int i = bits - 1; i >= 0; --i){
    output_distance |= (x >> i & 1) << (3 * i + 2);
    output_distance |= (y >> i & 1) << (3 * i + 1);
    output_distance |= (z >> i & 1) << (3 * i);
  }
  
  return output_distance;
}


/*
This assumes that the numbers are scaled to fit inside a 2^16x2^16x2^16 sized block
 */
__host__ __device__ unsigned long inline Z_Order_3D(unsigned long x, unsigned long y, unsigned long z){
     static const unsigned long B[] = {0x00000000FF0000FF, 0x000000F00F00F00F,
				       0x00000C30C30C30C3, 0X0000249249249249};           
     static const int S[] =  {16, 8, 4, 2}; 

     x = (x | (x << S[0])) & B[0];
     x = (x | (x << S[1])) & B[1];
     x = (x | (x << S[2])) & B[2];
     x = (x | (x << S[3])) & B[3];

     y = (y | (y << S[0])) & B[0];
     y = (y | (y << S[1])) & B[1];
     y = (y | (y << S[2])) & B[2];
     y = (y | (y << S[3])) & B[3];

     z = (z | (z <<  S[0])) & B[0];
     z = (z | (z <<  S[1])) & B[1];
     z = (z | (z <<  S[2])) & B[2];
     z = (z | (z <<  S[3])) & B[3];

     return ( x | (y << 1) | (z << 2) );
}

#endif
