#include <math.h>
#include "cuda_defs.h"

#ifndef MMVector_Math
#define MMVector_Math


template<typename T, unsigned int N>
class DataWrapper {
  T _data[N];
public:
  __host__ __device__ inline T
  operator[](unsigned int i) const {
    return _data[i];
  }
  
  __host__ __device__ inline T&
  operator[](unsigned int i) {
    return _data[i];
  }

};

template<typename T, typename V, unsigned int from>
struct COPY {
  __host__ __device__ inline static
  void apply(T& destination, const V& source) {
    destination[from] = source[from];
    COPY<T, V, from-1>::apply(destination, source);
  }
};

// terminal case
template<typename T, typename V>
struct COPY<T, V, 0> {
  __host__ __device__ inline static void
  apply(T& destination, const V& source) {
    destination[0] = source[0];
  }
};


template<typename T, typename V, unsigned int from>
struct SET {
  __host__ __device__ inline static void
  apply(T& destination, const V source) {
    destination[from] = source;
    SET<T, V, from-1>::apply(destination, source);
  }
};

// terminal case
template<typename T, typename V>
struct SET<T, V, 0> {
  __host__ __device__
  inline static void apply(T& destination, const V source) {
    destination[0] = source;
  }
};


///////////////////////////////////////////////////////////////////////////////
////////////////////////////Vector ND Expresion////////////////////////////////
///////////////////////////////////////////////////////////////////////////////


//We define an ast node 
template<typename E, typename T, unsigned int N>
class VecNdExpr {
public:
  __host__ __device__
  inline unsigned int size() const {
    return N;
  }

  __host__ __device__
  inline T operator[](unsigned int i) const {
    return static_cast<E const&>(*this)[i];
  }

  __host__ __device__ inline
  operator E&() {
    return static_cast<E&>(*this);
  }
  
  __host__ __device__ inline operator E
  const&() const {
    return static_cast<const E&>(*this);
  }
 
};


///////////////////////////////////////////////////////////////////////////////
////////////////////////////Vector Class///////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////

//This is a data element of the ast tree
template<typename T, unsigned int N>
class VecNd : public VecNdExpr<VecNd<T, N>, T, N> {
  typedef DataWrapper<T, N> container_t;
  container_t _data;
public:
  __host__ __device__
  VecNd(){
  }
  
  __host__ __device__ inline unsigned int
  size() const {
    return N;
  }

  __host__ __device__ inline T
  operator[](unsigned int i) const {
    return _data[i];
  }

  __host__ __device__ inline T&
  operator[](unsigned int i) {
    return _data[i];
  }

  __host__ __device__ inline VecNd<T, N>&
  operator=(T const& val){
    SET<container_t, T, N>::apply(_data, val);
    return *this;
  }

  template<typename E>
  __host__ __device__ inline
  VecNd(VecNdExpr<E, T, N> const& v_expr){
    E const& v = v_expr;
    COPY<container_t, E, N>::apply(_data, v);
  }

  __host__ __device__ inline
  VecNd(T v[N]){
    COPY<container_t, T*, N>::apply(_data, &v[0]);
  }
  
  template<typename E>
  __host__ __device__ inline VecNd<T, N>&
  operator=(VecNdExpr<E, T, N> const& v_expr){
    E const& v = v_expr;
    COPY<container_t, E, N>::apply(_data, v);
    return *this;
  }

};


/*
Since most of the template code is duplicated I defined this macro.
 */

#define VECTOR_BINARY_OPERATION(OPERATION, NAME)     \
template<typename E1, typename E2, typename T, unsigned int N> \
class Vec##NAME : public VecNdExpr<Vec##NAME<E1, E2, T, N>, T, N> { \
  E1 const& _u; \
  E2 const& _v; \
public:        \
  __host__ __device__ inline   \
  Vec##NAME(VecNdExpr<E1, T, N> const& u, VecNdExpr<E2, T, N> const& v)  \
    : _u(u), _v(v)    \
  {}   \
  __host__ __device__ inline T   \
  operator[](unsigned int i) const { return _u[i] OPERATION _v[i] ; }  \
};  \
    \
template<typename E1, typename E2, typename T, unsigned int N>   \
__host__ __device__ inline Vec##NAME<E1,E2,T, N> const   \
operator OPERATION (VecNdExpr<E1, T, N> const& u_1, VecNdExpr<E2, T, N> const& u_2) {  \
  return Vec##NAME<E1, E2, T, N>(u_1, u_2); \
}  \
   \
template<typename E2, typename T, unsigned int N>  \
__host__ __device__ inline VecNd<T, N>&  \
operator OPERATION##=(VecNd<T, N>& u_1, VecNdExpr<E2, T, N> const& u_2) { \
  u_1 = u_1 OPERATION u_2; \
  return u_1; \
} 


VECTOR_BINARY_OPERATION(/, Division)
VECTOR_BINARY_OPERATION(*, Multiplication)
VECTOR_BINARY_OPERATION(+, Addition)
VECTOR_BINARY_OPERATION(-, Difference)


#define VECTOR_SCALAR_OPERATION(OPERATION, NAME) \
template<typename E1, typename T, unsigned int N> \
class VecScaler##NAME : public VecNdExpr<VecScaler##NAME<E1, T, N>, T, N> { \
  E1 const& _u; \
  T _v; \
public: \
  __host__ __device__ inline \
  VecScaler##NAME(VecNdExpr<E1, T, N> const& u, T v)	\
    : _u(u), _v(v) \
  {} \
  __host__ __device__ T \
  operator[](unsigned int i) const { return _u[i] OPERATION _v; } \
};  \
 \
template<typename E1, typename T, unsigned int N> \
__host__ __device__ inline VecScaler##NAME<E1, T, N> const \
operator OPERATION (VecNdExpr<E1, T, N> const& u_1, T val) { \
  return VecScaler##NAME<E1, T, N>(u_1, val); \
} \
 \
template<typename E1, typename T, unsigned int N> \
__host__ __device__ inline VecScaler##NAME<E1, T, N> const \
operator OPERATION (T val, VecNdExpr<E1, T, N> const& u_1) { \
  return VecScaler##NAME<E1, T, N>(u_1, val); \
} \
 \
template<typename E2, typename T, unsigned int N> \
__host__ __device__ inline VecNd<T, N>&  \
operator OPERATION##=(VecNd<T, N>& u_1, T val) { \
  u_1 = u_1 OPERATION val; \
  return u_1; \
} 

VECTOR_SCALAR_OPERATION(/, Division)
VECTOR_SCALAR_OPERATION(*, Multiplication)


///////////////////////////////////////////////////////////////////////////////
////////////////////////////Vector Scaler Exponentiation///////////////////////
///////////////////////////////////////////////////////////////////////////////


template<typename E1, typename T, unsigned int N>
class VecScalerExponentiation : public VecNdExpr<VecScalerExponentiation<E1, T, N>, T, N> {
  E1 const& _u;
  T _v;
public:
  __host__ __device__ inline 
  VecScalerExponentiation(VecNdExpr<E1, T, N> const& u, T const& v)
    : _u(u), _v(v)
  {}
  __host__ __device__ T
  operator[](unsigned int i) const { return (T)pow(_u[i], _v); }
};

template<typename E1, typename T, unsigned int N>
__host__ __device__ inline VecScalerExponentiation<E1, T, N> const
operator^(VecNdExpr<E1, T, N> const& u_1, T val) {
  return VecScalerExponentiation<E1, T, N>(u_1, val);
}

template<typename E2, typename T, unsigned int N>
__host__ __device__ inline VecNd<T, N>& 
operator^=(VecNd<T, N>& u_1, T val) {
  u_1 = u_1 ^ val;
  return u_1;
}


///////////////////////////////////////////////////////////////////////////////
////////////////////////////Vector Dot Product/////////////////////////////////
///////////////////////////////////////////////////////////////////////////////

template<typename E1, typename E2, typename T, unsigned int from>
struct DOT {
  __host__ __device__ inline static T
  apply(const E1& e1, const E2& e2) {
    return e1[from] * e2[from] + DOT<E1, E2, T, from-1>::apply(e1, e2);
  }
};

template<typename E1, typename E2, typename T>
struct DOT<E1, E2, T, 0> {
  __host__ __device__ inline static T
  apply(const E1& e1, const E2& e2) {
    return e1[0] * e2[0];
  }
};

template<typename E1, typename E2, typename T, unsigned int N>
__host__ __device__ T const
dot(VecNdExpr<E1, T, N> const& u_1, VecNdExpr<E2, T, N> const& u_2) {
  return DOT<E1, E2, T, N>::apply(u_1, u_2);
}

///////////////////////////////////////////////////////////////////////////////
////////////////////////////Vector Magnitude///////////////////////////////////
///////////////////////////////////////////////////////////////////////////////

template<typename E1, typename T, unsigned int N>
__host__ __device__ T const
mag(VecNdExpr<E1, T, N> const& u_1) {
  return sqrt(DOT<E1, E1, T, N>::apply(u_1, u_1));
}

///////////////////////////////////////////////////////////////////////////////
////////////////////////////Vector Sum/////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////

template<typename E1, typename T, unsigned int from>
struct SUM {
  __host__ __device__ inline static T
  apply(const E1& e1) {
    return e1[from] + SUM<E1, T, from-1>::apply(e1);
  }
};

template<typename E1, typename T>
struct SUM<E1, T, 0> {
  __host__ __device__ inline static T
  apply(const E1& e1) {
    return e1[0];
  }
};

template<typename E1, typename T, unsigned int N>
__host__ __device__ T const
sum(VecNdExpr<E1, T, N> const& u_1) {
  return SUM<E1, T, N>::apply(u_1);
}


///////////////////////////////////////////////////////////////////////////////
////////////////////////////Vector Cross Product///////////////////////////////
///////////////////////////////////////////////////////////////////////////////

template<typename E1, typename E2, typename T>
class VecCross : public VecNdExpr<VecCross<E1, E2, T>, T, 3> {
  E1 const& _u;
  E2 const& _v;
public:
  __host__ __device__ inline
  VecCross(VecNdExpr<E1, T, 3> const& u, VecNdExpr<E2, T, 3> const& v)
    : _u(u), _v(v)
  {}
  __host__ __device__ inline T
  operator[](unsigned int i) const {
    //There is a probably a better way to do this...
    return _u[(i + 1) % 3] * _v[(i + 2) % 3] - _u[(i + 2) % 3] * _v[(i + 1) % 3];
  }
};

template<typename E1, typename E2, typename T>
__host__ __device__ inline VecCross<E1, E2, T> const
cross(VecNdExpr<E1, T, 3> const& u_1, VecNdExpr<E2, T, 3> const& u_2) {
  return VecCross<E1, E2, T>(u_1, u_2);
}



typedef VecNd<Real_t, 4> Vec4;
typedef VecNd<Real_t, 3> Vec3;
typedef VecNd<Real_t, 2> Vec2;

#endif 

