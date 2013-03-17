#include "ImageCleaner.h"

#ifndef SIZEX
#error Please define SIZEX.
#endif
#ifndef SIZEY
#error Please define SIZEY.
#endif

#define SIZE SIZEX

typedef unsigned int uint;

#if !defined(CUFFT) || CUFFT == 0

//The code below is based on this paper 
//Reference: http://www.cs.berkeley.edu/~kubitron/courses/cs258-S08/projects/reports/project6_report.pdf

#define _USE_MATH_DEFINES
#include <math.h>

inline dim3 grid2D(int nblocks) {
  int slices = 1;
  while (nblocks / slices > 65535) 
    slices *= 2;
  return dim3(nblocks/slices, slices);
}

inline __device__ float2 operator*(float2 a, float2 b) { 
  return make_float2(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x); 
}
inline __device__ float2 operator*(float2 a, float b) { 
  return make_float2(b * a.x, b * a.y); 
}
inline __device__ float2 operator+(float2 a, float2 b) { 
  return make_float2(a.x + b.x, a.y + b.y); 
}
inline __device__ float2 operator-(float2 a, float2 b) { 
  return make_float2(a.x - b.x, a.y - b.y); 
}

#define COS_PI_8  0.923879533f
#define SIN_PI_8  0.382683432f
#define EXP_1_16  make_float2(COS_PI_8, -SIN_PI_8)
#define EXP_3_16  make_float2(SIN_PI_8, -COS_PI_8)
#define EXP_5_16  make_float2(-SIN_PI_8, -COS_PI_8)
#define EXP_7_16  make_float2(-COS_PI_8, -SIN_PI_8)
#define EXP_9_16  make_float2(-COS_PI_8,  SIN_PI_8)
#define EXP_1_8   make_float2(M_SQRT1_2, -M_SQRT1_2)
#define EXP_1_4   make_float2(0, -1)
#define EXP_3_8   make_float2(-M_SQRT1_2, -M_SQRT1_2)

#define IEXP_1_16  make_float2(COS_PI_8,  SIN_PI_8)
#define IEXP_3_16  make_float2(SIN_PI_8,  COS_PI_8)
#define IEXP_5_16  make_float2(-SIN_PI_8,  COS_PI_8)
#define IEXP_7_16  make_float2(-COS_PI_8,  SIN_PI_8)
#define IEXP_9_16  make_float2(-COS_PI_8, -SIN_PI_8)
#define IEXP_1_8   make_float2(M_SQRT1_2, M_SQRT1_2)
#define IEXP_1_4   make_float2(0, 1)
#define IEXP_3_8   make_float2(-M_SQRT1_2, M_SQRT1_2)

inline __device__ float2 exp_i(float phi) {
  return make_float2(__cosf(phi), __sinf(phi));
}


enum Axis {
  AXIS_X, AXIS_Y
};

enum Direction {
  FORWARD, INVERSE
};

template<int radix> inline __device__ int rev(int bits);

template<> inline __device__ int rev<2>(int bits) {
  return bits;
}

template<> inline __device__ int rev<4>(int bits) {
  int reversed[] = {0,2,1,3};
  return reversed[bits];
}

template<> inline __device__ int rev<8>(int bits) {
  int reversed[] = {0,4,2,6,1,5,3,7};
  return reversed[bits];
}

template<> inline __device__ int rev<16>(int bits) {
  int reversed[] = {0,8,4,12,2,10,6,14,1,9,5,13,3,11,7,15};
  return reversed[bits];
}

inline __device__ int rev4x4(int bits) {
  int reversed[] = {0,2,1,3, 4,6,5,7, 8,10,9,11, 12,14,13,15};
  return reversed[bits];
}
//
// basic fft
//
template<Direction dir> inline __device__ void FFT2(float2 &a0, float2 &a1) { 
  float2 c0 = a0;
  a0 = c0 + a1; 
  a1 = c0 - a1;
}

template<Direction dir> inline __device__ void FFT4(float2 &a0, float2 &a1, float2 &a2, float2 &a3) {
  FFT2<dir>(a0, a2);
  FFT2<dir>(a1, a3);
  if (dir == FORWARD)
    a3 = a3 * EXP_1_4;
  else
    a3 = a3 * IEXP_1_4;
  FFT2<dir>(a0, a1);
  FFT2<dir>(a2, a3);
}

template<Direction dir> inline __device__ void FFT2(float2 *a) { 
  FFT2<dir>(a[0], a[1]); 
}

template<Direction dir> inline __device__ void FFT4(float2 *a) { 
  FFT4<dir>(a[0], a[1], a[2], a[3]); 
}

template<Direction dir> inline __device__ void FFT8(float2 *a) {
  FFT2<dir>(a[0], a[4]);
  FFT2<dir>(a[1], a[5]);
  FFT2<dir>(a[2], a[6]);
  FFT2<dir>(a[3], a[7]);
  
  if (dir == FORWARD) {
    a[5] = a[5] * EXP_1_8;
    a[6] = a[6] * EXP_1_4;
    a[7] = a[7] * EXP_3_8;
  } else {
    a[5] = a[5] * IEXP_1_8;
    a[6] = a[6] * IEXP_1_4;
    a[7] = a[7] * IEXP_3_8;
  }
  FFT4<dir>(a[0], a[1], a[2], a[3]);
  FFT4<dir>(a[4], a[5], a[6], a[7]);
}

template<Direction dir> inline __device__ void FFT16(float2 *a) {
  FFT4<dir>(a[0], a[4], a[8], a[12]);
  FFT4<dir>(a[1], a[5], a[9], a[13]);
  FFT4<dir>(a[2], a[6], a[10], a[14]);
  FFT4<dir>(a[3], a[7], a[11], a[15]);

  if (dir == FORWARD) {
    a[5] = a[5]  * EXP_1_8;
    a[6] = a[6]  * EXP_1_4;
    a[7] = a[7]  * EXP_3_8;
    a[9] =  a[9]  * EXP_1_16;
    a[10] = a[10] * EXP_1_8;
    a[11] = a[11] * EXP_3_16;
    a[13] = a[13] * EXP_3_16;
    a[14] = a[14] * EXP_3_8;
    a[15] = a[15] * EXP_9_16;
  } else {
    a[5]  = a[5]  * IEXP_1_8;
    a[6]  = a[6]  * IEXP_1_4;
    a[7]  = a[7]  * IEXP_3_8;
    a[9]  = a[9]  * IEXP_1_16;
    a[10] = a[10] * IEXP_1_8;
    a[11] = a[11] * IEXP_3_16;
    a[13] = a[13] * IEXP_3_16;
    a[14] = a[14] * IEXP_3_8;
    a[15] = a[15] * IEXP_9_16;
  }
  FFT4<dir>(a[0],  a[1],  a[2],  a[3]);
  FFT4<dir>(a[4],  a[5],  a[6],  a[7]);
  FFT4<dir>(a[8],  a[9],  a[10], a[11]);
  FFT4<dir>(a[12], a[13], a[14], a[15]);
}

template<Direction dir> inline __device__ void FFT4x4(float2 *a);

template<> inline __device__ void FFT4x4<FORWARD>(float2 *a)
{
    FFT4<FORWARD>(a[0],  a[1],  a[2],  a[3]);
    FFT4<FORWARD>(a[4],  a[5],  a[6],  a[7]);
    FFT4<FORWARD>(a[8],  a[9],  a[10], a[11]);
    FFT4<FORWARD>(a[12], a[13], a[14], a[15]);
}

template<> inline __device__ void FFT4x4<INVERSE>(float2 *a)
{
    FFT2<INVERSE>(a[0], a[2]);
    FFT2<INVERSE>(a[1], a[3]);
    FFT2<INVERSE>(a[4], a[6]);
    FFT2<INVERSE>(a[5], a[7]);
    FFT2<INVERSE>(a[8], a[10]);
    FFT2<INVERSE>(a[9], a[11]);
    FFT2<INVERSE>(a[12], a[14]);
    FFT2<INVERSE>(a[13], a[15]);

    a[3] = a[3] * IEXP_1_4;
    a[7] = a[7] * IEXP_1_4;
    a[11] = a[11] * IEXP_1_4;
    a[15] = a[15] * IEXP_1_4;

    FFT2<INVERSE>(a[0], a[1]);
    FFT2<INVERSE>(a[2], a[3]);
    FFT2<INVERSE>(a[4], a[5]);
    FFT2<INVERSE>(a[6], a[7]);
    FFT2<INVERSE>(a[8], a[9]);
    FFT2<INVERSE>(a[10], a[11]);
    FFT2<INVERSE>(a[12], a[13]);
    FFT2<INVERSE>(a[14], a[15]);
}

//
//  loads
//
template<int n> inline __device__ void load(float2 *a, float2 *x, int sx) {
    for (int i = 0; i < n; i++)
        a[i] = x[i*sx];
}

template<int n> inline __device__ void loadx(float2 *a, float *x, int sx) {
    for (int i = 0; i < n; i++)
        a[i].x = x[i*sx];
}

template<int n> inline __device__ void loady(float2 *a, float *x, int sx) {
    for (int i = 0; i < n; i++)
        a[i].y = x[i*sx];
}

template<int n> inline __device__ void loadx(float2 *a, float *x, int *ind) {
    for (int i = 0; i < n; i++)
        a[i].x = x[ind[i]];
}

template<int n> inline __device__ void loady(float2 *a, float *x, int *ind) {
    for (int i = 0; i < n; i++)
        a[i].y = x[ind[i]];
}

//
//  stores, input is in bit reversed order
//
template<int n> inline __device__ void store(float2 *a, float2 *x, int sx) {
#pragma unroll
    for (int i = 0; i < n; i++)
        x[i*sx] = a[rev<n>(i)];
}
template<int n> inline __device__ void storex(float2 *a, float *x, int sx) {
#pragma unroll
    for (int i = 0; i < n; i++)
        x[i*sx] = a[rev<n>(i)].x;
}
template<int n> inline __device__ void storey(float2 *a, float *x, int sx) {
#pragma unroll
    for (int i = 0; i < n; i++)
        x[i*sx] = a[rev<n>(i)].y;
}
inline __device__ void storex4x4(float2 *a, float *x, int sx)
{
#pragma unroll
    for (int i = 0; i < 16; i++)
        x[i*sx] = a[rev4x4(i)].x;
}
inline __device__ void storey4x4(float2 *a, float *x, int sx)
{
#pragma unroll
    for (int i = 0; i < 16; i++)
        x[i*sx] = a[rev4x4(i)].y;
}

//
//  multiply by twiddle factors in bit-reversed order
//
template<int radix, Direction dir> inline __device__ void twiddle(float2 *a, int i, int n) {
#pragma unroll
    for (int j = 1; j < radix; j++)
        a[j] = a[j] * exp_i(((dir == FORWARD ? -1 : 1) * 2 * M_PI * rev<radix>(j)/n) * i);
}

template<Direction dir> inline __device__ void twiddle4x4(float2 *a, int i) {
    float2 w1 = exp_i(((dir == FORWARD ? -1 : 1) * 2*M_PI/32)*i);
    a[1]  = a[1]  * w1;
    a[5]  = a[5]  * w1;
    a[9]  = a[9]  * w1;
    a[13] = a[13] * w1;
    
    float2 w2 = exp_i(((dir == FORWARD ? -1 : 1) * 1*M_PI/32)*i);
    a[2]  = a[2]  * w2;
    a[6]  = a[6]  * w2;
    a[10] = a[10] * w2;
    a[14] = a[14] * w2;
    
    float2 w3 = exp_i(((dir == FORWARD ? -1 : 1) * 3*M_PI/32)*i);
    a[3]  = a[3]  * w3;
    a[7]  = a[7]  * w3;
    a[11] = a[11] * w3;
    a[15] = a[15] * w3;
}

//
//  multiply by twiddle factors in straight order
//
template<int radix, Direction dir> inline __device__ void twiddle_straight(float2 *a, int i, int n) {
#pragma unroll
    for (int j = 1; j < radix; j++)
        a[j] = a[j] * exp_i(((dir == FORWARD ? -1 : 1) * 2 * M_PI * j/n) * i);
}

//
//  transpose via shared memory, input is in bit-reversed layout
//
template<int n> inline __device__ void transpose(float2 *a, float *s, int ds, float *l, int dl, int sync = 0xf) {
  storex<n>(a, s, ds);  if(sync & 8) __syncthreads();
  loadx<n> (a, l, dl);  if(sync & 4) __syncthreads();
  storey<n>(a, s, ds);  if(sync & 2) __syncthreads();
  loady<n> (a, l, dl);  if(sync & 1) __syncthreads();
}

template<int n> inline __device__ void transpose(float2 *a, float *s, int ds, float *l, int *il, int sync = 0xf) {
  storex<n>(a, s, ds);  if(sync & 8) __syncthreads();
  loadx<n> (a, l, il);  if(sync & 4) __syncthreads();
  storey<n>(a, s, ds);  if(sync & 2) __syncthreads();
  loady<n> (a, l, il);  if(sync & 1) __syncthreads();
}

inline __device__ void transpose4x4(float2 *a, float *s, int ds, float *l, int dl, int sync = 0xf) {
  storex4x4(a, s, ds);  if(sync & 8) __syncthreads();
  loadx<16>(a, l, dl);  if(sync & 4) __syncthreads();
  storey4x4(a, s, ds);  if(sync & 2) __syncthreads();
  loady<16>(a, l, dl);  if(sync & 1) __syncthreads();
}
template<int n> inline __device__ void scalar(float2 *a, float f) {
#pragma unroll
    for (int i = 0; i < n; i++)
        a[i] = a[i] * f;
}

template<int n, Axis axis, Direction dir> struct FFTComputing {
  static __device__ void FFT(float2 *dst, float2 *src, int tid);
};

template<Axis axis, Direction dir> struct FFTComputing<256, axis, dir> {
  static __device__ void FFT(float2 *dst, float2 *src, int tid) {
    float2 a[16];
    if (axis == AXIS_X)
      load<16>(a, src, 16);
    else
      load<16>(a, src, 16 * 256);

    __shared__ float s[64 * 17];
    int hi = tid >> 4;
    int lo = tid & 15;
 
    FFT16<dir>(a);
    
    twiddle<16, dir>(a, lo, 256);
    transpose<16>(a, &s[hi * 17 * 16 + 17 * lo], 1, &s[hi * 17 * 16 + lo], 17, 0);
    
    FFT16<dir>(a);

    if (dir == INVERSE)
      scalar<16>(a, 1./256);

    if (axis == AXIS_X)
      store<16>(a, dst, 16);
    else
      store<16>(a, dst, 16 * 256);
  }
};

template<Axis axis, Direction dir> struct FFTComputing<512, axis, dir> {
  static __device__ void FFT(float2 *dst, float2 *src, int tid) {
    float2 a[8];
    
    if (axis == AXIS_X)
      load<8>(a, src, 64);
    else
      load<8>(a, src, 64 * 512);

    __shared__ float s[8 * 8 * 9];
    int hi = tid >> 3;
    int lo = tid & 7;

    FFT8<dir>(a);
  
    twiddle<8, dir>(a, tid, 512);
    transpose<8>(a, &s[hi * 8 + lo], 66, &s[lo * 66 + hi], 8);
  
    FFT8<dir>(a);
  
    twiddle<8, dir>(a, hi, 64);
    transpose<8>(a, &s[hi * 8 + lo], 8 * 9, &s[hi * 8 * 9 + lo], 8, 0xE);
    
    FFT8<dir>(a);

    if (dir == INVERSE)
      scalar<8>(a, 1./512);

    if (axis == AXIS_X)
      store<8>(a, dst, 64);
    else
      store<8>(a, dst, 64 * 512); 
  }
};

template<Axis axis, Direction dir> struct FFTComputing<1024, axis, dir> {
  static __device__ void FFT(float2 *dst, float2 *src, int tid) {
    float2 a[16];

    if (axis == AXIS_X)
      load<16>(a, src, 64);
    else
      load<16>(a, src, 64 * 1024);

    __shared__ float s[69 * 16];
    int hi4 = tid >> 4;
    int lo4 = tid & 15;
    int hi2 = tid >> 4;
    int mi2 = (tid >> 2) & 3;
    int lo2 = tid & 3;

    FFT16<dir>(a);
    
    twiddle<16, dir>(a, tid, 1024);
    int il[] = {0,1,2,3, 16,17,18,19, 32,33,34,35, 48,49,50,51};
    transpose<16>(a, &s[lo4 * 65 + hi4], 4, &s[lo4 * 65 + hi4 * 4], il);
    
    FFT4x4<dir>(a);

    twiddle4x4<dir>(a, lo4);
    transpose4x4(a, &s[hi2 * 17 + mi2 * 4 + lo2], 69, &s[mi2 * 69 * 4 + hi2 * 69 + lo2 * 17], 1, 0xE);
    
    FFT16<dir>(a);


    if (dir == INVERSE)
      scalar<16>(a, 1./1024);
    
    if (axis == AXIS_X)
      store<16>(a, dst, 64);
    else
      store<16>(a, dst, 64 * 1024);  
  }
};

template<int n, Axis axis, Direction dir> __global__ void FFT_device(float2 *dst, float2 *src) {
    int tid = threadIdx.x;
    int iblock = blockIdx.y * gridDim.x + blockIdx.x;
    int index = axis == AXIS_X ? iblock * n + tid : iblock + tid * n;
    src += index;
    dst += index;
    
    FFTComputing<n, axis, dir>::FFT(dst, src, tid);
}

template<int n, Axis axis, Direction dir> __host__ void FFT(float2 *data, int batch, cudaStream_t& stream) {
    FFT_device<n, axis, dir><<<grid2D(batch), 64, 0, stream>>>(data, data);
    CUDA_ERROR_CHECK(cudaGetLastError());
}
__host__ void Filter(float2 *work, int size) {
  unsigned int eight = size / 8;
  unsigned int eight7 = size - eight;
  CUDA_ERROR_CHECK(cudaMemset2D(work + eight, sizeof(float2) * size, 0, sizeof(float2) * (eight7 - eight), size));
  CUDA_ERROR_CHECK(cudaMemset2D(work + eight * size, sizeof(float2) * size, 0, sizeof(float2) * size, eight7 - eight));
}
/*
template<int n> __global__ void MyFilter_device(float2 *data) {
    int y = threadIdx.x;
    int x = blockIdx.x;
    int index = x * n + y;
    if (!((x < n / 8 || x >= n - n / 8) && (y < n / 8 || y >= n - n / 8))) {
      data[index].x = data[index].y = 0;
    }
}
__host__ void MyFilter(float2 *work, int size) {
  MyFilter_device<n><<<grid2D(size), size, 0, stream>>>(data);
}
*/
__host__ float filterImage(float2* image, int size_x, int size_y) {
  #define PNUM  8
  #define BLOCK (SIZE * SIZE / PNUM)
 
  // check that the sizes match up
  assert(size_x == SIZEX);
  assert(size_y == SIZEY);

  int size = size_x;
  int matSize = size_x * size_y;

  // These variables are for timing purposes
  float transferDown = 0, transferUp = 0, execution = 0;
  cudaEvent_t start,stop;

  CUDA_ERROR_CHECK(cudaEventCreate(&start));
  CUDA_ERROR_CHECK(cudaEventCreate(&stop));

  // Create a stream and initialize it
  cudaStream_t filterStream[PNUM];
  for (int i = 0; i < PNUM; i++)
    CUDA_ERROR_CHECK(cudaStreamCreate(&filterStream[i]));

  // Alloc space on the device
  float2 *data;
  CUDA_ERROR_CHECK(cudaMalloc((void**)&data, matSize * sizeof(float2)));

  // Start timing for transfer down
  //CUDA_ERROR_CHECK(cudaEventRecord(start,filterStream));
  
  // Here is where we copy matrices down to the device 
 
  // Stop timing for transfer down
  //CUDA_ERROR_CHECK(cudaEventRecord(stop,filterStream));
  //CUDA_ERROR_CHECK(cudaEventSynchronize(stop));
  //CUDA_ERROR_CHECK(cudaEventElapsedTime(&transferDown,start,stop));

  // Start timing for the execution
  CUDA_ERROR_CHECK(cudaEventRecord(start, filterStream[PNUM - 1]));

  //----------------------------------------------------------------
  // TODO: YOU SHOULD PLACE ALL YOUR KERNEL EXECUTIONS
  //        HERE BETWEEN THE CALLS FOR STARTING AND
  //        FINISHING TIMING FOR THE EXECUTION PHASE
  // BEGIN ADD KERNEL CALLS
  //----------------------------------------------------------------

  // This is an example kernel call, you should feel free to create
  // as many kernel calls as you feel are needed for your program
  // Each of the parameters are as follows:
  //    1. Number of thread blocks, can be either int or dim3 (see CUDA manual)
  //    2. Number of threads per thread block, can be either int or dim3 (see CUDA manual)
  //    3. Always should be '0' unless you read the CUDA manual and learn about dynamically allocating shared memory
  //    4. Stream to execute kernel on, should always be 'filterStream'
  //
  // Also note that you pass the pointers to the device memory to the kernel call
  
  //http://www.pgroup.com/lit/articles/insider/v3n1a4.htm
  unsigned int eight = size / 8;
  unsigned int eight7 = size - eight;
  for (int i = 0; i < PNUM; i++) {
    CUDA_ERROR_CHECK(cudaMemcpyAsync(data + BLOCK * i, image + BLOCK * i, matSize * sizeof(float2) / PNUM, cudaMemcpyHostToDevice, filterStream[i]));
    FFT<SIZE, AXIS_X, FORWARD>(data + BLOCK * i, size / PNUM, filterStream[i]);
    cudaMemsetAsync(data + eight + BLOCK * i, 0, sizeof(float2) * (eight7 - eight), filterStream[i]);
  }
  FFT<SIZE, AXIS_Y, FORWARD>(data, size / 8, filterStream[PNUM - 1]);
  FFT<SIZE, AXIS_Y, FORWARD>(data + (size - size / 8), size / 8, filterStream[PNUM - 2]);
  Filter(data, size);
  FFT<SIZE, AXIS_Y, INVERSE>(data, size / 8, filterStream[PNUM - 1]);
  FFT<SIZE, AXIS_Y, INVERSE>(data + (size - size / 8), size / 8, filterStream[PNUM - 2]);
  for (int i = PNUM - 1; i >= 0; i--) {
    FFT<SIZE, AXIS_X, INVERSE>(data + BLOCK * i, size / PNUM, filterStream[i]);
    CUDA_ERROR_CHECK(cudaMemcpyAsync(image + BLOCK * i, data + BLOCK * i, sizeof(float2) * matSize / PNUM, cudaMemcpyDeviceToHost, filterStream[i]));
  }
  //CUDA_ERROR_CHECK(cudaMemcpyAsync(data, image, sizeof(float2) * matSize, cudaMemcpyHostToDevice));


  //FFT<SIZE, AXIS_X, FORWARD>(data, size / 8);
  //FFT<SIZE, AXIS_X, FORWARD>(data + (size - size / 8) * size, size / 8);
  //FFT<SIZE, AXIS_X, INVERSE>(data, size / 8);
  //FFT<SIZE, AXIS_X, INVERSE>(data + (size - size / 8) * size, size / 8); 
  //FFT<SIZE, AXIS_Y, INVERSE>(data, size);
  //for (int i = 0; i < PNUM; i++) 
  //   CUDA_ERROR_CHECK(cudaMemcpyAsync(image + BLOCK * i, data + BLOCK * i, matSize * sizeof(float2) / PNUM, cudaMemcpyDeviceToHost));
  //CUDA_ERROR_CHECK(cudaMemcpyAsync(image, data, sizeof(float2) * matSize / 2, cudaMemcpyDeviceToHost));
  //CUDA_ERROR_CHECK(cudaMemcpyAsync(image + matSize / 2, data + matSize / 2, sizeof(float2) * matSize / 2, cudaMemcpyDeviceToHost));


  //---------------------------------------------------------------- 
  // END ADD KERNEL CALLS
  //----------------------------------------------------------------

  // Finish timimg for the execution 
  CUDA_ERROR_CHECK(cudaEventRecord(stop,filterStream[PNUM - 1]));
  CUDA_ERROR_CHECK(cudaEventSynchronize(stop));
  CUDA_ERROR_CHECK(cudaEventElapsedTime(&execution,start,stop));

  // Start timing for the transfer up
  //CUDA_ERROR_CHECK(cudaEventRecord(start,filterStream));

  // Here is where we copy matrices back from the device 

  // Finish timing for transfer up
  //CUDA_ERROR_CHECK(cudaEventRecord(stop,filterStream));
  //CUDA_ERROR_CHECK(cudaEventSynchronize(stop));
  //CUDA_ERROR_CHECK(cudaEventElapsedTime(&transferUp,start,stop));
  
  // Synchronize the stream
  for (int i = 0; i < PNUM; i++) {
    CUDA_ERROR_CHECK(cudaStreamSynchronize(filterStream[i]));
    // Destroy the stream
    CUDA_ERROR_CHECK(cudaStreamDestroy(filterStream[i]));
  }
  // Destroy the events
  CUDA_ERROR_CHECK(cudaEventDestroy(start));
  CUDA_ERROR_CHECK(cudaEventDestroy(stop));

  // Free the memory
  CUDA_ERROR_CHECK(cudaFree(data));

  // Dump some usage statistics
  printf("CUDA IMPLEMENTATION STATISTICS:\n");
  printf("  Host to Device Transfer Time: %f ms\n", transferDown);
  printf("  Kernel(s) Execution Time: %f ms\n", execution);
  printf("  Device to Host Transfer Time: %f ms\n", transferUp);
  float totalTime = transferDown + execution + transferUp;
  printf("  Total CUDA Execution Time: %f ms\n\n", totalTime);
  // Return the total time to transfer and execute
  return totalTime;
}
#else
#include <cufft.h>

__host__ float filterImage(float2 *image, int size_x, int size_y)
{
  // These variables are for timing purposes
  float transferDown = 0, transferUp = 0, execution = 0;
  cudaEvent_t start,stop;
  CUDA_ERROR_CHECK(cudaEventCreate(&start));
  CUDA_ERROR_CHECK(cudaEventCreate(&stop));

  // Create a stream and initialize it
  cudaStream_t filterStream;
  CUDA_ERROR_CHECK(cudaStreamCreate(&filterStream));

  // Alloc space on the device

  unsigned int eight = size_y / 8;
  unsigned int eight7 = size_y - eight;
  cufftHandle plan;
  cufftComplex *data;
  cufftPlan2d(&plan, size_x, size_y, CUFFT_C2C);
  CUDA_ERROR_CHECK(cudaMalloc((void**) &data, sizeof(cufftComplex) * size_x * size_y));

  // Start timing for transfer down
  CUDA_ERROR_CHECK(cudaEventRecord(start,filterStream));
  
  // Here is where we copy matrices down to the device 

  // Stop timing for transfer down
  CUDA_ERROR_CHECK(cudaEventRecord(stop,filterStream));
  CUDA_ERROR_CHECK(cudaEventSynchronize(stop));
  CUDA_ERROR_CHECK(cudaEventElapsedTime(&transferDown,start,stop));

  // Start timing for the execution
  CUDA_ERROR_CHECK(cudaEventRecord(start,filterStream));
 
  //----------------------------------------------------------------
  // TODO:  YOU SHOULD PLACE ALL YOUR KERNEL EXECUTIONS
  //        HERE BETWEEN THE CALLS FOR STARTING AND
  //        FINISHING TIMING FOR THE EXECUTION PHASE
  //
  // BEGIN ADD KERNEL CALLS
  //----------------------------------------------------------------

  // This is an example kernel call, you should feel free to create
  // as many kernel calls as you feel are needed for your program
  // Each of the parameters are as follows:
  //    1. Number of thread blocks, can be either int or dim3 (see CUDA manual)
  //    2. Number of threads per thread block, can be either int or dim3 (see CUDA manual)
  //    3. Always should be '0' unless you read the CUDA manual and learn about dynamically allocating shared memory
  //    4. Stream to execute kernel on, should always be 'filterStream'
  //
  // Also note that you pass the pointers to the device memory to the kernel call
  
  CUDA_ERROR_CHECK(cudaMemcpyAsync(data, image, sizeof(cufftComplex) * size_x * size_y, cudaMemcpyHostToDevice));
  cufftExecC2C(plan, data, data, CUFFT_FORWARD);
  //Filter
  CUDA_ERROR_CHECK(cudaMemset2D(data + eight, sizeof(cufftComplex) * size_y, 0, sizeof(cufftComplex) * (eight7 - eight), size_x));
  CUDA_ERROR_CHECK(cudaMemset2D(data + eight * size_y,  sizeof(cufftComplex) * size_y, 0, sizeof(cufftComplex) * size_y, eight7 - eight));
  cufftExecC2C(plan, data, data, CUFFT_INVERSE);
  CUDA_ERROR_CHECK(cudaMemcpyAsync(image, data, sizeof(cufftComplex) * size_x * size_y, cudaMemcpyDeviceToHost));

  //---------------------------------------------------------------- 
  // END ADD KERNEL CALLS
  //----------------------------------------------------------------

  // Finish timimg for the execution 
  CUDA_ERROR_CHECK(cudaEventRecord(stop,filterStream));
  CUDA_ERROR_CHECK(cudaEventSynchronize(stop));
  CUDA_ERROR_CHECK(cudaEventElapsedTime(&execution,start,stop));

  // Start timing for the transfer up
  CUDA_ERROR_CHECK(cudaEventRecord(start,filterStream));

  // Here is where we copy matrices back from the device 
  for (int i = 0; i < size_x * size_y; i++) {
    image[i].x /= (size_x * size_y);
    image[i].y /= (size_x * size_y);
  }

  // Finish timing for transfer up
  CUDA_ERROR_CHECK(cudaEventRecord(stop,filterStream));
  CUDA_ERROR_CHECK(cudaEventSynchronize(stop));
  CUDA_ERROR_CHECK(cudaEventElapsedTime(&transferUp,start,stop));

  // Synchronize the stream
  CUDA_ERROR_CHECK(cudaStreamSynchronize(filterStream));
  // Destroy the stream
  CUDA_ERROR_CHECK(cudaStreamDestroy(filterStream));
  // Destroy the events
  CUDA_ERROR_CHECK(cudaEventDestroy(start));
  CUDA_ERROR_CHECK(cudaEventDestroy(stop));

  // Free the memory
  CUDA_ERROR_CHECK(cudaFree(data));
  cufftDestroy(plan);

  // Dump some usage statistics
  printf("CUDA IMPLEMENTATION STATISTICS:\n");
  printf("  Host to Device Transfer Time: %f ms\n", transferDown);
  printf("  Kernel(s) Execution Time: %f ms\n", execution);
  printf("  Device to Host Transfer Time: %f ms\n", transferUp);
  float totalTime = transferDown + execution + transferUp;
  printf("  Total CUDA Execution Time: %f ms\n\n", totalTime);
  // Return the total time to transfer and execute
  return totalTime;
}


#endif

