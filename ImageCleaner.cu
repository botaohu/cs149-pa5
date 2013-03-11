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

#define _USE_MATH_DEFINES
#include <math.h>

inline dim3 grid2D( int nblocks )
{
    int slices = 1;
    while( nblocks/slices > 65535 ) 
        slices *= 2;
    return dim3( nblocks/slices, slices );
}

inline __device__ float2 operator*( float2 a, float2 b ) { return make_float2( a.x*b.x-a.y*b.y, a.x*b.y+a.y*b.x ); }
inline __device__ float2 operator*( float2 a, float  b ) { return make_float2( b*a.x, b*a.y ); }
inline __device__ float2 operator+( float2 a, float2 b ) { return make_float2( a.x + b.x, a.y + b.y ); }
inline __device__ float2 operator-( float2 a, float2 b ) { return make_float2( a.x - b.x, a.y - b.y ); }

#define COS_PI_8  0.923879533f
#define SIN_PI_8  0.382683432f
#define exp_1_16  make_float2(  COS_PI_8, -SIN_PI_8 )
#define exp_3_16  make_float2(  SIN_PI_8, -COS_PI_8 )
#define exp_5_16  make_float2( -SIN_PI_8, -COS_PI_8 )
#define exp_7_16  make_float2( -COS_PI_8, -SIN_PI_8 )
#define exp_9_16  make_float2( -COS_PI_8,  SIN_PI_8 )
#define exp_1_8   make_float2(  M_SQRT1_2, -M_SQRT1_2 )//requires post-multiply by 1/sqrt(2)
#define exp_1_4   make_float2(  0, -1 )
#define exp_3_8   make_float2( -M_SQRT1_2, -M_SQRT1_2 )//requires post-multiply by 1/sqrt(2)

#define iexp_1_16  make_float2(  COS_PI_8,  SIN_PI_8 )
#define iexp_3_16  make_float2(  SIN_PI_8,  COS_PI_8 )
#define iexp_5_16  make_float2( -SIN_PI_8,  COS_PI_8 )
#define iexp_7_16  make_float2( -COS_PI_8,  SIN_PI_8 )
#define iexp_9_16  make_float2( -COS_PI_8, -SIN_PI_8 )
#define iexp_1_8   make_float2(  M_SQRT1_2, M_SQRT1_2 )//requires post-multiply by 1/sqrt(2)
#define iexp_1_4   make_float2(  0, 1 )
#define iexp_3_8   make_float2( -M_SQRT1_2, M_SQRT1_2 )//requires post-multiply by 1/sqrt(2)

inline __device__ float2 exp_i( float phi )
{
    return make_float2( __cosf(phi), __sinf(phi) );
}


template<int radix> inline __device__ int rev( int bits );

template<> inline __device__ int rev<2>( int bits )
{
    return bits;
}

template<> inline __device__ int rev<4>( int bits )
{
    int reversed[] = {0,2,1,3};
    return reversed[bits];
}

template<> inline __device__ int rev<8>( int bits )
{
    int reversed[] = {0,4,2,6,1,5,3,7};
    return reversed[bits];
}

template<> inline __device__ int rev<16>( int bits )
{
    int reversed[] = {0,8,4,12,2,10,6,14,1,9,5,13,3,11,7,15};
    return reversed[bits];
}

inline __device__ int rev4x4( int bits )
{
    int reversed[] = {0,2,1,3, 4,6,5,7, 8,10,9,11, 12,14,13,15};
    return reversed[bits];
}
#define IFFT2 FFT2
inline __device__ void FFT2( float2 &a0, float2 &a1 )
{ 
    float2 c0 = a0;
    a0 = c0 + a1; 
    a1 = c0 - a1;
}

inline __device__ void FFT4( float2 &a0, float2 &a1, float2 &a2, float2 &a3 )
{
    FFT2( a0, a2 );
    FFT2( a1, a3 );
    a3 = a3 * exp_1_4;
    FFT2( a0, a1 );
    FFT2( a2, a3 );
}

inline __device__ void IFFT4( float2 &a0, float2 &a1, float2 &a2, float2 &a3 )
{
    IFFT2( a0, a2 );
    IFFT2( a1, a3 );
    a3 = a3 * iexp_1_4;
    IFFT2( a0, a1 );
    IFFT2( a2, a3 );
}

inline __device__ void FFT2( float2 *a ) { FFT2( a[0], a[1] ); }
inline __device__ void FFT4( float2 *a ) { FFT4( a[0], a[1], a[2], a[3] ); }
inline __device__ void IFFT4( float2 *a ) { IFFT4( a[0], a[1], a[2], a[3] ); }

inline __device__ void FFT8( float2 *a )
{
    FFT2( a[0], a[4] );
    FFT2( a[1], a[5] );
    FFT2( a[2], a[6] );
    FFT2( a[3], a[7] );
    
    a[5] = ( a[5] * exp_1_8 ) ;
    a[6] =   a[6] * exp_1_4;
    a[7] = ( a[7] * exp_3_8 ) ;

    FFT4( a[0], a[1], a[2], a[3] );
    FFT4( a[4], a[5], a[6], a[7] );
}

inline __device__ void IFFT8( float2 *a )
{
    IFFT2( a[0], a[4] );
    IFFT2( a[1], a[5] );
    IFFT2( a[2], a[6] );
    IFFT2( a[3], a[7] );
    
    a[5] = ( a[5] * iexp_1_8 ) ;
    a[6] =   a[6] * iexp_1_4;
    a[7] = ( a[7] * iexp_3_8 ) ;

    IFFT4( a[0], a[1], a[2], a[3] );
    IFFT4( a[4], a[5], a[6], a[7] );
}

inline __device__ void FFT16( float2 *a )
{
    FFT4( a[0], a[4], a[8], a[12] );
    FFT4( a[1], a[5], a[9], a[13] );
    FFT4( a[2], a[6], a[10], a[14] );
    FFT4( a[3], a[7], a[11], a[15] );

    a[5]  = (a[5]  * exp_1_8 ) ;
    a[6]  =  a[6]  * exp_1_4;
    a[7]  = (a[7]  * exp_3_8 ) ;
    a[9]  =  a[9]  * exp_1_16;
    a[10] = (a[10] * exp_1_8 ) ;
    a[11] =  a[11] * exp_3_16;
    a[13] =  a[13] * exp_3_16;
    a[14] = (a[14] * exp_3_8 ) ;
    a[15] =  a[15] * exp_9_16;

    FFT4( a[0],  a[1],  a[2],  a[3] );
    FFT4( a[4],  a[5],  a[6],  a[7] );
    FFT4( a[8],  a[9],  a[10], a[11] );
    FFT4( a[12], a[13], a[14], a[15] );
}

inline __device__ void IFFT16( float2 *a )
{
    IFFT4( a[0], a[4], a[8], a[12] );
    IFFT4( a[1], a[5], a[9], a[13] );
    IFFT4( a[2], a[6], a[10], a[14] );
    IFFT4( a[3], a[7], a[11], a[15] );

    a[5]  = (a[5]  * iexp_1_8 ) ;
    a[6]  =  a[6]  * iexp_1_4;
    a[7]  = (a[7]  * iexp_3_8 ) ;
    a[9]  =  a[9]  * iexp_1_16;
    a[10] = (a[10] * iexp_1_8 ) ;
    a[11] =  a[11] * iexp_3_16;
    a[13] =  a[13] * iexp_3_16;
    a[14] = (a[14] * iexp_3_8 ) ;
    a[15] =  a[15] * iexp_9_16;

    IFFT4( a[0],  a[1],  a[2],  a[3] );
    IFFT4( a[4],  a[5],  a[6],  a[7] );
    IFFT4( a[8],  a[9],  a[10], a[11] );
    IFFT4( a[12], a[13], a[14], a[15] );
}

inline __device__ void FFT4x4( float2 *a )
{
    FFT4( a[0],  a[1],  a[2],  a[3] );
    FFT4( a[4],  a[5],  a[6],  a[7] );
    FFT4( a[8],  a[9],  a[10], a[11] );
    FFT4( a[12], a[13], a[14], a[15] );
}

inline __device__ void IFFT4x4( float2 *a )
{
    IFFT2( a[0], a[2] );
    IFFT2( a[1], a[3] );
    IFFT2( a[4], a[6] );
    IFFT2( a[5], a[7] );
    IFFT2( a[8], a[10] );
    IFFT2( a[9], a[11] );
    IFFT2( a[12], a[14] );
    IFFT2( a[13], a[15] );

    a[3] = a[3] * iexp_1_4;
    a[7] = a[7] * iexp_1_4;
    a[11] = a[11] * iexp_1_4;
    a[15] = a[15] * iexp_1_4;

    IFFT2( a[0], a[1] );
    IFFT2( a[2], a[3] );
    IFFT2( a[4], a[5] );
    IFFT2( a[6], a[7] );
    IFFT2( a[8], a[9] );
    IFFT2( a[10], a[11] );
    IFFT2( a[12], a[13] );
    IFFT2( a[14], a[15] );
}

//
//  loads
//
template<int n> inline __device__ void load( float2 *a, float2 *x, int sx )
{
    for( int i = 0; i < n; i++ )
        a[i] = x[i*sx];
}
template<int n> inline __device__ void loadx( float2 *a, float *x, int sx )
{
    for( int i = 0; i < n; i++ )
        a[i].x = x[i*sx];
}
template<int n> inline __device__ void loady( float2 *a, float *x, int sx )
{
    for( int i = 0; i < n; i++ )
        a[i].y = x[i*sx];
}
template<int n> inline __device__ void loadx( float2 *a, float *x, int *ind )
{
    for( int i = 0; i < n; i++ )
        a[i].x = x[ind[i]];
}
template<int n> inline __device__ void loady( float2 *a, float *x, int *ind )
{
    for( int i = 0; i < n; i++ )
        a[i].y = x[ind[i]];
}

//
//  stores, input is in bit reversed order
//
template<int n> inline __device__ void store( float2 *a, float2 *x, int sx )
{
#pragma unroll
    for( int i = 0; i < n; i++ )
        x[i*sx] = a[rev<n>(i)];
}
template<int n> inline __device__ void storex( float2 *a, float *x, int sx )
{
#pragma unroll
    for( int i = 0; i < n; i++ )
        x[i*sx] = a[rev<n>(i)].x;
}
template<int n> inline __device__ void storey( float2 *a, float *x, int sx )
{
#pragma unroll
    for( int i = 0; i < n; i++ )
        x[i*sx] = a[rev<n>(i)].y;
}
inline __device__ void storex4x4( float2 *a, float *x, int sx )
{
#pragma unroll
    for( int i = 0; i < 16; i++ )
        x[i*sx] = a[rev4x4(i)].x;
}
inline __device__ void storey4x4( float2 *a, float *x, int sx )
{
#pragma unroll
    for( int i = 0; i < 16; i++ )
        x[i*sx] = a[rev4x4(i)].y;
}

//
//  multiply by twiddle factors in bit-reversed order
//
template<int radix>inline __device__ void twiddle( float2 *a, int i, int n )
{
#pragma unroll
    for( int j = 1; j < radix; j++ )
        a[j] = a[j] * exp_i((-2*M_PI*rev<radix>(j)/n)*i);
}

template<int radix>inline __device__ void itwiddle( float2 *a, int i, int n )
{
#pragma unroll
    for( int j = 1; j < radix; j++ )
        a[j] = a[j] * exp_i((2*M_PI*rev<radix>(j)/n)*i);
}

inline __device__ void twiddle4x4( float2 *a, int i )
{
    float2 w1 = exp_i((-2*M_PI/32)*i);
    a[1]  = a[1]  * w1;
    a[5]  = a[5]  * w1;
    a[9]  = a[9]  * w1;
    a[13] = a[13] * w1;
    
    float2 w2 = exp_i((-1*M_PI/32)*i);
    a[2]  = a[2]  * w2;
    a[6]  = a[6]  * w2;
    a[10] = a[10] * w2;
    a[14] = a[14] * w2;
    
    float2 w3 = exp_i((-3*M_PI/32)*i);
    a[3]  = a[3]  * w3;
    a[7]  = a[7]  * w3;
    a[11] = a[11] * w3;
    a[15] = a[15] * w3;
}

inline __device__ void itwiddle4x4( float2 *a, int i )
{
    float2 w1 = exp_i((2*M_PI/32)*i);
    a[1]  = a[1]  * w1;
    a[5]  = a[5]  * w1;
    a[9]  = a[9]  * w1;
    a[13] = a[13] * w1;
    
    float2 w2 = exp_i((1*M_PI/32)*i);
    a[2]  = a[2]  * w2;
    a[6]  = a[6]  * w2;
    a[10] = a[10] * w2;
    a[14] = a[14] * w2;
    
    float2 w3 = exp_i((3*M_PI/32)*i);
    a[3]  = a[3]  * w3;
    a[7]  = a[7]  * w3;
    a[11] = a[11] * w3;
    a[15] = a[15] * w3;
}

//
//  multiply by twiddle factors in straight order
//
template<int radix>inline __device__ void twiddle_straight( float2 *a, int i, int n )
{
#pragma unroll
    for( int j = 1; j < radix; j++ )
        a[j] = a[j] * exp_i((-2*M_PI*j/n)*i);
}

template<int radix>inline __device__ void itwiddle_straight( float2 *a, int i, int n )
{
#pragma unroll
    for( int j = 1; j < radix; j++ )
        a[j] = a[j] * exp_i((2*M_PI*j/n)*i);
}

//
//  transpose via shared memory, input is in bit-reversed layout
//
template<int n> inline __device__ void transpose( float2 *a, float *s, int ds, float *l, int dl, int sync = 0xf )
{
    storex<n>( a, s, ds );  if( sync&8 ) __syncthreads();
    loadx<n> ( a, l, dl );  if( sync&4 ) __syncthreads();
    storey<n>( a, s, ds );  if( sync&2 ) __syncthreads();
    loady<n> ( a, l, dl );  if( sync&1 ) __syncthreads();
}

template<int n> inline __device__ void transpose( float2 *a, float *s, int ds, float *l, int *il, int sync = 0xf )
{
    storex<n>( a, s, ds );  if( sync&8 ) __syncthreads();
    loadx<n> ( a, l, il );  if( sync&4 ) __syncthreads();
    storey<n>( a, s, ds );  if( sync&2 ) __syncthreads();
    loady<n> ( a, l, il );  if( sync&1 ) __syncthreads();
}

inline __device__ void transpose4x4( float2 *a, float *s, int ds, float *l, int dl, int sync = 0xf )
{
    storex4x4( a, s, ds );  if( sync&8 ) __syncthreads();
    loadx<16>( a, l, dl );  if( sync&4 ) __syncthreads();
    storey4x4( a, s, ds );  if( sync&2 ) __syncthreads();
    loady<16>( a, l, dl );  if( sync&1 ) __syncthreads();
}

__global__ void FFT1024_device( float2 *dst, float2 *src )
{ 
    int tid = threadIdx.x;
    
    int iblock = blockIdx.y * gridDim.x + blockIdx.x;
    int index = iblock * 1024 + tid;
    src += index;
    dst += index;
    
    int hi4 = tid>>4;
    int lo4 = tid&15;
    int hi2 = tid>>4;
    int mi2 = (tid>>2)&3;
    int lo2 = tid&3;

    float2 a[16];
    __shared__ float smem[69*16];
    
    load<16>( a, src, 64 );

    FFT16( a );
    
    twiddle<16>( a, tid, 1024 );
    int il[] = {0,1,2,3, 16,17,18,19, 32,33,34,35, 48,49,50,51};
    transpose<16>( a, &smem[lo4*65+hi4], 4, &smem[lo4*65+hi4*4], il );
    
    FFT4x4( a );

    twiddle4x4( a, lo4 );
    transpose4x4( a, &smem[hi2*17 + mi2*4 + lo2], 69, &smem[mi2*69*4 + hi2*69 + lo2*17 ], 1, 0xE );
    
    FFT16( a );

    store<16>( a, dst, 64 );
}   
    
void FFT1024( float2 *work, int batch )
{ 
    FFT1024_device<<< grid2D(batch), 64 >>>( work, work );
} 

__global__ void IFFT1024_device( float2 *dst, float2 *src )
{ 
    int tid = threadIdx.x;
    
    int iblock = blockIdx.y * gridDim.x + blockIdx.x;
    int index = iblock * 1024 + tid;
    src += index;
    dst += index;
    
    int hi4 = tid>>4;
    int lo4 = tid&15;
    int hi2 = tid>>4;
    int mi2 = (tid>>2)&3;
    int lo2 = tid&3;

    float2 a[16];
    __shared__ float smem[69*16];
    
    load<16>( a, src, 64 );

    IFFT16( a );
    
    itwiddle<16>( a, tid, 1024 );
    int il[] = {0,1,2,3, 16,17,18,19, 32,33,34,35, 48,49,50,51};
    transpose<16>( a, &smem[lo4*65+hi4], 4, &smem[lo4*65+hi4*4], il );
    
    IFFT4x4( a );

    itwiddle4x4( a, lo4 );
    transpose4x4( a, &smem[hi2*17 + mi2*4 + lo2], 69, &smem[mi2*69*4 + hi2*69 + lo2*17 ], 1, 0xE );
    
    IFFT16( a );

    store<16>( a, dst, 64 );
}   
    
void IFFT1024( float2 *work, int batch )
{ 
    IFFT1024_device<<< grid2D(batch), 64 >>>( work, work );
} 


__global__ void FFT512_device( float2 *work )
{ 
    int tid = threadIdx.x;
    int hi = tid>>3;
    int lo = tid&7;
    
    work += (blockIdx.y * gridDim.x + blockIdx.x) * 512 + tid;
  
    float2 a[8];
    __shared__ float smem[8*8*9];
    
    load<8>( a, work, 64 );

    FFT8( a );
  
    twiddle<8>( a, tid, 512 );
    transpose<8>( a, &smem[hi*8+lo], 66, &smem[lo*66+hi], 8 );
  
    FFT8( a );
  
    twiddle<8>( a, hi, 64);
    transpose<8>( a, &smem[hi*8+lo], 8*9, &smem[hi*8*9+lo], 8, 0xE );
    
    FFT8( a );

    store<8>( a, work, 64 );
} 
    
void FFT512( float2 *work, int batch )
{ 
    FFT512_device<<< grid2D(batch), 64 >>>( work );
} 

__global__ void IFFT512_device( float2 *work )
{ 
    int tid = threadIdx.x;
    int hi = tid>>3;
    int lo = tid&7;
    
    work += (blockIdx.y * gridDim.x + blockIdx.x) * 512 + tid;
  
    float2 a[8];
    __shared__ float smem[8*8*9];
    
    load<8>( a, work, 64 );

    IFFT8( a );
  
    itwiddle<8>( a, tid, 512 );
    transpose<8>( a, &smem[hi*8+lo], 66, &smem[lo*66+hi], 8 );
  
    IFFT8( a );
  
    itwiddle<8>( a, hi, 64);
    transpose<8>( a, &smem[hi*8+lo], 8*9, &smem[hi*8*9+lo], 8, 0xE );
    
    IFFT8( a );

    store<8>( a, work, 64 );
} 

void IFFT512( float2 *work, int batch )
{ 
    IFFT512_device<<< grid2D(batch), 64 >>>( work );
} 

__host__ void mainKernel(float2 *data, int size) {
  if (size == 512) {
    FFT512(data, size);
    IFFT512(data, size);
  } else {
    FFT1024(data, size);
    IFFT1024(data, size);
  }
}
__host__ float filterImage(float *real_image, float *imag_image, int size_x, int size_y)
{
  // check that the sizes match up
  assert(size_x == SIZEX);
  assert(size_y == SIZEY);

  int matSize = size_x * size_y;

  // These variables are for timing purposes
  float transferDown = 0, transferUp = 0, execution = 0;
  cudaEvent_t start,stop;

  CUDA_ERROR_CHECK(cudaEventCreate(&start));
  CUDA_ERROR_CHECK(cudaEventCreate(&stop));

  // Create a stream and initialize it
  cudaStream_t filterStream;
  CUDA_ERROR_CHECK(cudaStreamCreate(&filterStream));

  // Alloc space on the device
  float2 *data;
  CUDA_ERROR_CHECK(cudaMalloc((void**)&data, matSize * sizeof(float2)));

  // Start timing for transfer down
  CUDA_ERROR_CHECK(cudaEventRecord(start,filterStream));
  
  // Here is where we copy matrices down to the device 
  float2 *dataLocal = new float2[matSize];
  for (int i = 0; i < matSize; i++) {
    dataLocal[i].x = real_image[i];
    dataLocal[i].y = imag_image[i];
  }
  CUDA_ERROR_CHECK(cudaMemcpy(data, dataLocal, matSize * sizeof(float2), cudaMemcpyHostToDevice));
  
  // Stop timing for transfer down
  CUDA_ERROR_CHECK(cudaEventRecord(stop,filterStream));
  CUDA_ERROR_CHECK(cudaEventSynchronize(stop));
  CUDA_ERROR_CHECK(cudaEventElapsedTime(&transferDown,start,stop));

  // Start timing for the execution
  CUDA_ERROR_CHECK(cudaEventRecord(start,filterStream));

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
  
  //precompute the bit reversal.
 

  mainKernel(data, SIZE);
  CUDA_ERROR_CHECK(cudaGetLastError());
  
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
  CUDA_ERROR_CHECK(cudaMemcpy(dataLocal, data, sizeof(float2) * matSize, cudaMemcpyDeviceToHost));
  for (int i = 0; i < matSize; i++) {
    real_image[i] =  dataLocal[i].x / matSize;
    imag_image[i] = dataLocal[i].y / matSize;
  }
  delete [] dataLocal;

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

__host__ float filterImage(float *real_image, float *imag_image, int size_x, int size_y)
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
  cufftComplex *dataLocal = new cufftComplex[size_x * size_y];
  for (int i = 0; i < size_x * size_y; i++) {
    dataLocal[i].x = real_image[i];
    dataLocal[i].y = imag_image[i];
  }
  CUDA_ERROR_CHECK(cudaMemcpy(data, dataLocal, sizeof(cufftComplex) * size_x * size_y, cudaMemcpyHostToDevice));


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
  
  cufftExecC2C(plan, data, data, CUFFT_FORWARD);
  //Filter
  CUDA_ERROR_CHECK(cudaMemset2D(data + eight, sizeof(cufftComplex) * size_y, 0, sizeof(cufftComplex) * (eight7 - eight), size_x));
  CUDA_ERROR_CHECK(cudaMemset2D(data + eight * size_y,  sizeof(cufftComplex) * size_y, 0, sizeof(cufftComplex) * size_y, eight7 - eight));
  cufftExecC2C(plan, data, data, CUFFT_INVERSE);

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
  CUDA_ERROR_CHECK(cudaMemcpy(dataLocal, data, sizeof(cufftComplex) * size_x * size_y, cudaMemcpyDeviceToHost));
  for (int i = 0; i < size_x * size_y; i++) {
    real_image[i] =  dataLocal[i].x / (size_x * size_y);
    imag_image[i] = dataLocal[i].y / (size_x * size_y);
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

