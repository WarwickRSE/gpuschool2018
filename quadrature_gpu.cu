// -*- mode: C -*-

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>  // clock() and clock_t
#include <cuda.h>

// Global data which defines f(x,y,z)

// Number of points P
#define P 500      

// Array of points at which Gaussians are located. This
// is now a static array in constant memory
//double *points;
__constant__ double points[3*P];
     
// Gaussian height parameter A
__constant__ double A = 0.1;

// Gaussian width parameter w
__constant__ double w = 0.2;

// This function evaluates f(x,y,z). It is only
// visible to the device and cannot be called
// directly by the host.
__device__ double func(double x, double y, double z) {

  double tot,arg;
  int i;
  
  tot = 0.0;
  for (i=0;i<P;i++) {
    arg  = 0.0;
    arg += (x-points[3*i+0])*(x-points[3*i+0]);
    arg += (y-points[3*i+1])*(y-points[3*i+1]);
    arg += (z-points[3*i+2])*(z-points[3*i+2]);
    tot += A*expf(-w*arg);
  }

  return tot;

}

// This is our kernel. for a point on our two-dimension grid
// integrate over the function in the third dimension.
// kernels must be global, visible to the host and device.
__global__ void integrate1D(double Min, double Max, int Ngrid, double *g_dev ) {

  int i,j,k;
  double delta,x,y,z,p_old,p_new,g_loc;

  // Which element on the 2D grid does this instance 
  // of the kernel need to compute?
  i = blockIdx.x*blockDim.x + threadIdx.x;
  j = blockIdx.y*blockDim.y + threadIdx.y;

  if ( (i<Ngrid) && (j <Ngrid) ) {

    // Compute grid spacing and current x and y value
    delta = (Max-Min)/(double)(Ngrid-1);
    x     = Min+(double)i*delta;
    y     = Min+(double)j*delta;
    
    // Integrate along z using trapezoidal rule
    g_loc = 0.0;
    z     = Min;
    p_old = expf(func(x,y,z));
    for (k=1;k<Ngrid;k++) {
      z += delta;
      p_new  = expf(func(x,y,z));
      g_loc += delta*0.5f*(p_old+p_new);
      p_old  = p_new;
    }
    
    // Store at the appropriate location in g
    g_dev[i*Ngrid + j] = g_loc;

  }

  return;

}

/*================================================
  This program uses the data and functions above 
  module above to compute the two-dimensional 
  function g(x,y). Note that we store the function 
  in a long vector of length Ngrid*Ngrid rather 
  than in a 2D matrix.
!===============================================*/
int main () {

  // Extend of domain in each dimension
  double gridMax =  10.0;
  double gridMin = -10.0;

  // Number of grid points in each dimension
  int Ngrid = 128;

  // Array to hold set of P points on the host
  double *tmpPoints;

  // Memory for g on the host
  double *g_host;

  // Memory for g on the device
  double *g_dev;

  // Variables to hold the dimensions of the block
  // and thread grids. The dim3 type is provided in cuda.h
  int blocksPerGrid,threadsPerBlock;
  dim3 blocks,threads;

  // cudaError_t is a type defined in cuda.h
  cudaError_t err;

  int i,j,count,idev;
  double x;

  // Make sure we have a CUDA capable device to work with
  err = cudaGetDeviceCount(&count);
  if ( (count==0) || (err!=cudaSuccess) ) {
    printf("No CUDA supported devices are available in this system.\n");
    exit(EXIT_FAILURE);
  } else {
    printf("Found %d CUDA devices in this system\n",count);
  }

  err = cudaGetDevice(&idev);
  if ( err!=cudaSuccess ) {
    printf("Error identifying active device\n");
    exit(EXIT_FAILURE);
  }
  printf("Using device %d\n",idev);

  // Allocate memory for points on the host
  tmpPoints = (double *)malloc(3*P*sizeof(double));

  // Populate tmpPoints with random numbers between -10 and 10
  for (i=0;i<P;i++) {
    for (j=0;j<3;j++) {
      x = rand()/(double)RAND_MAX;
      tmpPoints[3*i+j] = gridMin + (gridMax-gridMin)*x;
    }
  }

  // Allocate memory for points on the device
  //err = cudaMalloc(&points,3*P*sizeof(double));
  //if ( err!=cudaSuccess ) {
  //  printf("Error allocating memory for points on device\n");
  //  exit(EXIT_FAILURE);
  //}
  
  
  // Copy from tmpPoints on the host to points on the device
  //err = cudaMemcpy(points,tmpPoints,3*P*sizeof(double),cudaMemcpyHostToDevice);
  err = cudaMemcpyToSymbol(points,tmpPoints,3*P*sizeof(double));
  if ( err!=cudaSuccess ) {
    printf("Error copying points to device\n");
    exit(EXIT_FAILURE);
  }

  printf("Copied array of points to device memory\n");

  // Release memory on the host
  free(tmpPoints);

  // Allocate memory for g on the device and zero it out
  err = cudaMalloc(&g_dev,Ngrid*Ngrid*sizeof(double));
  if ( err!=cudaSuccess ) {
    printf("Error allocating memory for g_dev on device\n");
    exit(EXIT_FAILURE);
  }
  cudaMemset(g_dev,0,Ngrid*Ngrid*sizeof(double));

  // Allocate memory for g on the host and zero it out
  g_host = (double *)malloc(Ngrid*Ngrid*sizeof(double));
  memset(g_host,0,Ngrid*Ngrid*sizeof(double));

  // We want a thread running the integrate1D kernel for every
  // point in g_dev that we want to evaluate. 
  // Pick a sensible block size
  blocksPerGrid   = 4;
 
  // Calculate the number of threads per block to make up the
  // entire grid of Ngrid*Ngrid threads
  threadsPerBlock = Ngrid/blocksPerGrid;
  if (Ngrid%blocksPerGrid!=0) { threadsPerBlock += 1; }

  // Multidimensional grid dimensions, use the dim3 type
  blocks.x  = blocksPerGrid   ; blocks.y  = blocksPerGrid   ; blocks.z  = 1;
  threads.x = threadsPerBlock ; threads.y = threadsPerBlock ; threads.z = 1;

  // Launch our kernel to compute g_dev on the device
  printf("Using block grid dimensions of %d by %d\n",blocks.x,blocks.y);
  printf("Thread grid within a block is  %d by %d\n",threads.x,threads.y);

  clock_t t1 = clock();

  printf("Launching %d threads\n",blocks.x*blocks.y*threads.x*threads.y);

  integrate1D<<<blocks,threads>>>(gridMin,gridMax,Ngrid,g_dev);
  cudaThreadSynchronize();

  clock_t t2 = clock();

  // Copy from the device to the host
  err = cudaMemcpy(g_host,g_dev,Ngrid*Ngrid*sizeof(double),cudaMemcpyDeviceToHost);
  if ( err != cudaSuccess ) {
    printf("Error copying g from device to host\n");
    exit(EXIT_FAILURE);
  }

  printf("Time taken on GPU = %f milliseconds\n",(double)(t2-t1)*1000.0/(double)CLOCKS_PER_SEC);


  // Release device memory
  cudaFree(g_dev);
  cudaFree(points);

  // Release host memory
  free(g_host);

  return 0;

}
