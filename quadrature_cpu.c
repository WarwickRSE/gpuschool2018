// -*- mode: C -*-

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>  // clock() and clock_t

// Global data which defines f(x,y,z)

// Number of points P
#define P 500

// Array of points at which Gaussians are located
double *points;

// Gaussian height parameter A
double A = 0.1f;

// Gaussian width parameter w
double w = 0.2f;

// This function evaluates f(x,y,z)
double func(double x, double y,double z) {

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

// This will, for a point on our two-dimension grid
// integrate over the function in the third dimension
void integrate1D(int i,int j, double Min, double Max, int Ngrid, double *g_host) {

  int k;
  double delta,x,y,z,p_old,p_new,g_loc;

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
  g_host[i*Ngrid + j] = g_loc;

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

  // Memory for g on the host
  double *g_host;

  int i,j;
  double x;

  // Allocate memory for points
  points = (double *)malloc(3*P*sizeof(double));

  // Populate points with random numbers between -10 and 10
  for (i=0;i<P;i++) {
    for (j=0;j<3;j++) {
      x = rand()/(double)RAND_MAX;
      points[3*i+j] = gridMin + (gridMax-gridMin)*x;
    }
  }

  // Allocate memory for g on the host and zero it out
  g_host = (double *)malloc(Ngrid*Ngrid*sizeof(double));
  memset(g_host,0,Ngrid*Ngrid*sizeof(double));
  
  clock_t t1 = clock();

  // Compute g and store in g_host
  for (i=0;i<Ngrid;i++) {
    for (j=0;j<Ngrid;j++) {
      integrate1D(i,j,gridMin,gridMax,Ngrid,g_host);
    }
  }

  clock_t t2 = clock();

  printf("Time taken on CPU = %f seconds\n",(double)(t2-t1)/(double)CLOCKS_PER_SEC);


  free(g_host);

  return 0;

}
