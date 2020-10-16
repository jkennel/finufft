#include <Rcpp.h>
using namespace Rcpp;

// this is all you must include for the finufft
#include "finufft.h"

// also used in this example...
#include <vector>
#include <complex>
#include <cstdio>
#include <stdlib.h>
#include <cassert>


//==============================================================================
//' @title
//' nufft_1
//'
//' @description
//' wrapper for 1d dim nufft
//'
//' @param x locations
//' @param c complex weights
//' @param N1 number of output modes
//' @param tol precision
//' @param type dimension
//'
//' @return complex vector
//'
//' 
//' @export
// [[Rcpp::export]]
Rcpp::ComplexVector nufft_1(std::vector<double> x,
                            std::vector<std::complex<double>> c, 
                            int N1,
                            double tol = 1e-9,
                            int type = 1) {
  
  int M = x.size();
  
  int64_t Ns[] = {N1};       // N1,N2 as 64-bit int array
  int ntrans = 1;           // we want to do a single transform at a time
  finufft_plan plan;         // creates a plan struct
  
  finufft_makeplan(type, 1, Ns, +1, ntrans, tol, &plan, NULL);
  
  // note FINUFFT doesn't use std::vector types, so we need to make a pointer...
  finufft_setpts(plan, M, &x[0], NULL, NULL, 0, NULL, NULL, NULL);
  
  // alloc output array for the Fourier modes, then do the transform
  std::vector<std::complex<double>> F(N1);
  
  int ier = finufft_execute(plan, &c[0], &F[0]);
  
  finufft_destroy(plan);    // done with transforms of this size
  
  return Rcpp::wrap(F);
  
}


//==============================================================================
//' @title
//' nufft_2
//'
//' @description
//' wrapper for 2d nufft
//'
//' @param x locations
//' @param y locations
//' @param c complex weights
//' @param N1 number of output modes
//' @param N2 number of output modes
//' @param tol precision
//' @param type dimension
//'
//' @return complex vector
//'
//' 
//' @export
// [[Rcpp::export]]
Rcpp::ComplexVector nufft_2(std::vector<double> x,
                            std::vector<double> y,
                            std::vector<std::complex<double>> c, 
                            int N1,
                            int N2,
                            double tol = 1e-9,
                            int type = 1) {
  
  int M = x.size();
  
  int64_t Ns[] = {N1, N2};       // N1,N2 as 64-bit int array
  int ntrans = 1;                // we want to do a single transform at a time
  finufft_plan plan;             // creates a plan struct
  
  finufft_makeplan(type, 2, Ns, +1, ntrans, tol, &plan, NULL);
  
  // note FINUFFT doesn't use std::vector types, so we need to make a pointer...
  finufft_setpts(plan, M, &x[0], &y[0], NULL, 0, NULL, NULL, NULL);
  
  // alloc output array for the Fourier modes, then do the transform
  std::vector<std::complex<double>> F(N1*N2);
  
  int ier = finufft_execute(plan, &c[0], &F[0]);
  
  finufft_destroy(plan);    // done with transforms of this size
  
  return Rcpp::wrap(F);
  
}



//==============================================================================
//' @title
//' nufft_3
//'
//' @description
//' wrapper for 3d nufft
//'
//' @param x locations
//' @param y locations
//' @param z locations
//' @param c complex weights
//' @param N1 number of output modes
//' @param N2 number of output modes
//' @param N3 number of output modes
//' @param tol precision
//' @param type dimension
//'
//' @return complex vector
//'
//' 
//' @export
// [[Rcpp::export]]
Rcpp::ComplexVector nufft_3(std::vector<double> x,
                            std::vector<double> y,
                            std::vector<double> z,
                            std::vector<std::complex<double>> c, 
                            int N1,
                            int N2,
                            int N3,
                            double tol = 1e-9,
                            int type = 1) {
  
  int M = x.size();
  
  int64_t Ns[] = {N1, N2, N3};       // N1,N2 as 64-bit int array
  int ntrans = 1;                // we want to do a single transform at a time
  finufft_plan plan;             // creates a plan struct
  
  finufft_makeplan(type, 3, Ns, +1, ntrans, tol, &plan, NULL);
  
  // note FINUFFT doesn't use std::vector types, so we need to make a pointer...
  finufft_setpts(plan, M, &x[0], &y[0], &z[0], 0, NULL, NULL, NULL);
  
  // alloc output array for the Fourier modes, then do the transform
  std::vector<std::complex<double>> F(N1*N2*N3);
  
  int ier = finufft_execute(plan, &c[0], &F[0]);
  
  finufft_destroy(plan);    // done with transforms of this size
  
  return Rcpp::wrap(F);
  
}
