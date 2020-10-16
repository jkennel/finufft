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
//' nufft_1d1
//'
//' @description
//' wrapper for 1d dim nufft (dimension 1, type 1)
//'
//' @param xj locations
//' @param cj complex weights
//' @param n1 number of output modes
//' @param tol precision
//'
//' @return complex vector
//'
//' 
//' @export
// [[Rcpp::export]]
Rcpp::ComplexVector nufft_1d1(std::vector<double> xj,
                              std::vector<std::complex<double>> cj, 
                              int n1,
                              double tol = 1e-9) {
  
  int m = xj.size();
  
  int64_t ns[] = {n1};       // N1,N2 as 64-bit int array
  int ntrans = 1;            // we want to do a single transform at a time
  finufft_plan plan;         // creates a plan struct
  
  finufft_makeplan(1, 1, ns, +1, ntrans, tol, &plan, NULL);
  
  // note FINUFFT doesn't use std::vector types, so we need to make a pointer...
  finufft_setpts(plan, m, &xj[0], NULL, NULL, 0, NULL, NULL, NULL);
  
  // alloc output array for the Fourier modes, then do the transform
  std::vector<std::complex<double>> fk(n1);
  
  int ier = finufft_execute(plan, &cj[0], &fk[0]);
  
  finufft_destroy(plan);    // done with transforms of this size
  
  return Rcpp::wrap(fk);
  
}


//==============================================================================
//' @title
//' nufft_1d2 (dimension 1, type 2)
//'
//' @description
//' wrapper for 1d dim nufft (dimension 1, type 2)
//'
//' @param xj locations
//' @param fk complex weights
//' @param tol precision
//'
//' @return complex vector
//'
//' 
//' @export
// [[Rcpp::export]]
Rcpp::ComplexVector nufft_1d2(std::vector<double> xj,
                              std::vector<std::complex<double>> fk, 
                              double tol = 1e-9) {
  
  int m = fk.size();
  int n1 = xj.size();
  
  int64_t ns[] = {n1};       // N1,N2 as 64-bit int array
  int ntrans = 1;            // we want to do a single transform at a time
  finufft_plan plan;         // creates a plan struct
  
  finufft_makeplan(1, 1, ns, +1, ntrans, tol, &plan, NULL);
  
  // note FINUFFT doesn't use std::vector types, so we need to make a pointer...
  finufft_setpts(plan, m, &xj[0], NULL, NULL, 0, NULL, NULL, NULL);
  
  // alloc output array for the Fourier modes, then do the transform
  std::vector<std::complex<double>> cj(n1);
  
  int ier = finufft_execute(plan, &cj[0], &fk[0]);
  
  finufft_destroy(plan);    // done with transforms of this size
  
  return Rcpp::wrap(cj);
  
}



//==============================================================================
//' @title
//' nufft_1d3
//'
//' @description
//' wrapper for 1d dim nufft (dimension 1, type 3)
//'
//' @param xj locations
//' @param cj complex weights
//' @param sk locations
//' @param tol precision
//'
//' @return complex vector
//'
//' 
//' @export
// [[Rcpp::export]]
Rcpp::ComplexVector nufft_1d3(std::vector<double> xj,
                              std::vector<std::complex<double>> cj, 
                              std::vector<double> sk,
                              double tol = 1e-9) {
  
  int m = xj.size();
  int n1 = sk.size();
  
  int64_t ns[] = {n1};       // N1,N2 as 64-bit int array
  int ntrans = 1;            // we want to do a single transform at a time
  finufft_plan plan;         // creates a plan struct
  
  finufft_makeplan(1, 1, ns, +1, ntrans, tol, &plan, NULL);
  
  // note FINUFFT doesn't use std::vector types, so we need to make a pointer...
  finufft_setpts(plan, m, &xj[0], NULL, NULL, n1, &sk[0], NULL, NULL);
  
  // alloc output array for the Fourier modes, then do the transform
  std::vector<std::complex<double>> fk(n1);
  
  int ier = finufft_execute(plan, &cj[0], &fk[0]);
  
  finufft_destroy(plan);    // done with transforms of this size
  
  return Rcpp::wrap(fk);
  
}

