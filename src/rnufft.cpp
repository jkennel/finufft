#define BOOST_DISABLE_ASSERTS
#define ARMA_DONT_PRINT_ERRORS
// #define ARMA_USE_TBB_ALLOC
// #define ARMA_DONT_USE_OPENMP

// [[Rcpp::depends(RcppArmadillo)]]


#include <RcppArmadillo.h>

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
//' @param iflag + or - i
//'
//' @return complex vector
//'
//'
//' @export
// [[Rcpp::export]]
arma::cx_vec nufft_1d1(arma::vec xj,
                       arma::cx_vec cj,
                       int n1,
                       double tol = 1e-9,
                       int iflag = 1) {
  
  int m = xj.n_elem;
  
  int64_t ns[] = {n1,1,1};       // N1,N2 as 64-bit int array
  int ntrans = 1;            // we want to do a single transform at a time
  finufft_plan plan;         // creates a plan struct
  
  finufft_makeplan(1, 1, ns, +1, ntrans, tol, &plan, NULL);
  
  // note FINUFFT doesn't use std::vector types, so we need to make a pointer...
  finufft_setpts(plan, m, &xj[0], NULL, NULL, 0, NULL, NULL, NULL);
  
  // alloc output array for the Fourier modes, then do the transform
  arma::cx_vec fk(n1);
  
  int ier = finufft_execute(plan, &cj[0], &fk[0]);
  
  finufft_destroy(plan);    // done with transforms of this size
  
  return fk;
  
}

//==============================================================================
//' @title
//' nufft_2d1
//'
//' @description
//' wrapper for 2d dim nufft (dimension 2, type 1)
//'
//' @param xj locations
//' @param yj locations
//' @param cj complex weights
//' @param n1 number of output modes
//' @param n2 number of output modes
//' @param tol precision
//' @param iflag + or - i
//'
//' @return complex vector
//'
//'
//' @export
// [[Rcpp::export]]
arma::cx_mat nufft_2d1(arma::vec xj,
                       arma::vec yj,
                       arma::cx_vec cj,
                       int n1,
                       int n2,
                       double tol = 1e-9,
                       int iflag = 1) {
  
  int m = xj.n_elem;
  
  int64_t ns[] = {n1, n2, 1};       // N1,N2 as 64-bit int array
  int ntrans = 1;                   // we want to do a single transform at a time
  finufft_plan plan;                // creates a plan struct
  
  finufft_makeplan(1, 2, ns, +1, ntrans, tol, &plan, NULL);
  
  // note FINUFFT doesn't use std::vector types, so we need to make a pointer...
  finufft_setpts(plan, m, &xj[0], &yj[0], NULL, 0, NULL, NULL, NULL);
  
  // alloc output array for the Fourier modes, then do the transform
  arma::cx_mat fk(n1, n2);
  
  int ier = finufft_execute(plan, &cj[0], &fk[0]);
  
  finufft_destroy(plan);    // done with transforms of this size
  
  return fk;
  
}

//==============================================================================
//' @title
//' nufft_3d1
//'
//' @description
//' wrapper for 3d dim nufft (dimension 1, type 1)
//'
//' @param xj locations
//' @param yj locations
//' @param zj locations
//' @param cj complex weights
//' @param n1 number of output modes
//' @param n2 number of output modes
//' @param n3 number of output modes
//' @param tol precision
//' @param iflag + or - i
//'
//' @return complex vector
//'
//'
//' @export
// [[Rcpp::export]]
arma::cx_cube nufft_3d1(arma::vec xj,
                        arma::vec yj,
                        arma::vec zj,
                        arma::cx_vec cj,
                        int n1,
                        int n2,
                        int n3,
                        double tol = 1e-9,
                        int iflag = 1) {
  
  int m = xj.n_elem;
  
  int64_t ns[] = {n1, n2, n3};       // N1,N2 as 64-bit int array
  int ntrans = 1;            // we want to do a single transform at a time
  finufft_plan plan;         // creates a plan struct
  
  finufft_makeplan(1, 3, ns, +1, ntrans, tol, &plan, NULL);
  
  // note FINUFFT doesn't use std::vector types, so we need to make a pointer...
  finufft_setpts(plan, m, &xj[0], &yj[0], &zj[0], 0, NULL, NULL, NULL);
  
  // alloc output array for the Fourier modes, then do the transform
  arma::cx_cube fk(n1, n2, n3);
  
  int ier = finufft_execute(plan, &cj[0], &fk[0]);
  
  finufft_destroy(plan);    // done with transforms of this size
  
  return fk;
  
}



//==============================================================================
//==============================================================================
//==============================================================================
//==============================================================================
//==============================================================================
//==============================================================================
//' @title
//' nufft_1d2
//'
//' @description
//' wrapper for 1d dim nufft (dimension 1, type 2)
//'
//' @param xj locations
//' @param fk complex weights
//' @param tol precision
//' @param iflag + or - i
//'
//' @return complex vector
//'
//'
//' @export
// [[Rcpp::export]]
arma::cx_vec nufft_1d2(arma::vec xj,
                       arma::cx_vec fk,
                       double tol = 1e-9,
                       int iflag = 1) {
  
  int m = xj.n_elem;
  int n1 = fk.n_elem;
  
  int64_t ns[] = {n1,1,1};        // N1,N2 as 64-bit int array
  int ntrans = 1;            // we want to do a single transform at a time
  finufft_plan plan;         // creates a plan struct
  
  finufft_makeplan(2, 1, ns, +1, ntrans, tol, &plan, NULL);
  
  // note FINUFFT doesn't use std::vector types, so we need to make a pointer...
  finufft_setpts(plan, m, &xj[0], NULL, NULL, 0, NULL, NULL, NULL);
  
  // alloc output array for the Fourier modes, then do the transform
  arma::cx_vec cj(m);
  
  int ier = finufft_execute(plan, &cj[0], &fk[0]);
  
  finufft_destroy(plan);    // done with transforms of this size
  
  return cj;
  
}


//==============================================================================
//' @title
//' nufft_2d2
//'
//' @description
//' wrapper for 2d dim nufft (dimension 2, type 2)
//'
//' @param xj locations
//' @param yj locations
//' @param fk complex weights
//' @param tol precision
//' @param iflag + or - i
//'
//' @return complex vector
//'
//'
//' @export
// [[Rcpp::export]]
arma::cx_vec nufft_2d2(arma::vec xj,
                       arma::vec yj,
                       arma::cx_mat fk,
                       double tol = 1e-9,
                       int iflag = 1) {
  
  int m  = xj.n_elem;
  int n1 = fk.n_rows;
  int n2 = fk.n_cols;
  
  int64_t ns[] = {n1, n2, 1};   // N1,N2 as 64-bit int array
  int ntrans = 1;               // we want to do a single transform at a time
  finufft_plan plan;            // creates a plan struct
  
  finufft_makeplan(2, 2, ns, +1, ntrans, tol, &plan, NULL);
  
  // note FINUFFT doesn't use std::vector types, so we need to make a pointer...
  finufft_setpts(plan, m, &xj[0], &yj[0], NULL, 0, NULL, NULL, NULL);
  
  // alloc output array for the Fourier modes, then do the transform
  arma::cx_vec cj(m);
  
  int ier = finufft_execute(plan, &cj[0], &fk[0]);
  
  finufft_destroy(plan);    // done with transforms of this size
  
  return cj;
  
}

//==============================================================================
//' @title
//' nufft_3d2
//'
//' @description
//' wrapper for 3d dim nufft (dimension 3, type 2)
//'
//' @param xj locations
//' @param yj locations
//' @param zj locations
//' @param fk complex weights
//' @param tol precision
//' @param iflag + or - i
//'
//' @return complex vector
//'
//'
//' @export
// [[Rcpp::export]]
arma::cx_vec nufft_3d2(arma::vec xj,
                       arma::vec yj,
                       arma::vec zj,
                       arma::cx_cube fk,
                       double tol = 1e-9,
                       int iflag = 1) {
  
  int m = xj.n_elem;
  
  int n1 = fk.n_rows;
  int n2 = fk.n_cols;
  int n3 = fk.n_slices;
  
  int64_t ns[] = {n1, n2, n3};   // N1,N2 as 64-bit int array
  int ntrans = 1;                // we want to do a single transform at a time
  finufft_plan plan;             // creates a plan struct
  
  finufft_makeplan(2, 3, ns, +1, ntrans, tol, &plan, NULL);
  
  // note FINUFFT doesn't use std::vector types, so we need to make a pointer...
  finufft_setpts(plan, m, &xj[0], &yj[0], &zj[0], 0, NULL, NULL, NULL);
  
  // alloc output array for the Fourier modes, then do the transform
  arma::cx_vec cj(m);
  
  int ier = finufft_execute(plan, &cj[0], &fk[0]);
  
  finufft_destroy(plan);    // done with transforms of this size
  
  return cj;
  
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
//' @param iflag + or - i
//'
//' @return complex vector
//'
//'
//' @export
// [[Rcpp::export]]
arma::cx_vec nufft_1d3(arma::vec xj,
                       arma::cx_vec cj,
                       arma::vec sk,
                       double tol = 1e-9,
                       int iflag = 1) {
  
  int m = xj.n_elem;
  int n1 = sk.n_elem;
  
  int64_t ns[] = {n1,1,1};   // N1,N2 as 64-bit int array
  int ntrans = 1;            // we want to do a single transform at a time
  finufft_plan plan;         // creates a plan struct
  
  finufft_makeplan(3, 1, ns, +1, ntrans, tol, &plan, NULL);
  
  // note FINUFFT doesn't use std::vector types, so we need to make a pointer...
  finufft_setpts(plan, m, &xj[0], NULL, NULL, n1, &sk[0], NULL, NULL);
  
  // alloc output array for the Fourier modes, then do the transform
  arma::cx_vec fk(n1);
  
  int ier = finufft_execute(plan, &cj[0], &fk[0]);
  
  finufft_destroy(plan);    // done with transforms of this size
  
  return fk;
  
}


//==============================================================================
//' @title
//' nufft_2d3
//'
//' @description
//' wrapper for 2d dim nufft (dimension 2, type 3)
//'
//' @param xj locations
//' @param yj locations
//' @param cj complex weights
//' @param sk locations
//' @param tk locations
//' @param tol precision
//' @param iflag + or - i
//'
//' @return complex vector
//'
//'
//' @export
// [[Rcpp::export]]
arma::cx_vec nufft_2d3(arma::vec xj,
                       arma::vec yj,
                       arma::cx_vec cj,
                       arma::vec sk,
                       arma::vec tk,
                       double tol = 1e-9,
                       int iflag = 1) {
  
  int m = xj.n_elem;
  int n1 = sk.n_elem;
  
  int64_t ns[] = {n1,1,1};   // N1,N2 as 64-bit int array
  int ntrans = 1;            // we want to do a single transform at a time
  finufft_plan plan;         // creates a plan struct
  
  finufft_makeplan(3, 2, ns, +1, ntrans, tol, &plan, NULL);
  
  // note FINUFFT doesn't use std::vector types, so we need to make a pointer...
  finufft_setpts(plan, m, &xj[0], &yj[0], NULL, n1, &sk[0], &tk[0], NULL);
  
  // alloc output array for the Fourier modes, then do the transform
  arma::cx_vec fk(n1);
  
  int ier = finufft_execute(plan, &cj[0], &fk[0]);
  
  finufft_destroy(plan);    // done with transforms of this size
  
  return fk;
  
}


//==============================================================================
//' @title
//' nufft_3d3
//'
//' @description
//' wrapper for 3d dim nufft (dimension 3, type 3)
//'
//' @param xj locations
//' @param yj locations
//' @param zj locations
//' @param cj complex weights
//' @param sk locations
//' @param tk locations
//' @param uk locations
//' @param tol precision
//' @param iflag + or - i
//'
//' @return complex vector
//'
//'
//' @export
// [[Rcpp::export]]
arma::cx_vec nufft_3d3(arma::vec xj,
                       arma::vec yj,
                       arma::vec zj,
                       arma::cx_vec cj,
                       arma::vec sk,
                       arma::vec tk,
                       arma::vec uk,
                       double tol = 1e-9,
                       int iflag = 1) {
  
  int m = xj.n_elem;
  int n1 = sk.n_elem;
  
  int64_t ns[] = {n1,1,1};   // N1,N2 as 64-bit int array
  int ntrans = 1;            // we want to do a single transform at a time
  finufft_plan plan;         // creates a plan struct
  
  finufft_makeplan(3, 3, ns, +1, ntrans, tol, &plan, NULL);
  
  // note FINUFFT doesn't use std::vector types, so we need to make a pointer...
  finufft_setpts(plan, m, &xj[0], &yj[0], &zj[0], n1, &sk[0], &tk[0], &uk[0]);
  
  // alloc output array for the Fourier modes, then do the transform
  arma::cx_vec fk(n1);
  
  int ier = finufft_execute(plan, &cj[0], &fk[0]);
  
  finufft_destroy(plan);    // done with transforms of this size
  
  return fk;
  
}
