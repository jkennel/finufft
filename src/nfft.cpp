// -*- mode: C++; c-indent-level: 4; c-basic-offset: 4; indent-tabs-mode: nil; -*-

#include <Rcpp.h>
using namespace Rcpp;

// this is all you must include for the finufft lib...
#include "finufft.h"

// also used in this example...
#include <vector>
#include <complex>
#include <cstdio>
#include <stdlib.h>
#include <cassert>


// [[Rcpp::export]]
Rcpp::ComplexVector nfft(std::vector<double> x,
                         std::vector<std::complex<double>>  c, 
                         int N,
                         double tol = 1e-9) {
  
  int M = x.size();
  
  int type = 1, dim = 1;     // 1d1
  BIGINT Ns[3];              // guru describes mode array by vector [N1,N2..]
  Ns[0] = N;
  int ntransf = 1;           // we want to do a single transform at a time
  finufft_plan plan;         // creates a plan struct
  
  finufft_makeplan(type, dim, Ns, +1, ntransf, tol, &plan, NULL);
  
  // note FINUFFT doesn't use std::vector types, so we need to make a pointer...
  finufft_setpts(plan, M, &x[0], NULL, NULL, 0, NULL, NULL, NULL);
  
  // alloc output array for the Fourier modes, then do the transform
  std::vector<std::complex<double>> F(N);
  
  int ier = finufft_execute(plan, &c[0], &F[0]);

  finufft_destroy(plan);    // done with transforms of this size
  
  return Rcpp::wrap(F);
  
}

