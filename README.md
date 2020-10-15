## finufft

Provide an interface to FINUFFT (https://github.com/flatironinstitute/finufft).

### Steps 
- Clone FINUFFT git repository (https://github.com/flatironinstitute/finufft) to inst folder.
- Run copy_finufft.R script to prepare files to be compiled.  This copies the source files from src, contrib, and include to the r package finufft/src directory and does a few minor changes so that it will compile.
- Make sure compiler flags work.

### Issues

- Conflicts when compiling with Intel MKL BLAS due to an FFTW conflict