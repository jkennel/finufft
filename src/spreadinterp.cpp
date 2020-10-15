#include "spreadinterp.h"
#include "dataTypes.h"
#include "defs.h"
#include "utils.h"
#include "utils_precindep.h"

#include <stdlib.h>
#include <vector>
#include <math.h>
#include <stdio.h>
using namespace std;

// declarations of purely internal functions...
static inline void set_kernel_args(FLT *args, FLT x, const spread_opts& opts);
static inline void evaluate_kernel_vector(FLT *ker, FLT *args, const spread_opts& opts, const int N);
static inline void eval_kernel_vec_Horner(FLT *ker, const FLT z, const int w, const spread_opts &opts);
void interp_line(FLT *out,FLT *du, FLT *ker,BIGINT i1,BIGINT N1,int ns);
void interp_square(FLT *out,FLT *du, FLT *ker1, FLT *ker2, BIGINT i1,BIGINT i2,BIGINT N1,BIGINT N2,int ns);
void interp_cube(FLT *out,FLT *du, FLT *ker1, FLT *ker2, FLT *ker3,
		 BIGINT i1,BIGINT i2,BIGINT i3,BIGINT N1,BIGINT N2,BIGINT N3,int ns);
void spread_subproblem_1d(BIGINT N1,FLT *du0,BIGINT M0,FLT *kx0,FLT *dd0,
			  const spread_opts& opts);
void spread_subproblem_2d(BIGINT N1,BIGINT N2,FLT *du0,BIGINT M0,
			  FLT *kx0,FLT *ky0,FLT *dd0,const spread_opts& opts);
void spread_subproblem_3d(BIGINT N1,BIGINT N2,BIGINT N3,FLT *du0,BIGINT M0,
			  FLT *kx0,FLT *ky0,FLT *kz0,FLT *dd0,
			  const spread_opts& opts);
void add_wrapped_subgrid(BIGINT offset1,BIGINT offset2,BIGINT offset3,
			 BIGINT size1,BIGINT size2,BIGINT size3,BIGINT N1,
			 BIGINT N2,BIGINT N3,FLT *data_uniform, FLT *du0);
void bin_sort_singlethread(BIGINT *ret, BIGINT M, FLT *kx, FLT *ky, FLT *kz,
	      BIGINT N1,BIGINT N2,BIGINT N3,int pirange,
	      double bin_size_x,double bin_size_y,double bin_size_z, int debug);
void bin_sort_multithread(BIGINT *ret, BIGINT M, FLT *kx, FLT *ky, FLT *kz,
	      BIGINT N1,BIGINT N2,BIGINT N3,int pirange,
              double bin_size_x,double bin_size_y,double bin_size_z, int debug,
              int nthr);
void get_subgrid(BIGINT &offset1,BIGINT &offset2,BIGINT &offset3,BIGINT &size1,
		 BIGINT &size2,BIGINT &size3,BIGINT M0,FLT* kx0,FLT* ky0,
		 FLT* kz0,int ns, int ndims);



/* local NU coord fold+rescale macro: does the following affine transform to x:
     when p=true:   map [-3pi,-pi) and [-pi,pi) and [pi,3pi)    each to [0,N)
     otherwise,     map [-N,0) and [0,N) and [N,2N)             each to [0,N)
   Thus, only one period either side of the principal domain is folded.
   (It is *so* much faster than slow std::fmod that we stick to it.)
   This explains FINUFFT's allowed input domain of [-3pi,3pi).
   Speed comparisons of this macro vs a function are in devel/foldrescale*.
   The macro wins hands-down on i7, even for modern GCC9.
*/
#define FOLDRESCALE(x,N,p) (p ?                                         \
         (x + (x>=-PI ? (x<PI ? PI : -PI) : 3*PI)) * ((FLT)M_1_2PI*N) : \
                        (x>=0.0 ? (x<(FLT)N ? x : x-(FLT)N) : x+(FLT)N))



// ==========================================================================
int spreadinterp(
        BIGINT N1, BIGINT N2, BIGINT N3, FLT *data_uniform,
        BIGINT M, FLT *kx, FLT *ky, FLT *kz, FLT *data_nonuniform,
        spread_opts opts)
/* ------------Spreader/interpolator for 1, 2, or 3 dimensions --------------
   If opts.spread_direction=1, evaluate, in the 1D case,

                         N1-1
   data_nonuniform[j] =  SUM phi(kx[j] - n) data_uniform[n],   for j=0...M-1
                         n=0

   If opts.spread_direction=2, evaluate its transpose, in the 1D case,

                      M-1
   data_uniform[n] =  SUM phi(kx[j] - n) data_nonuniform[j],   for n=0...N1-1
                      j=0

   In each case phi is the spreading kernel, which has support
   [-opts.nspread/2,opts.nspread/2]. In 2D or 3D, the generalization with
   product of 1D kernels is performed.
   For 1D set N2=N3=1; for 2D set N3=1; for 3D set N1,N2,N3>0.

   Notes:
   No particular normalization of the spreading kernel is assumed.
   Uniform (U) points are centered at coords
   [0,1,...,N1-1] in 1D, analogously in 2D and 3D. They are stored in x
   fastest, y medium, z slowest ordering, up to however many
   dimensions are relevant; note that this is Fortran-style ordering for an
   array f(x,y,z), but C style for f[z][y][x]. This is to match the fortran
   interface of the original CMCL libraries.
   Non-uniform (NU) points kx,ky,kz are real.
   If pirange=0, should be in the range [0,N1] in 1D, analogously in 2D and 3D.
   If pirange=1, the range is instead [-pi,pi] for each coord.
   The spread_opts struct must have been set up already by calling setup_kernel.
   It is assumed that 2*opts.nspread < min(N1,N2,N3), so that the kernel
   only ever wraps once when falls below 0 or off the top of a uniform grid
   dimension.

   Inputs:
   N1,N2,N3 - grid sizes in x (fastest), y (medium), z (slowest) respectively.
              If N2==0, 1D spreading is done. If N3==0, 2D spreading.
	      Otherwise, 3D.
   M - number of NU pts.
   kx, ky, kz - length-M real arrays of NU point coordinates (only kz used in
                1D, only kx and ky used in 2D).
		These should lie in the box 0<=kx<=N1 etc (if pirange=0),
                or -pi<=kx<=pi (if pirange=1). However, points up to +-1 period
                outside this domain are also correctly folded back into this
                domain, but pts beyond this either raise an error (if chkbnds=1)
                or a crash (if chkbnds=0).
   opts - object controlling spreading method and text output, has fields
          including:
        spread_direction=1, spreads from nonuniform input to uniform output, or
        spread_direction=2, interpolates ("spread transpose") from uniform input
                            to nonuniform output.
	pirange = 0: kx,ky,kz coords in [0,N]. 1: coords in [-pi,pi].
                (due to +-1 box folding these can be out to [-N,2N] and
                [-3pi/2,3pi/2] respectively).
	sort = 0,1,2: whether to sort NU points using natural yz-grid
	       ordering. 0: don't, 1: do, 2: use heuristic choice (default)
        sort_threads = 0, 1,... : if >0, set # sorting threads; if 0
                   allow heuristic choice (either single or all avail).
	kerpad = 0,1: whether pad to next mult of 4, helps SIMD (kerevalmeth=0).
	kerevalmeth = 0: direct exp(sqrt(..)) eval; 1: Horner piecewise poly.
	debug = 0: no text output, 1: some openmp output, 2: mega output
	           (each NU pt)
	chkbnds = 0: don't check incoming NU pts for bounds (but still fold +-1)
                  1: do, and stop with error if any found outside valid bnds
	flags = integer with binary bits determining various timing options
                (set to 0 unless expert; see cnufftspread.h)

   Inputs/Outputs:
   data_uniform - output values on grid (dir=1) OR input grid data (dir=2)
   data_nonuniform - input strengths of the sources (dir=1)
                     OR output values at targets (dir=2)
   Returned value:
   0 indicates success; other values as follows (see spreadcheck below and
   see utils.h and ../docs/usage.rst for error codes):
      3 : one or more non-trivial box dimensions is less than 2.nspread.
      4 : nonuniform points outside [-Nm,2*Nm] or [-3pi,3pi] in at least one
          dimension m=1,2,3.
      5 : failed allocate sort indices
      6 : invalid opts.spread_direction


   Magland Dec 2016. Barnett openmp version, many speedups 1/16/17-2/16/17
   error codes 3/13/17. pirange 3/28/17. Rewritten 6/15/17. parallel sort 2/9/18
   No separate subprob indices in t-1 2/11/18.
   sort_threads (since for M<<N, multithread sort slower than single) 3/27/18
   kereval, kerpad 4/24/18
   Melody Shih split into 3 routines: check, sort, spread. Jun 2018, making
   this routine just a caller to them. Name change, Barnett 7/27/18
   Tidy, Barnett 5/20/20.
*/
{
  int ier = spreadcheck(N1, N2, N3, M, kx, ky, kz, opts);
  if (ier)
    return ier;
  BIGINT* sort_indices = (BIGINT*)malloc(sizeof(BIGINT)*M);
  if (!sort_indices) {
    fprintf(stderr,"%s failed to allocate sort_indices!\n",__func__);
    return ERR_SPREAD_ALLOC;
  }
  int did_sort = indexSort(sort_indices, N1, N2, N3, M, kx, ky, kz, opts);
  spreadinterpSorted(sort_indices, N1, N2, N3, data_uniform,
                     M, kx, ky, kz, data_nonuniform, opts, did_sort);
  free(sort_indices);
  return 0;
}

static int ndims_from_Ns(BIGINT N1, BIGINT N2, BIGINT N3)
/* rule for getting number of spreading dimensions from the list of Ns per dim.
   Split out, Barnett 7/26/18
*/
{
  int ndims = 1;                // decide ndims: 1,2 or 3
  if (N2>1) ++ndims;
  if (N3>1) ++ndims;
  return ndims;
}

int spreadcheck(BIGINT N1, BIGINT N2, BIGINT N3, BIGINT M, FLT *kx, FLT *ky,
                FLT *kz, spread_opts opts)
/* This does just the input checking and reporting for the spreader.
   See spreadinterp() for input arguments and meaning of returned value.
   Split out by Melody Shih, Jun 2018. Finiteness chk Barnett 7/30/18.
*/
{
  CNTime timer;
  // INPUT CHECKING & REPORTING .... cuboid not too small for spreading?
  int minN = 2*opts.nspread;
  if (N1<minN || (N2>1 && N2<minN) || (N3>1 && N3<minN)) {
    fprintf(stderr,"%s error: one or more non-trivial box dims is less than 2.nspread!\n",__func__);
    return ERR_SPREAD_BOX_SMALL;
  }
  if (opts.spread_direction!=1 && opts.spread_direction!=2) {
    fprintf(stderr,"%s error: opts.spread_direction must be 1 or 2!\n",__func__);
    return ERR_SPREAD_DIR;
  }
  int ndims = ndims_from_Ns(N1,N2,N3);
  
  // BOUNDS CHECKING .... check NU pts are valid (incl +-1 box), exit gracefully
  if (opts.chkbnds) {
    timer.start();
    for (BIGINT i=0; i<M; ++i) {
      FLT x=FOLDRESCALE(kx[i],N1,opts.pirange);  // this includes +-1 box folding
      if (x<0 || x>N1 || !isfinite(x)) {     // note isfinite() breaks with -Ofast
        fprintf(stderr,"%s NU pt not in valid range (central three periods): kx=%g, N1=%lld (pirange=%d)\n",__func__,x,(long long)N1,opts.pirange);
        return ERR_SPREAD_PTS_OUT_RANGE;
      }
    }
    if (ndims>1)
      for (BIGINT i=0; i<M; ++i) {
        FLT y=FOLDRESCALE(ky[i],N2,opts.pirange);
        if (y<0 || y>N2 || !isfinite(y)) {
          fprintf(stderr,"%s NU pt not in valid range (central three periods): ky=%g, N2=%lld (pirange=%d)\n",__func__,y,(long long)N2,opts.pirange);
          return ERR_SPREAD_PTS_OUT_RANGE;
        }
      }
    if (ndims>2)
      for (BIGINT i=0; i<M; ++i) {
        FLT z=FOLDRESCALE(kz[i],N3,opts.pirange);
        if (z<0 || z>N3 || !isfinite(z)) {
          fprintf(stderr,"%s NU pt not in valid range (central three periods): kz=%g, N3=%lld (pirange=%d)\n",__func__,z,(long long)N3,opts.pirange);
          return ERR_SPREAD_PTS_OUT_RANGE;
        }
      }
    if (opts.debug) printf("\tNU bnds check:\t\t%.3g s\n",timer.elapsedsec());
  }
  return 0; 
}

int indexSort(BIGINT* sort_indices, BIGINT N1, BIGINT N2, BIGINT N3, BIGINT M, 
               FLT *kx, FLT *ky, FLT *kz, spread_opts opts)
/* This makes a decision whether or not to sort the NU pts, and if so, calls
   either single- or multi-threaded bin sort, writing reordered index list to
   sort_indices.
   See spreadinterp() for input arguments, and ../docs/opts.rst for opts.
   Return value is whether a sort was done or not.
   Barnett 2017; split out by Melody Shih, Jun 2018.
*/
{
  CNTime timer;
  int ndims = ndims_from_Ns(N1,N2,N3);
  BIGINT N=N1*N2*N3;            // output array size
  
  // heuristic binning box size for U grid... affects performance:
  double bin_size_x = 16, bin_size_y = 4, bin_size_z = 4;
  // put in heuristics based on cache sizes (only useful for single-thread) ?

  int better_to_sort = !(ndims==1 && (opts.spread_direction==2 || (M > 1000*N1))); // 1D small-N or dir=2 case: don't sort

  timer.start();                 // if needed, sort all the NU pts...
  int did_sort=0;
  int maxnthr = MY_OMP_GET_MAX_THREADS();
  if (opts.nthreads>0)           // user override up to max avail
    maxnthr = min(maxnthr,opts.nthreads);
  
  if (opts.sort==1 || (opts.sort==2 && better_to_sort)) {
    // store a good permutation ordering of all NU pts (dim=1,2 or 3)
    int sort_debug = (opts.debug>=2);    // show timing output?
    int sort_nthr = opts.sort_threads;   // choose # threads for sorting
    if (sort_nthr==0)   // use auto choice: when N>>M, one thread is better!
      sort_nthr = (10*M>N) ? maxnthr : 1;      // heuristic
    if (sort_nthr==1)
      bin_sort_singlethread(sort_indices,M,kx,ky,kz,N1,N2,N3,opts.pirange,bin_size_x,bin_size_y,bin_size_z,sort_debug);
    else                                      // sort_nthr>1, sets # threads
      bin_sort_multithread(sort_indices,M,kx,ky,kz,N1,N2,N3,opts.pirange,bin_size_x,bin_size_y,bin_size_z,sort_debug,sort_nthr);
    if (opts.debug) 
      printf("\tsorted (%d threads):\t%.3g s\n",sort_nthr,timer.elapsedsec());
    did_sort=1;
  } else {
#pragma omp parallel for num_threads(maxnthr) schedule(static,1000000)
    for (BIGINT i=0; i<M; i++)                // omp helps xeon, hinders i7
      sort_indices[i]=i;                      // the identity permutation
    if (opts.debug)
      printf("\tnot sorted (sort=%d): \t%.3g s\n",(int)opts.sort,timer.elapsedsec());
  }
  return did_sort;
}


// --------------------------------------------------------------------------
int spreadSorted(BIGINT* sort_indices,BIGINT N1, BIGINT N2, BIGINT N3, 
		      FLT *data_uniform,BIGINT M, FLT *kx, FLT *ky, FLT *kz,
		      FLT *data_nonuniform, spread_opts opts, int did_sort)
{
  CNTime timer;
  int ndims = ndims_from_Ns(N1,N2,N3);
  BIGINT N=N1*N2*N3;            // output array size
  int ns=opts.nspread;          // abbrev. for w, kernel width
  int nthr = MY_OMP_GET_MAX_THREADS();  // # threads to use to spread
  if (opts.nthreads>0)
    nthr = min(nthr,opts.nthreads);     // user override up to max avail
  if (opts.debug)
    printf("\tspread %dD (M=%lld; N1=%lld,N2=%lld,N3=%lld; pir=%d), nthr=%d\n",ndims,(long long)M,(long long)N1,(long long)N2,(long long)N3,opts.pirange,nthr);
  
  timer.start();
  for (BIGINT i=0; i<2*N; i++) // zero the output array. std::fill is no faster
    data_uniform[i]=0.0;
  if (opts.debug) printf("\tzero output array\t%.3g s\n",timer.elapsedsec());
  if (M==0)                     // no NU pts, we're done
    return 0;
  
  int spread_single = (nthr==1) || (M*100<N);     // low-density heuristic?
  spread_single = 0;                 // for now
  timer.start();
  if (spread_single) {    // ------- Basic single-core t1 spreading ------
    for (BIGINT j=0; j<M; j++) {
      // *** todo, not urgent
      // ... (question is: will the index wrapping per NU pt slow it down?)
    }
    if (opts.debug) printf("\tt1 simple spreading:\t%.3g s\n",timer.elapsedsec());
    
  } else {           // ------- Fancy multi-core blocked t1 spreading ----
                     // Splits sorted inds (jfm's advanced2), could double RAM.
    int nb = min(4*(BIGINT)nthr,M);  // choose nb (# subprobs) via used nthreads
    if (nb*opts.max_subproblem_size<M)
      nb = 1 + (M-1)/opts.max_subproblem_size;  // int div does ceil(M/opts.max_subproblem_size)
    if (M*1000<N) {         // low-density heuristic: one thread per NU pt!
      nb = M;
      if (opts.debug) printf("\tusing low-density speed rescue nb=M...\n");
    }
    if (!did_sort && nthr==1) {
      nb = 1;
      if (opts.debug) printf("\tunsorted nthr=1: forcing single subproblem...\n");
    }
    std::vector<BIGINT> brk(nb+1); // NU index breakpoints defining nb subproblems
    for (int p=0;p<=nb;++p)
      brk[p] = (BIGINT)(0.5 + M*p/(double)nb);
    
#pragma omp parallel for num_threads(nthr) schedule(dynamic,1)  // each is big
      for (int isub=0; isub<nb; isub++) {   // Main loop through the subproblems
        BIGINT M0 = brk[isub+1]-brk[isub];  // # NU pts in this subproblem
        // copy the location and data vectors for the nonuniform points
        FLT *kx0=(FLT*)malloc(sizeof(FLT)*M0), *ky0=NULL, *kz0=NULL;
        if (N2>1)
          ky0=(FLT*)malloc(sizeof(FLT)*M0);
        if (N3>1)
          kz0=(FLT*)malloc(sizeof(FLT)*M0);
        FLT *dd0=(FLT*)malloc(sizeof(FLT)*M0*2);    // complex strength data
        for (BIGINT j=0; j<M0; j++) {           // todo: can avoid this copying?
          BIGINT kk=sort_indices[j+brk[isub]];  // NU pt from subprob index list
          kx0[j]=FOLDRESCALE(kx[kk],N1,opts.pirange);
          if (N2>1) ky0[j]=FOLDRESCALE(ky[kk],N2,opts.pirange);
          if (N3>1) kz0[j]=FOLDRESCALE(kz[kk],N3,opts.pirange);
          dd0[j*2]=data_nonuniform[kk*2];     // real part
          dd0[j*2+1]=data_nonuniform[kk*2+1]; // imag part
        }
        // get the subgrid which will include padding by roughly nspread/2
        BIGINT offset1,offset2,offset3,size1,size2,size3; // get_subgrid sets
        get_subgrid(offset1,offset2,offset3,size1,size2,size3,M0,kx0,ky0,kz0,ns,ndims);  // sets offsets and sizes
        if (opts.debug>1) { // verbose
          if (ndims==1)
            printf("\tsubgrid: off %lld\t siz %lld\t #NU %lld\n",(long long)offset1,(long long)size1,(long long)M0);
          else if (ndims==2)
            printf("\tsubgrid: off %lld,%lld\t siz %lld,%lld\t #NU %lld\n",(long long)offset1,(long long)offset2,(long long)size1,(long long)size2,(long long)M0);
          else
            printf("\tsubgrid: off %lld,%lld,%lld\t siz %lld,%lld,%lld\t #NU %lld\n",(long long)offset1,(long long)offset2,(long long)offset3,(long long)size1,(long long)size2,(long long)size3,(long long)M0);
	}
        for (BIGINT j=0; j<M0; j++) {
          kx0[j]-=offset1;  // now kx0 coords are relative to corner of subgrid
          if (N2>1) ky0[j]-=offset2;  // only accessed if 2D or 3D
          if (N3>1) kz0[j]-=offset3;  // only access if 3D
        }
        // allocate output data for this subgrid
        FLT *du0=(FLT*)malloc(sizeof(FLT)*2*size1*size2*size3); // complex
        
        // Spread to subgrid without need for bounds checking or wrapping
        if (!(opts.flags & TF_OMIT_SPREADING)) {
          if (ndims==1)
            spread_subproblem_1d(size1,du0,M0,kx0,dd0,opts);
          else if (ndims==2)
            spread_subproblem_2d(size1,size2,du0,M0,kx0,ky0,dd0,opts);
          else
            spread_subproblem_3d(size1,size2,size3,du0,M0,kx0,ky0,kz0,dd0,opts);
	}
        
#pragma omp critical
        {  // do the adding of subgrid to output; only here threads cannot clash
          if (!(opts.flags & TF_OMIT_WRITE_TO_GRID))
            add_wrapped_subgrid(offset1,offset2,offset3,size1,size2,size3,N1,N2,N3,data_uniform,du0);
        }  // end critical block

        // free up stuff from this subprob... (that was malloc'ed by hand)
        free(dd0);
        free(du0);
        free(kx0);
        if (N2>1) free(ky0);
        if (N3>1) free(kz0); 
      }     // end main loop over subprobs
      if (opts.debug) printf("\tt1 fancy spread: \t%.3g s (%d subprobs)\n",timer.elapsedsec(), nb);
    }   // end of choice of which t1 spread type to use
    return 0;
};


// --------------------------------------------------------------------------
int interpSorted(BIGINT* sort_indices,BIGINT N1, BIGINT N2, BIGINT N3, 
		      FLT *data_uniform,BIGINT M, FLT *kx, FLT *ky, FLT *kz,
		      FLT *data_nonuniform, spread_opts opts, int did_sort){
  CNTime timer;
  int ndims = ndims_from_Ns(N1,N2,N3);
  int ns=opts.nspread;          // abbrev. for w, kernel width
  FLT ns2 = (FLT)ns/2;          // half spread width, used as stencil shift
  int nthr = MY_OMP_GET_MAX_THREADS();   // # threads to use to interp
  if (opts.nthreads>0)
    nthr = min(nthr,opts.nthreads);      // user override up to max avail
  if (opts.debug)
    printf("\tinterp %dD (M=%lld; N1=%lld,N2=%lld,N3=%lld; pir=%d), nthr=%d\n",ndims,(long long)M,(long long)N1,(long long)N2,(long long)N3,opts.pirange,nthr);

  timer.start();  
#pragma omp parallel num_threads(nthr)
  {
#define CHUNKSIZE 16     // Chunks of Type 2 targets (Ludvig found by expt)
    BIGINT jlist[CHUNKSIZE];
    FLT xjlist[CHUNKSIZE], yjlist[CHUNKSIZE], zjlist[CHUNKSIZE];
    FLT outbuf[2*CHUNKSIZE];
    // Kernels: static alloc is faster, so we do it for up to 3D...
    FLT kernel_args[3*MAX_NSPREAD];
    FLT kernel_values[3*MAX_NSPREAD];
    FLT *ker1 = kernel_values;
    FLT *ker2 = kernel_values + ns;
    FLT *ker3 = kernel_values + 2*ns;       

    // Loop over interpolation chunks
#pragma omp for schedule (dynamic,1000)  // assign threads to NU targ pts:
    for (BIGINT i=0; i<M; i+=CHUNKSIZE)  // main loop over NU targs, interp each from U
      {
        // Setup buffers for this chunk
        int bufsize = (i+CHUNKSIZE > M) ? M-i : CHUNKSIZE;
        for (int ibuf=0; ibuf<bufsize; ibuf++) {
          BIGINT j = sort_indices[i+ibuf];
          jlist[ibuf] = j;
	  xjlist[ibuf] = FOLDRESCALE(kx[j],N1,opts.pirange);
	  if(ndims >=2)
	    yjlist[ibuf] = FOLDRESCALE(ky[j],N2,opts.pirange);
	  if(ndims == 3)
	    zjlist[ibuf] = FOLDRESCALE(kz[j],N3,opts.pirange);                              
	}
      
    // Loop over targets in chunk
    for (int ibuf=0; ibuf<bufsize; ibuf++) {
      FLT xj = xjlist[ibuf];
      FLT yj = (ndims > 1) ? yjlist[ibuf] : 0;
      FLT zj = (ndims > 2) ? zjlist[ibuf] : 0;

      FLT *target = outbuf+2*ibuf;
        
      // coords (x,y,z), spread block corner index (i1,i2,i3) of current NU targ
      BIGINT i1=(BIGINT)std::ceil(xj-ns2); // leftmost grid index
      BIGINT i2= (ndims > 1) ? (BIGINT)std::ceil(yj-ns2) : 0; // min y grid index
      BIGINT i3= (ndims > 1) ? (BIGINT)std::ceil(zj-ns2) : 0; // min z grid index
     
      FLT x1=(FLT)i1-xj;           // shift of ker center, in [-w/2,-w/2+1]
      FLT x2= (ndims > 1) ? (FLT)i2-yj : 0 ;
      FLT x3= (ndims > 2)? (FLT)i3-zj : 0;

      // eval kernel values patch and use to interpolate from uniform data...
      if (!(opts.flags & TF_OMIT_SPREADING)) {

	  if (opts.kerevalmeth==0) {               // choose eval method
	    set_kernel_args(kernel_args, x1, opts);
	    if(ndims > 1)  set_kernel_args(kernel_args+ns, x2, opts);
	    if(ndims > 2)  set_kernel_args(kernel_args+2*ns, x3, opts);
	    
	    evaluate_kernel_vector(kernel_values, kernel_args, opts, ndims*ns);
	  }

	  else{
	    eval_kernel_vec_Horner(ker1,x1,ns,opts);
	    if (ndims > 1) eval_kernel_vec_Horner(ker2,x2,ns,opts);  
	    if (ndims > 2) eval_kernel_vec_Horner(ker3,x3,ns,opts);
	  }

	  switch(ndims){
	  case 1:
	    interp_line(target,data_uniform,ker1,i1,N1,ns);
	    break;
	  case 2:
	    interp_square(target,data_uniform,ker1,ker2,i1,i2,N1,N2,ns);
	    break;
	  case 3:
	    interp_cube(target,data_uniform,ker1,ker2,ker3,i1,i2,i3,N1,N2,N3,ns);
	    break;
	  default: //can't get here
	    break;
	     
	  }	 
      }
    } // end loop over targets in chunk
        
    // Copy result buffer to output array
    for (int ibuf=0; ibuf<bufsize; ibuf++) {
      BIGINT j = jlist[ibuf];
      data_nonuniform[2*j] = outbuf[2*ibuf];
      data_nonuniform[2*j+1] = outbuf[2*ibuf+1];              
    }         
        
      } // end NU targ loop
  } // end parallel section
  if (opts.debug) printf("\tt2 spreading loop: \t%.3g s\n",timer.elapsedsec());
  return 0;
};


int spreadinterpSorted(BIGINT* sort_indices,BIGINT N1, BIGINT N2, BIGINT N3, 
		      FLT *data_uniform,BIGINT M, FLT *kx, FLT *ky, FLT *kz,
		      FLT *data_nonuniform, spread_opts opts, int did_sort)
/* Logic to select the main spreading (dir=1) vs interpolation (dir=2) routine.
   See spreadinterp() above for inputs arguments and definitions.
   Return value should always be 0 (no error reporting).
   Split out by Melody Shih, Jun 2018; renamed Barnett 5/20/20.
*/
{
  if (opts.spread_direction==1)  // ========= direction 1 (spreading) =======
    spreadSorted(sort_indices, N1, N2, N3, data_uniform, M, kx, ky, kz, data_nonuniform, opts, did_sort);
  
  else           // ================= direction 2 (interpolation) ===========
    interpSorted(sort_indices, N1, N2, N3, data_uniform, M, kx, ky, kz, data_nonuniform, opts, did_sort);
  
  return 0;
}


///////////////////////////////////////////////////////////////////////////

int setup_spreader(spread_opts &opts, FLT eps, double upsampfac,
                   int kerevalmeth, int debug, int showwarn)
/* Initializes spreader kernel parameters given desired NUFFT tolerance eps,
   upsampling factor (=sigma in paper, or R in Dutt-Rokhlin), ker eval meth
   (either 0:exp(sqrt()), 1: Horner ppval), and some debug-level flags.
   Also sets all default options in spread_opts. See spread_opts.h for opts.
   See finufft.cpp:finufft_plan() for where upsampfac is set.
   Must call this before any kernel evals done, otherwise segfault likely.
   Returns:
     0  : success
     WARN_EPS_TOO_SMALL : requested eps cannot be achieved, but proceed with
                          best possible eps
     otherwise : failure (see codes in defs.h); spreading must not proceed
   Barnett 2017. debug, loosened eps logic 6/14/20.
*/
{
  if (upsampfac!=2.0 && upsampfac!=1.25) {   // nonstandard sigma
    if (kerevalmeth==1) {
      fprintf(stderr,"FINUFFT setup_spreader: nonstandard upsampfac=%.3g cannot be handled by kerevalmeth=1\n",upsampfac);
      return ERR_HORNER_WRONG_BETA;
    }
    if (upsampfac<=1.0) {       // no digits would result
      fprintf(stderr,"FINUFFT setup_spreader: error, upsampfac=%.3g is <=1.0\n",upsampfac);
      return ERR_UPSAMPFAC_TOO_SMALL;
    }
    // calling routine must abort on above errors, since opts is garbage!
    if (showwarn && upsampfac>4.0)
      fprintf(stderr,"FINUFFT setup_spreader warning: upsampfac=%.3g way too large to be beneficial.\n",upsampfac);
  }
    
  // write out default spread_opts (some overridden in setup_spreader_for_nufft)
  opts.spread_direction = 0;    // user should always set to 1 or 2 as desired
  opts.pirange = 1;             // user also should always set this
  opts.chkbnds = 0;
  opts.sort = 2;                // 2:auto-choice
  opts.kerpad = 0;              // affects only evaluate_kernel_vector
  opts.kerevalmeth = kerevalmeth;
  opts.upsampfac = upsampfac;
  opts.nthreads = 0;            // all avail
  opts.sort_threads = 0;        // 0:auto-choice
  opts.max_subproblem_size = (BIGINT)1e4;   // was larger (1e5 bit worse in 1D)
  opts.flags = 0;               // 0:no timing flags (>0 for experts only)
  opts.debug = 0;               // 0:no debug output

  int ns, ier = 0;  // Set kernel width w (aka ns, nspread) then copy to opts...
  if (eps<EPSILON) {            // safety; there's no hope of beating e_mach
    if (showwarn)
      fprintf(stderr,"%s warning: increasing tol=%.3g to eps_mach=%.3g.\n",__func__,(double)eps,(double)EPSILON);
    eps = EPSILON;              // only changes local copy (not any opts)
    ier = WARN_EPS_TOO_SMALL;
  }
  if (upsampfac==2.0)           // standard sigma (see SISC paper)
    ns = std::ceil(-log10(eps/(FLT)10.0));          // 1 digit per power of 10
  else                          // custom sigma
    ns = std::ceil(-log(eps) / (PI*sqrt(1.0-1.0/upsampfac)));  // formula, gam=1
  ns = max(2,ns);               // (we don't have ns=1 version yet)
  if (ns>MAX_NSPREAD) {         // clip to fit allocated arrays, Horner rules
    if (showwarn)
      fprintf(stderr,"%s warning: at upsampfac=%.3g, tol=%.3g would need kernel width ns=%d; clipping to max %d.\n",__func__,
              upsampfac,(double)eps,ns,MAX_NSPREAD);
    ns = MAX_NSPREAD;
    ier = WARN_EPS_TOO_SMALL;
  }
  opts.nspread = ns;

  // setup for reference kernel eval (via formula): select beta width param...
  // (even when kerevalmeth=1, this ker eval needed for FTs in onedim_*_kernel)
  opts.ES_halfwidth=(FLT)ns/2;   // constants to help (see below routines)
  opts.ES_c = 4.0/(FLT)(ns*ns);
  FLT betaoverns = 2.30;         // gives decent betas for default sigma=2.0
  if (ns==2) betaoverns = 2.20;  // some small-width tweaks...
  if (ns==3) betaoverns = 2.26;
  if (ns==4) betaoverns = 2.38;
  if (upsampfac!=2.0) {          // again, override beta for custom sigma
    FLT gamma=0.97;              // must match devel/gen_all_horner_C_code.m !
    betaoverns = gamma*PI*(1.0-1.0/(2*upsampfac));  // formula based on cutoff
  }
  opts.ES_beta = betaoverns * (FLT)ns;    // set the kernel beta parameter
  if (debug)
    printf("%s (kerevalmeth=%d) eps=%.3g sigma=%.3g: chose ns=%d beta=%.3g\n",__func__,kerevalmeth,(double)eps,upsampfac,ns,(double)opts.ES_beta);
  
  return ier;
}

FLT evaluate_kernel(FLT x, const spread_opts &opts)
/* ES ("exp sqrt") kernel evaluation at single real argument:
      phi(x) = exp(beta.sqrt(1 - (2x/n_s)^2)),    for |x| < nspread/2
   related to an asymptotic approximation to the Kaiser--Bessel, itself an
   approximation to prolate spheroidal wavefunction (PSWF) of order 0.
   This is the "reference implementation", used by eg finufft/onedim_* 2/17/17
*/
{
  if (abs(x)>=opts.ES_halfwidth)
    // if spreading/FT careful, shouldn't need this if, but causes no speed hit
    return 0.0;
  else
    return exp(opts.ES_beta * sqrt(1.0 - opts.ES_c*x*x));
}

FLT evaluate_kernel_noexp(FLT x, const spread_opts &opts)
// Version of the above just for timing purposes - gives wrong answer!!!
{
  if (abs(x)>=opts.ES_halfwidth)
    return 0.0;
  else {
    FLT s = sqrt(1.0 - opts.ES_c*x*x);
    //  return sinh(opts.ES_beta * s)/s; // roughly, backward K-B kernel of NFFT
        return opts.ES_beta * s;
  }
}

static inline void set_kernel_args(FLT *args, FLT x, const spread_opts& opts)
// Fills vector args[] with kernel arguments x, x+1, ..., x+ns-1.
// needed for the vectorized kernel eval of Ludvig af K.
{
  int ns=opts.nspread;
  for (int i=0; i<ns; i++)
    args[i] = x + (FLT) i;
}

static inline void evaluate_kernel_vector(FLT *ker, FLT *args, const spread_opts& opts, const int N)
/* Evaluate ES kernel for a vector of N arguments; by Ludvig af K.
   If opts.kerpad true, args and ker must be allocated for Npad, and args is
   written to (to pad to length Npad), only first N outputs are correct.
   Barnett 4/24/18 option to pad to mult of 4 for better SIMD vectorization.

   Obsolete (replaced by Horner), but keep around for experimentation since
   works for arbitrary beta. Formula must match reference implementation. */
{
  FLT b = opts.ES_beta;
  FLT c = opts.ES_c;
  if (!(opts.flags & TF_OMIT_EVALUATE_KERNEL)) {
    // Note (by Ludvig af K): Splitting kernel evaluation into two loops
    // seems to benefit auto-vectorization.
    // gcc 5.4 vectorizes first loop; gcc 7.2 vectorizes both loops
    int Npad = N;
    if (opts.kerpad) {        // since always same branch, no speed hit
      Npad = 4*(1+(N-1)/4);   // pad N to mult of 4; help i7 GCC, not xeon
      for (int i=N;i<Npad;++i)    // pad with 1-3 zeros for safe eval
	args[i] = 0.0;
    }
    for (int i = 0; i < Npad; i++) { // Loop 1: Compute exponential arguments
      ker[i] = b * sqrt(1.0 - c*args[i]*args[i]);
    }
    if (!(opts.flags & TF_OMIT_EVALUATE_EXPONENTIAL))
      for (int i = 0; i < Npad; i++) // Loop 2: Compute exponentials
	ker[i] = exp(ker[i]);
  } else {
    for (int i = 0; i < N; i++)             // dummy for timing only
      ker[i] = 1.0;
  }
  // Separate check from arithmetic (Is this really needed? doesn't slow down)
  for (int i = 0; i < N; i++)
    if (abs(args[i])>=opts.ES_halfwidth) ker[i] = 0.0;
}

static inline void eval_kernel_vec_Horner(FLT *ker, const FLT x, const int w,
					  const spread_opts &opts)
/* Fill ker[] with Horner piecewise poly approx to [-w/2,w/2] ES kernel eval at
   x_j = x + j,  for j=0,..,w-1.  Thus x in [-w/2,-w/2+1].   w is aka ns.
   This is the current evaluation method, since it's faster (except i7 w=16).
   Two upsampfacs implemented. Params must match ref formula. Barnett 4/24/18 */
{
  if (!(opts.flags & TF_OMIT_EVALUATE_KERNEL)) {
    FLT z = 2*x + w - 1.0;         // scale so local grid offset z in [-1,1]
    // insert the auto-generated code which expects z, w args, writes to ker...
    if (opts.upsampfac==2.0) {     // floating point equality is fine here
// Code generated by gen_all_horner_C_code.m in finufft/devel
// Authors: Alex Barnett & Ludvig af Klinteberg.
// (C) The Simons Foundation, Inc.
  if (w==2) {
    FLT c0[] = {4.5147043243215315E+01, 4.5147043243215300E+01, 0.0000000000000000E+00, 0.0000000000000000E+00};
    FLT c1[] = {5.7408070938221300E+01, -5.7408070938221293E+01, 0.0000000000000000E+00, 0.0000000000000000E+00};
    FLT c2[] = {-1.8395117920046484E+00, -1.8395117920046560E+00, 0.0000000000000000E+00, 0.0000000000000000E+00};
    FLT c3[] = {-2.0382426253182082E+01, 2.0382426253182086E+01, 0.0000000000000000E+00, 0.0000000000000000E+00};
    FLT c4[] = {-2.0940804433577420E+00, -2.0940804433577389E+00, 0.0000000000000000E+00, 0.0000000000000000E+00};
    for (int i=0; i<4; i++) ker[i] = c0[i] + z*(c1[i] + z*(c2[i] + z*(c3[i] + z*(c4[i]))));
  } else if (w==3) {
    FLT c0[] = {1.5653991189315119E+02, 8.8006872410780295E+02, 1.5653991189967152E+02, 0.0000000000000000E+00};
    FLT c1[] = {3.1653018869611077E+02, 7.4325702843759617E-14, -3.1653018868907071E+02, 0.0000000000000000E+00};
    FLT c2[] = {1.7742692790454484E+02, -3.3149255274727801E+02, 1.7742692791117119E+02, 0.0000000000000000E+00};
    FLT c3[] = {-1.5357716116473156E+01, 9.5071486252033243E-15, 1.5357716122720193E+01, 0.0000000000000000E+00};
    FLT c4[] = {-3.7757583061523668E+01, 5.3222970968867315E+01, -3.7757583054647384E+01, 0.0000000000000000E+00};
    FLT c5[] = {-3.9654011076088804E+00, 1.8062124448285358E-13, 3.9654011139270540E+00, 0.0000000000000000E+00};
    for (int i=0; i<4; i++) ker[i] = c0[i] + z*(c1[i] + z*(c2[i] + z*(c3[i] + z*(c4[i] + z*(c5[i])))));
  } else if (w==4) {
    FLT c0[] = {5.4284366850213200E+02, 1.0073871433088398E+04, 1.0073871433088396E+04, 5.4284366850213223E+02};
    FLT c1[] = {1.4650917259256939E+03, 6.1905285583602863E+03, -6.1905285583602881E+03, -1.4650917259256937E+03};
    FLT c2[] = {1.4186910680718345E+03, -1.3995339862725591E+03, -1.3995339862725598E+03, 1.4186910680718347E+03};
    FLT c3[] = {5.1133995502497419E+02, -1.4191608683682996E+03, 1.4191608683682998E+03, -5.1133995502497424E+02};
    FLT c4[] = {-4.8293622641174039E+01, 3.9393732546135226E+01, 3.9393732546135816E+01, -4.8293622641174061E+01};
    FLT c5[] = {-7.8386867802392288E+01, 1.4918904800408930E+02, -1.4918904800408751E+02, 7.8386867802392359E+01};
    FLT c6[] = {-1.0039212571700894E+01, 5.0626747735616746E+00, 5.0626747735625512E+00, -1.0039212571700640E+01};
    for (int i=0; i<4; i++) ker[i] = c0[i] + z*(c1[i] + z*(c2[i] + z*(c3[i] + z*(c4[i] + z*(c5[i] + z*(c6[i]))))));
  } else if (w==5) {
    FLT c0[] = {9.9223677575398392E+02, 3.7794697666613320E+04, 9.8715771010760494E+04, 3.7794697666613283E+04, 9.9223677575398403E+02, 0.0000000000000000E+00, 0.0000000000000000E+00, 0.0000000000000000E+00};
    FLT c1[] = {3.0430174925083825E+03, 3.7938404259811403E+04, -1.1842989705877139E-11, -3.7938404259811381E+04, -3.0430174925083829E+03, 0.0000000000000000E+00, 0.0000000000000000E+00, 0.0000000000000000E+00};
    FLT c2[] = {3.6092689177271222E+03, 7.7501368899498666E+03, -2.2704627332475000E+04, 7.7501368899498730E+03, 3.6092689177271218E+03, 0.0000000000000000E+00, 0.0000000000000000E+00, 0.0000000000000000E+00};
    FLT c3[] = {1.9990077310495396E+03, -3.8875294641277296E+03, 9.7116927320010791E-12, 3.8875294641277369E+03, -1.9990077310495412E+03, 0.0000000000000000E+00, 0.0000000000000000E+00, 0.0000000000000000E+00};
    FLT c4[] = {4.0071733590403869E+02, -1.5861137916762602E+03, 2.3839858699098645E+03, -1.5861137916762643E+03, 4.0071733590403909E+02, 0.0000000000000000E+00, 0.0000000000000000E+00, 0.0000000000000000E+00};
    FLT c5[] = {-9.1301168206167262E+01, 1.2316471075214675E+02, 2.0698495299948402E-11, -1.2316471075214508E+02, 9.1301168206167233E+01, 0.0000000000000000E+00, 0.0000000000000000E+00, 0.0000000000000000E+00};
    FLT c6[] = {-5.5339722671223846E+01, 1.1960590540261879E+02, -1.5249941358311668E+02, 1.1960590540262307E+02, -5.5339722671223605E+01, 0.0000000000000000E+00, 0.0000000000000000E+00, 0.0000000000000000E+00};
    FLT c7[] = {-3.3762488150353924E+00, 2.2839981872948751E+00, 7.1884725699454154E-12, -2.2839981872943818E+00, 3.3762488150341459E+00, 0.0000000000000000E+00, 0.0000000000000000E+00, 0.0000000000000000E+00};
    for (int i=0; i<8; i++) ker[i] = c0[i] + z*(c1[i] + z*(c2[i] + z*(c3[i] + z*(c4[i] + z*(c5[i] + z*(c6[i] + z*(c7[i])))))));
  } else if (w==6) {
    FLT c0[] = {2.0553833234911876E+03, 1.5499537739913128E+05, 8.1177907023291115E+05, 8.1177907023291173E+05, 1.5499537739913136E+05, 2.0553833235005691E+03, 0.0000000000000000E+00, 0.0000000000000000E+00};
    FLT c1[] = {7.1269776034442639E+03, 2.0581923258843314E+05, 3.1559612614917674E+05, -3.1559612614917627E+05, -2.0581923258843317E+05, -7.1269776034341394E+03, 0.0000000000000000E+00, 0.0000000000000000E+00};
    FLT c2[] = {1.0023404568475091E+04, 9.0916650498360192E+04, -1.0095927514054619E+05, -1.0095927514054628E+05, 9.0916650498360177E+04, 1.0023404568484635E+04, 0.0000000000000000E+00, 0.0000000000000000E+00};
    FLT c3[] = {7.2536109410387417E+03, 4.8347162752602981E+03, -5.0512736602018522E+04, 5.0512736602018478E+04, -4.8347162752603008E+03, -7.2536109410297540E+03, 0.0000000000000000E+00, 0.0000000000000000E+00};
    FLT c4[] = {2.7021878300949752E+03, -7.8773465553972646E+03, 5.2105876478342780E+03, 5.2105876478343343E+03, -7.8773465553972710E+03, 2.7021878301048723E+03, 0.0000000000000000E+00, 0.0000000000000000E+00};
    FLT c5[] = {3.2120291706547636E+02, -1.8229189469936762E+03, 3.7928113414429808E+03, -3.7928113414427025E+03, 1.8229189469937312E+03, -3.2120291705638243E+02, 0.0000000000000000E+00, 0.0000000000000000E+00};
    FLT c6[] = {-1.2051267090537374E+02, 2.2400507411399673E+02, -1.2506575852541796E+02, -1.2506575852521925E+02, 2.2400507411398695E+02, -1.2051267089640181E+02, 0.0000000000000000E+00, 0.0000000000000000E+00};
    FLT c7[] = {-4.5977202613350237E+01, 1.1536880606853076E+02, -1.7819720186493959E+02, 1.7819720186497622E+02, -1.1536880606854736E+02, 4.5977202622148909E+01, 0.0000000000000000E+00, 0.0000000000000000E+00};
    FLT c8[] = {-1.5631081288842275E+00, 7.1037430591266115E-01, -6.9838401121429056E-02, -6.9838401186476856E-02, 7.1037430589285400E-01, -1.5631081203754575E+00, 0.0000000000000000E+00, 0.0000000000000000E+00};
    for (int i=0; i<8; i++) ker[i] = c0[i] + z*(c1[i] + z*(c2[i] + z*(c3[i] + z*(c4[i] + z*(c5[i] + z*(c6[i] + z*(c7[i] + z*(c8[i]))))))));
  } else if (w==7) {
    FLT c0[] = {3.9948351830487481E+03, 5.4715865608590771E+05, 5.0196413492771760E+06, 9.8206709220713247E+06, 5.0196413492771825E+06, 5.4715865608590783E+05, 3.9948351830642519E+03, 0.0000000000000000E+00};
    FLT c1[] = {1.5290160332974696E+04, 8.7628248584320408E+05, 3.4421061790934438E+06, -2.6908159596373561E-10, -3.4421061790934461E+06, -8.7628248584320408E+05, -1.5290160332958067E+04, 0.0000000000000000E+00};
    FLT c2[] = {2.4458227486779251E+04, 5.3904618484139396E+05, 2.4315566181017534E+05, -1.6133959371974322E+06, 2.4315566181017453E+05, 5.3904618484139396E+05, 2.4458227486795113E+04, 0.0000000000000000E+00};
    FLT c3[] = {2.1166189345881645E+04, 1.3382732160223130E+05, -3.3113450969689694E+05, 6.9013724510092140E-10, 3.3113450969689724E+05, -1.3382732160223136E+05, -2.1166189345866893E+04, 0.0000000000000000E+00};
    FLT c4[] = {1.0542795672344864E+04, -7.0739172265098678E+03, -6.5563293056049893E+04, 1.2429734005960064E+05, -6.5563293056049602E+04, -7.0739172265098332E+03, 1.0542795672361213E+04, 0.0000000000000000E+00};
    FLT c5[] = {2.7903491906228419E+03, -1.0975382873973093E+04, 1.3656979541144799E+04, 7.7346408577822045E-10, -1.3656979541143772E+04, 1.0975382873973256E+04, -2.7903491906078298E+03, 0.0000000000000000E+00};
    FLT c6[] = {1.6069721418053300E+02, -1.5518707872251393E+03, 4.3634273936642621E+03, -5.9891976420595174E+03, 4.3634273936642730E+03, -1.5518707872251064E+03, 1.6069721419533221E+02, 0.0000000000000000E+00};
    FLT c7[] = {-1.2289277373867256E+02, 2.8583630927743314E+02, -2.8318194617327981E+02, 6.9043515551118249E-10, 2.8318194617392436E+02, -2.8583630927760140E+02, 1.2289277375319763E+02, 0.0000000000000000E+00};
    FLT c8[] = {-3.2270164914249058E+01, 9.1892112257581346E+01, -1.6710678096334209E+02, 2.0317049305432383E+02, -1.6710678096383771E+02, 9.1892112257416159E+01, -3.2270164900224913E+01, 0.0000000000000000E+00};
    FLT c9[] = {-1.4761409685186277E-01, -9.1862771280377487E-01, 1.2845147741777752E+00, 5.6547359492808854E-10, -1.2845147728310689E+00, 9.1862771293147971E-01, 1.4761410890866353E-01, 0.0000000000000000E+00};
    for (int i=0; i<8; i++) ker[i] = c0[i] + z*(c1[i] + z*(c2[i] + z*(c3[i] + z*(c4[i] + z*(c5[i] + z*(c6[i] + z*(c7[i] + z*(c8[i] + z*(c9[i])))))))));
  } else if (w==8) {
    FLT c0[] = {7.3898000697447915E+03, 1.7297637497600035E+06, 2.5578341605285794E+07, 8.4789650417103335E+07, 8.4789650417103350E+07, 2.5578341605285816E+07, 1.7297637497600049E+06, 7.3898000697447915E+03};
    FLT c1[] = {3.0719636811267599E+04, 3.1853145713323927E+06, 2.3797981861403696E+07, 2.4569731244678464E+07, -2.4569731244678471E+07, -2.3797981861403704E+07, -3.1853145713323941E+06, -3.0719636811267606E+04};
    FLT c2[] = {5.4488498478251728E+04, 2.4101183255475131E+06, 6.4554051283428287E+06, -8.9200440393090546E+06, -8.9200440393090583E+06, 6.4554051283428324E+06, 2.4101183255475126E+06, 5.4488498478251728E+04};
    FLT c3[] = {5.3926359802542116E+04, 9.0469037926849292E+05, -6.0897036277696118E+05, -3.0743852105799988E+06, 3.0743852105800058E+06, 6.0897036277696711E+05, -9.0469037926849339E+05, -5.3926359802542138E+04};
    FLT c4[] = {3.2444118016247590E+04, 1.3079802224392134E+05, -5.8652889370129269E+05, 4.2333306008151924E+05, 4.2333306008152053E+05, -5.8652889370128722E+05, 1.3079802224392109E+05, 3.2444118016247590E+04};
    FLT c5[] = {1.1864306345505294E+04, -2.2700360645707988E+04, -5.0713607251414309E+04, 1.8308704458211688E+05, -1.8308704458210632E+05, 5.0713607251413123E+04, 2.2700360645707628E+04, -1.1864306345505294E+04};
    FLT c6[] = {2.2812256770903232E+03, -1.1569135767377773E+04, 2.0942387020798891E+04, -1.1661592834945191E+04, -1.1661592834940149E+04, 2.0942387020801420E+04, -1.1569135767377924E+04, 2.2812256770903286E+03};
    FLT c7[] = {8.5503535636821422E+00, -9.7513976461238224E+02, 3.8242995179171526E+03, -6.9201295567267280E+03, 6.9201295567248662E+03, -3.8242995179155446E+03, 9.7513976461209836E+02, -8.5503535637013552E+00};
    FLT c8[] = {-1.0230637348345023E+02, 2.8246898554269114E+02, -3.8638201738139219E+02, 1.9106407993320320E+02, 1.9106407993289886E+02, -3.8638201738492717E+02, 2.8246898554219217E+02, -1.0230637348345138E+02};
    FLT c9[] = {-1.9200143062947848E+01, 6.1692257626706223E+01, -1.2981109187842989E+02, 1.8681284210471688E+02, -1.8681284209654376E+02, 1.2981109187880142E+02, -6.1692257626845532E+01, 1.9200143062947120E+01};
    FLT c10[] = {3.7894993760177598E-01, -1.7334408836731494E+00, 2.5271184057877303E+00, -1.2600963971824484E+00, -1.2600963917834651E+00, 2.5271184069685657E+00, -1.7334408840526812E+00, 3.7894993760636758E-01};
    for (int i=0; i<8; i++) ker[i] = c0[i] + z*(c1[i] + z*(c2[i] + z*(c3[i] + z*(c4[i] + z*(c5[i] + z*(c6[i] + z*(c7[i] + z*(c8[i] + z*(c9[i] + z*(c10[i]))))))))));
  } else if (w==9) {
    FLT c0[] = {1.3136365370186100E+04, 5.0196413492771806E+06, 1.1303327711722563E+08, 5.8225443924996686E+08, 9.7700272582690656E+08, 5.8225443924996758E+08, 1.1303327711722568E+08, 5.0196413492772207E+06, 1.3136365370186135E+04, 0.0000000000000000E+00, 0.0000000000000000E+00, 0.0000000000000000E+00};
    FLT c1[] = {5.8623313038274340E+04, 1.0326318537280345E+07, 1.2898448324824864E+08, 3.0522863709830385E+08, -3.9398045056223735E-08, -3.0522863709830391E+08, -1.2898448324824864E+08, -1.0326318537280388E+07, -5.8623313038274347E+04, 0.0000000000000000E+00, 0.0000000000000000E+00, 0.0000000000000000E+00};
    FLT c2[] = {1.1335001341875963E+05, 9.0726133144784812E+06, 5.3501544534038112E+07, -2.6789524644146336E+05, -1.2483923718899371E+08, -2.6789524644172983E+05, 5.3501544534038112E+07, 9.0726133144785129E+06, 1.1335001341875960E+05, 0.0000000000000000E+00, 0.0000000000000000E+00, 0.0000000000000000E+00};
    FLT c3[] = {1.2489113703229747E+05, 4.3035547171861930E+06, 6.3021978510598792E+06, -2.6014941986659057E+07, 6.0417403157325170E-08, 2.6014941986659389E+07, -6.3021978510598652E+06, -4.3035547171862079E+06, -1.2489113703229751E+05, 0.0000000000000000E+00, 0.0000000000000000E+00, 0.0000000000000000E+00};
    FLT c4[] = {8.6425493435991244E+04, 1.0891182836653308E+06, -2.0713033564200639E+06, -2.8994941183506218E+06, 7.5905338661205899E+06, -2.8994941183505375E+06, -2.0713033564200667E+06, 1.0891182836653353E+06, 8.6425493435991288E+04, 0.0000000000000000E+00, 0.0000000000000000E+00, 0.0000000000000000E+00};
    FLT c5[] = {3.8657354724013814E+04, 7.9936390113331305E+04, -7.0458265546791907E+05, 1.0151095605715880E+06, 1.2138090419648379E-07, -1.0151095605717725E+06, 7.0458265546794771E+05, -7.9936390113331567E+04, -3.8657354724013821E+04, 0.0000000000000000E+00, 0.0000000000000000E+00, 0.0000000000000000E+00};
    FLT c6[] = {1.0779131453134638E+04, -3.3466718311300596E+04, -1.3245366619006139E+04, 1.8238470515353698E+05, -2.9285656292977190E+05, 1.8238470515350526E+05, -1.3245366619000662E+04, -3.3466718311299621E+04, 1.0779131453134616E+04, 0.0000000000000000E+00, 0.0000000000000000E+00, 0.0000000000000000E+00};
    FLT c7[] = {1.4992527030548456E+03, -9.7024371533891372E+03, 2.3216330734057381E+04, -2.3465262819040818E+04, 5.3299736484284360E-08, 2.3465262819251962E+04, -2.3216330734049119E+04, 9.7024371533890644E+03, -1.4992527030548747E+03, 0.0000000000000000E+00, 0.0000000000000000E+00, 0.0000000000000000E+00};
    FLT c8[] = {-7.9857427421129714E+01, -4.0585588534807385E+02, 2.6054813773472697E+03, -6.1806593581075495E+03, 8.0679596874001718E+03, -6.1806593581869265E+03, 2.6054813773147021E+03, -4.0585588535363172E+02, -7.9857427421126204E+01, 0.0000000000000000E+00, 0.0000000000000000E+00, 0.0000000000000000E+00};
    FLT c9[] = {-7.1572272057937070E+01, 2.2785637019511205E+02, -3.9109820765665262E+02, 3.3597424711470910E+02, 1.0596763818009852E-07, -3.3597424723359080E+02, 3.9109820766854079E+02, -2.2785637019009673E+02, 7.1572272057939983E+01, 0.0000000000000000E+00, 0.0000000000000000E+00, 0.0000000000000000E+00};
    FLT c10[] = {-9.8886360698074700E+00, 3.5359026949867051E+01, -8.5251867715709949E+01, 1.4285748012617628E+02, -1.6935269668779691E+02, 1.4285748010331625E+02, -8.5251867711661305E+01, 3.5359026944299828E+01, -9.8886360698207305E+00, 0.0000000000000000E+00, 0.0000000000000000E+00, 0.0000000000000000E+00};
    for (int i=0; i<12; i++) ker[i] = c0[i] + z*(c1[i] + z*(c2[i] + z*(c3[i] + z*(c4[i] + z*(c5[i] + z*(c6[i] + z*(c7[i] + z*(c8[i] + z*(c9[i] + z*(c10[i]))))))))));
  } else if (w==10) {
    FLT c0[] = {2.2594586605749264E+04, 1.3595989066786593E+07, 4.4723032442444897E+08, 3.3781755837397518E+09, 8.6836783895849819E+09, 8.6836783895849762E+09, 3.3781755837397494E+09, 4.4723032442444897E+08, 1.3595989066786474E+07, 2.2594586605749344E+04, 0.0000000000000000E+00, 0.0000000000000000E+00};
    FLT c1[] = {1.0729981697645642E+05, 3.0651490267742988E+07, 5.9387966085130465E+08, 2.4434902657508330E+09, 2.0073077861288922E+09, -2.0073077861288943E+09, -2.4434902657508330E+09, -5.9387966085130453E+08, -3.0651490267742816E+07, -1.0729981697645638E+05, 0.0000000000000000E+00, 0.0000000000000000E+00};
    FLT c2[] = {2.2340399734184606E+05, 3.0258214643190462E+07, 3.1512411458738232E+08, 4.3618276932319808E+08, -7.8178848450497293E+08, -7.8178848450497019E+08, 4.3618276932319826E+08, 3.1512411458738232E+08, 3.0258214643190313E+07, 2.2340399734184548E+05, 0.0000000000000000E+00, 0.0000000000000000E+00};
    FLT c3[] = {2.6917433004353486E+05, 1.6875651476661228E+07, 7.4664745481963441E+07, -9.5882157211118385E+07, -2.0622994435532519E+08, 2.0622994435532743E+08, 9.5882157211118177E+07, -7.4664745481963515E+07, -1.6875651476661161E+07, -2.6917433004353428E+05, 0.0000000000000000E+00, 0.0000000000000000E+00};
    FLT c4[] = {2.0818422772177903E+05, 5.6084730690362519E+06, 1.4435118192351763E+06, -4.0063869969544649E+07, 3.2803674392747045E+07, 3.2803674392746095E+07, -4.0063869969546899E+07, 1.4435118192351642E+06, 5.6084730690362034E+06, 2.0818422772177853E+05, 0.0000000000000000E+00, 0.0000000000000000E+00};
    FLT c5[] = {1.0781139496011091E+05, 9.9202615851199068E+05, -3.3266265543962116E+06, -4.8557049011479173E+05, 1.0176155522772279E+07, -1.0176155522772269E+07, 4.8557049011678610E+05, 3.3266265543963453E+06, -9.9202615851196018E+05, -1.0781139496011072E+05, 0.0000000000000000E+00, 0.0000000000000000E+00};
    FLT c6[] = {3.7380102688153558E+04, 1.2716675000355666E+04, -6.2163527451774501E+05, 1.4157962667184104E+06, -8.4419693137680157E+05, -8.4419693137743860E+05, 1.4157962667189445E+06, -6.2163527451771160E+05, 1.2716675000340010E+04, 3.7380102688153442E+04, 0.0000000000000000E+00, 0.0000000000000000E+00};
    FLT c7[] = {8.1238936393894646E+03, -3.4872365530450072E+04, 2.3913680325196314E+04, 1.2428850301830019E+05, -3.2158255329716846E+05, 3.2158255329951923E+05, -1.2428850301867779E+05, -2.3913680325277423E+04, 3.4872365530457188E+04, -8.1238936393894255E+03, 0.0000000000000000E+00, 0.0000000000000000E+00};
    FLT c8[] = {7.8515926628982663E+02, -6.6607899119372642E+03, 2.0167398338513311E+04, -2.8951401344519112E+04, 1.4622828142848679E+04, 1.4622828143544031E+04, -2.8951401346900999E+04, 2.0167398338398041E+04, -6.6607899119505255E+03, 7.8515926628967964E+02, 0.0000000000000000E+00, 0.0000000000000000E+00};
    FLT c9[] = {-1.0147176570537010E+02, -3.5304284185385157E+01, 1.3576976854876134E+03, -4.3921059353471856E+03, 7.3232085271125388E+03, -7.3232085273978546E+03, 4.3921059367737662E+03, -1.3576976854043962E+03, 3.5304284185385157E+01, 1.0147176570550941E+02, 0.0000000000000000E+00, 0.0000000000000000E+00};
    FLT c10[] = {-4.3161545259389186E+01, 1.5498490981579428E+02, -3.1771250774232175E+02, 3.7215448796427023E+02, -1.7181762832770994E+02, -1.7181763036843782E+02, 3.7215448789408123E+02, -3.1771250773692140E+02, 1.5498490982186786E+02, -4.3161545259547800E+01, 0.0000000000000000E+00, 0.0000000000000000E+00};
    FLT c11[] = {-4.2916172038214198E+00, 1.7402146071148604E+01, -4.7947588069135868E+01, 9.2697698088029625E+01, -1.2821427596894478E+02, 1.2821427705670308E+02, -9.2697698297776569E+01, 4.7947588093524907E+01, -1.7402146074502035E+01, 4.2916172038452141E+00, 0.0000000000000000E+00, 0.0000000000000000E+00};
    for (int i=0; i<12; i++) ker[i] = c0[i] + z*(c1[i] + z*(c2[i] + z*(c3[i] + z*(c4[i] + z*(c5[i] + z*(c6[i] + z*(c7[i] + z*(c8[i] + z*(c9[i] + z*(c10[i] + z*(c11[i])))))))))));
  } else if (w==11) {
    FLT c0[] = {3.7794653219809625E+04, 3.4782300224660739E+07, 1.6188020733727551E+09, 1.7196758809615005E+10, 6.3754384857724617E+10, 9.7196447559193497E+10, 6.3754384857724617E+10, 1.7196758809614998E+10, 1.6188020733727560E+09, 3.4782300224660769E+07, 3.7794653219808984E+04, 0.0000000000000000E+00};
    FLT c1[] = {1.8969206922085886E+05, 8.4769319065313652E+07, 2.4230555767723408E+09, 1.5439732722639101E+10, 2.7112836839612309E+10, 2.5609833368650835E-06, -2.7112836839612328E+10, -1.5439732722639105E+10, -2.4230555767723408E+09, -8.4769319065313682E+07, -1.8969206922085711E+05, 0.0000000000000000E+00};
    FLT c2[] = {4.2138380313901440E+05, 9.2050522922791913E+07, 1.5259983101266613E+09, 4.7070559561237173E+09, -1.2448027572952359E+09, -1.0161446790279301E+10, -1.2448027572952316E+09, 4.7070559561237268E+09, 1.5259983101266615E+09, 9.2050522922791913E+07, 4.2138380313901149E+05, 0.0000000000000000E+00};
    FLT c3[] = {5.4814313598122005E+05, 5.8085130777589552E+07, 4.9484006166551048E+08, 1.6222124676640952E+08, -2.0440440381345339E+09, 9.1416457449079640E-06, 2.0440440381345336E+09, -1.6222124676640788E+08, -4.9484006166551071E+08, -5.8085130777589560E+07, -5.4814313598121714E+05, 0.0000000000000000E+00};
    FLT c4[] = {4.6495183529254980E+05, 2.3067199578027144E+07, 6.9832590192482382E+07, -2.2024799260683522E+08, -1.2820270942588677E+08, 5.1017181199129778E+08, -1.2820270942588474E+08, -2.2024799260683942E+08, 6.9832590192482322E+07, 2.3067199578027155E+07, 4.6495183529254742E+05, 0.0000000000000000E+00};
    FLT c5[] = {2.7021781043532980E+05, 5.6764510325100143E+06, -5.5650761736748898E+06, -3.9907385617900200E+07, 7.2453390663687646E+07, 1.2300109686762266E-05, -7.2453390663684472E+07, 3.9907385617899075E+07, 5.5650761736749066E+06, -5.6764510325099993E+06, -2.7021781043532846E+05, 0.0000000000000000E+00};
    FLT c6[] = {1.0933249308680627E+05, 6.9586821127987828E+05, -3.6860240321937902E+06, 2.7428169457736355E+06, 8.3392008440593518E+06, -1.6402201025046850E+07, 8.3392008440698013E+06, 2.7428169457778852E+06, -3.6860240321937371E+06, 6.9586821127989423E+05, 1.0933249308680571E+05, 0.0000000000000000E+00};
    FLT c7[] = {3.0203516161820498E+04, -3.6879059542768438E+04, -4.1141031216788280E+05, 1.4111389975267777E+06, -1.5914376635331670E+06, 9.4095582602103753E-06, 1.5914376635379130E+06, -1.4111389975247320E+06, 4.1141031216776522E+05, 3.6879059542750314E+04, -3.0203516161820549E+04, 0.0000000000000000E+00};
    FLT c8[] = {5.1670143574922731E+03, -2.8613147115372190E+04, 4.3560195427081359E+04, 4.8438679582765450E+04, -2.5856630639231802E+05, 3.7994883866738499E+05, -2.5856630640319458E+05, 4.8438679579510936E+04, 4.3560195426766244E+04, -2.8613147115376054E+04, 5.1670143574922913E+03, 0.0000000000000000E+00};
    FLT c9[] = {3.0888018539740131E+02, -3.7949446187471626E+03, 1.4313303204988082E+04, -2.6681600235594462E+04, 2.3856005166166615E+04, 8.6424601730164351E-06, -2.3856005155895236E+04, 2.6681600234453199E+04, -1.4313303205083188E+04, 3.7949446187583080E+03, -3.0888018539728523E+02, 0.0000000000000000E+00};
    FLT c10[] = {-8.3747489794189363E+01, 1.1948077479405792E+02, 4.8528498015072080E+02, -2.5024391114755094E+03, 5.3511195318669425E+03, -6.7655484107390166E+03, 5.3511195362291774E+03, -2.5024391131167667E+03, 4.8528498019392708E+02, 1.1948077480620087E+02, -8.3747489794426258E+01, 0.0000000000000000E+00};
    FLT c11[] = {-2.2640047135517630E+01, 9.0840898563949466E+01, -2.1597187544386938E+02, 3.1511229111443720E+02, -2.4856617998395282E+02, 6.1683918215190516E-06, 2.4856618439352349E+02, -3.1511228757800421E+02, 2.1597187557069353E+02, -9.0840898570046704E+01, 2.2640047135565219E+01, 0.0000000000000000E+00};
    FLT c12[] = {-1.6306382886201207E+00, 7.3325946591320434E+00, -2.3241017682854558E+01, 5.1715494398901185E+01, -8.2673000279130790E+01, 9.6489719151212370E+01, -8.2673010381149226E+01, 5.1715494328769353E+01, -2.3241018024860580E+01, 7.3325946448852415E+00, -1.6306382886460551E+00, 0.0000000000000000E+00};
    for (int i=0; i<12; i++) ker[i] = c0[i] + z*(c1[i] + z*(c2[i] + z*(c3[i] + z*(c4[i] + z*(c5[i] + z*(c6[i] + z*(c7[i] + z*(c8[i] + z*(c9[i] + z*(c10[i] + z*(c11[i] + z*(c12[i]))))))))))));
  } else if (w==12) {
    FLT c0[] = {6.1722991679852908E+04, 8.4789650417103648E+07, 5.4431675199498701E+09, 7.8788892335272232E+10, 4.0355760945670044E+11, 8.8071481911347949E+11, 8.8071481911347961E+11, 4.0355760945670044E+11, 7.8788892335272430E+10, 5.4431675199498835E+09, 8.4789650417103708E+07, 6.1722991679871957E+04};
    FLT c1[] = {3.2561466099406168E+05, 2.2112758120210618E+08, 8.9911609880089817E+09, 8.3059508064200943E+10, 2.3965569143469864E+11, 1.6939286803305212E+11, -1.6939286803305203E+11, -2.3965569143469864E+11, -8.3059508064201080E+10, -8.9911609880089989E+09, -2.2112758120210618E+08, -3.2561466099404311E+05};
    FLT c2[] = {7.6621098001581512E+05, 2.6026568260310286E+08, 6.4524338253008652E+09, 3.3729904113826820E+10, 2.8555202212474091E+10, -6.8998572040731537E+10, -6.8998572040731445E+10, 2.8555202212474079E+10, 3.3729904113826824E+10, 6.4524338253008757E+09, 2.6026568260310274E+08, 7.6621098001583829E+05};
    FLT c3[] = {1.0657807616803218E+06, 1.8144472126890984E+08, 2.5524827004349842E+09, 5.2112383911371660E+09, -1.0268350564014645E+10, -1.4763245309081306E+10, 1.4763245309081314E+10, 1.0268350564014671E+10, -5.2112383911371059E+09, -2.5524827004349871E+09, -1.8144472126890984E+08, -1.0657807616803099E+06};
    FLT c4[] = {9.7829638830158755E+05, 8.2222351241519913E+07, 5.5676911894064474E+08, -4.8739037675427330E+08, -2.7153428193078227E+09, 2.5627633609246106E+09, 2.5627633609246163E+09, -2.7153428193078651E+09, -4.8739037675430620E+08, 5.5676911894064546E+08, 8.2222351241519868E+07, 9.7829638830161188E+05};
    FLT c5[] = {6.2536876825114002E+05, 2.4702814073680203E+07, 4.1488431554846466E+07, -2.9274790542418826E+08, 1.0742154109191516E+08, 6.2185168968032193E+08, -6.2185168968012476E+08, -1.0742154109184742E+08, 2.9274790542423087E+08, -4.1488431554843128E+07, -2.4702814073680237E+07, -6.2536876825112454E+05};
    FLT c6[] = {2.8527714307528478E+05, 4.6266378435690766E+06, -1.0665598090790771E+07, -2.6048960239891130E+07, 9.1597254427317813E+07, -5.9794495983264342E+07, -5.9794495983220413E+07, 9.1597254427343085E+07, -2.6048960239921503E+07, -1.0665598090794146E+07, 4.6266378435690673E+06, 2.8527714307530399E+05};
    FLT c7[] = {9.2873647411234080E+04, 3.6630046787425119E+05, -3.1271047224730137E+06, 4.8612412939252760E+06, 3.3820440907796426E+06, -1.6880127953704204E+07, 1.6880127953756198E+07, -3.3820440907614031E+06, -4.8612412938993908E+06, 3.1271047224752530E+06, -3.6630046787425695E+05, -9.2873647411217215E+04};
    FLT c8[] = {2.0817947751046438E+04, -5.5660303410315042E+04, -1.9519783923444615E+05, 1.0804817251338551E+06, -1.8264985852555393E+06, 9.7602844968061335E+05, 9.7602844962902542E+05, -1.8264985852963410E+06, 1.0804817251124913E+06, -1.9519783923503032E+05, -5.5660303410363231E+04, 2.0817947751063632E+04};
    FLT c9[] = {2.7986023314783361E+03, -1.9404411093655592E+04, 4.3922625000519314E+04, -7.6450317451901383E+03, -1.5273911974273989E+05, 3.3223441458516393E+05, -3.3223441441930021E+05, 1.5273911979752057E+05, 7.6450317512768806E+03, -4.3922624998141677E+04, 1.9404411093637758E+04, -2.7986023314644049E+03};
    FLT c10[] = {6.7849020474048089E+01, -1.7921351308204744E+03, 8.4980694686552797E+03, -1.9742624859769410E+04, 2.4620674845030797E+04, -1.1676544851227827E+04, -1.1676544869194569E+04, 2.4620674845030626E+04, -1.9742624831436660E+04, 8.4980694630406069E+03, -1.7921351308312935E+03, 6.7849020488592075E+01};
    FLT c11[] = {-5.4577020998836872E+01, 1.3637112867242237E+02, 4.5513616580246023E+01, -1.1174001367986359E+03, 3.2018769312434206E+03, -5.0580351396215219E+03, 5.0580351683422405E+03, -3.2018769242193171E+03, 1.1174000998831286E+03, -4.5513609243969356E+01, -1.3637112867730119E+02, 5.4577021011726984E+01};
    FLT c12[] = {-1.0538365872268786E+01, 4.6577222488645518E+01, -1.2606964198473415E+02, 2.1881091668968099E+02, -2.3273399614976032E+02, 1.0274275204276027E+02, 1.0274270265494516E+02, -2.3273401859852868E+02, 2.1881091865396468E+02, -1.2606964777237258E+02, 4.6577222453584369E+01, -1.0538365860573146E+01};
    FLT c13[] = {-4.6087004144309118E-01, 2.5969759128998060E+00, -9.6946932216381381E+00, 2.4990041962121211E+01, -4.6013909139329137E+01, 6.2056985032913090E+01, -6.2056925855365186E+01, 4.6013921000662158E+01, -2.4990037445376750E+01, 9.6946954085586885E+00, -2.5969759201692755E+00, 4.6087004744129911E-01};
    for (int i=0; i<12; i++) ker[i] = c0[i] + z*(c1[i] + z*(c2[i] + z*(c3[i] + z*(c4[i] + z*(c5[i] + z*(c6[i] + z*(c7[i] + z*(c8[i] + z*(c9[i] + z*(c10[i] + z*(c11[i] + z*(c12[i] + z*(c13[i])))))))))))));
  } else if (w==13) {
    FLT c0[] = {9.8715725867495363E+04, 1.9828875496808097E+08, 1.7196758809614983E+10, 3.3083776881353577E+11, 2.2668873993375439E+12, 6.7734720591167568E+12, 9.6695220682534785E+12, 6.7734720591167432E+12, 2.2668873993375430E+12, 3.3083776881353503E+11, 1.7196758809614998E+10, 1.9828875496807891E+08, 9.8715725867496090E+04, 0.0000000000000000E+00, 0.0000000000000000E+00, 0.0000000000000000E+00};
    FLT c1[] = {5.4491110456935549E+05, 5.4903670125539351E+08, 3.0879465445278183E+10, 3.9588436413399969E+11, 1.6860562536749778E+12, 2.4256447893117891E+12, -5.5583944938791784E-05, -2.4256447893117847E+12, -1.6860562536749768E+12, -3.9588436413399890E+11, -3.0879465445278183E+10, -5.4903670125538898E+08, -5.4491110456935526E+05, 0.0000000000000000E+00, 0.0000000000000000E+00, 0.0000000000000000E+00};
    FLT c2[] = {1.3504711883426071E+06, 6.9286979077463162E+08, 2.4618123595484577E+10, 1.9493985627722607E+11, 3.9422703517046350E+11, -1.8678883613919861E+11, -8.5538079834550110E+11, -1.8678883613919730E+11, 3.9422703517046375E+11, 1.9493985627722589E+11, 2.4618123595484566E+10, 6.9286979077462614E+08, 1.3504711883426069E+06, 0.0000000000000000E+00, 0.0000000000000000E+00, 0.0000000000000000E+00};
    FLT c3[] = {1.9937206140846491E+06, 5.2512029493765980E+08, 1.1253303793811750E+10, 4.6205527735932152E+10, -1.1607472377983305E+10, -1.6305241755642313E+11, 3.5385440504350348E-04, 1.6305241755642365E+11, 1.1607472377982582E+10, -4.6205527735932213E+10, -1.1253303793811750E+10, -5.2512029493765628E+08, -1.9937206140846489E+06, 0.0000000000000000E+00, 0.0000000000000000E+00, 0.0000000000000000E+00};
    FLT c4[] = {1.9607419630386413E+06, 2.6425362558103892E+08, 3.1171259341747193E+09, 2.9839860297839913E+09, -1.9585031917561897E+10, -5.0666917387065792E+09, 3.6568794485480583E+10, -5.0666917387057562E+09, -1.9585031917561817E+10, 2.9839860297838497E+09, 3.1171259341747184E+09, 2.6425362558103728E+08, 1.9607419630386417E+06, 0.0000000000000000E+00, 0.0000000000000000E+00, 0.0000000000000000E+00};
    FLT c5[] = {1.3593773865640305E+06, 9.1556445104158267E+07, 4.7074012944133747E+08, -1.1192579335657008E+09, -2.1090780087868555E+09, 5.2270306737951984E+09, 5.6467240041521856E-04, -5.2270306737934217E+09, 2.1090780087880819E+09, 1.1192579335658383E+09, -4.7074012944133127E+08, -9.1556445104157984E+07, -1.3593773865640305E+06, 0.0000000000000000E+00, 0.0000000000000000E+00, 0.0000000000000000E+00};
    FLT c6[] = {6.8417206432039209E+05, 2.1561705510027152E+07, 7.5785249893055111E+06, -2.7456096030221754E+08, 3.4589095671054310E+08, 4.0256106808894646E+08, -1.0074306926603404E+09, 4.0256106809081393E+08, 3.4589095670997137E+08, -2.7456096030236483E+08, 7.5785249893030487E+06, 2.1561705510027405E+07, 6.8417206432039209E+05, 0.0000000000000000E+00, 0.0000000000000000E+00, 0.0000000000000000E+00};
    FLT c7[] = {2.5248269397037517E+05, 3.0985559672616189E+06, -1.1816517087616559E+07, -8.2958498770184973E+06, 8.0546642347355247E+07, -1.0594657799485898E+08, 2.1816722293163801E-04, 1.0594657799424352E+08, -8.0546642347497791E+07, 8.2958498771036500E+06, 1.1816517087615721E+07, -3.0985559672621777E+06, -2.5248269397037517E+05, 0.0000000000000000E+00, 0.0000000000000000E+00, 0.0000000000000000E+00};
    FLT c8[] = {6.7530100970876694E+04, 1.2373362326658823E+05, -2.1245597183281910E+06, 5.1047323238754412E+06, -1.4139444405488928E+06, -1.1818267555096827E+07, 2.0121548578624789E+07, -1.1818267557079868E+07, -1.4139444401348191E+06, 5.1047323236516044E+06, -2.1245597183309775E+06, 1.2373362326702787E+05, 6.7530100970876316E+04, 0.0000000000000000E+00, 0.0000000000000000E+00, 0.0000000000000000E+00};
    FLT c9[] = {1.2421368748961073E+04, -5.0576243647011936E+04, -4.8878193436902722E+04, 6.5307896872028301E+05, -1.5497610127060430E+06, 1.5137725917321201E+06, 4.1615986404011299E-04, -1.5137725918538549E+06, 1.5497610130469005E+06, -6.5307896856811445E+05, 4.8878193438804832E+04, 5.0576243646433126E+04, -1.2421368748961073E+04, 0.0000000000000000E+00, 0.0000000000000000E+00, 0.0000000000000000E+00};
    FLT c10[] = {1.2904654687550299E+03, -1.1169946055009055E+04, 3.3275109713863385E+04, -3.1765222274236821E+04, -5.9810982085323274E+04, 2.2355863038592847E+05, -3.1083591705219547E+05, 2.2355863445202672E+05, -5.9810982721084511E+04, -3.1765222464963932E+04, 3.3275109714208855E+04, -1.1169946054555618E+04, 1.2904654687545376E+03, 0.0000000000000000E+00, 0.0000000000000000E+00, 0.0000000000000000E+00};
    FLT c11[] = {-1.9043622268674213E+01, -6.8296542209516542E+02, 4.2702512274202591E+03, -1.2165497317825058E+04, 1.9423733298269544E+04, -1.6010024066956401E+04, 3.4018642874429026E-04, 1.6010021599471667E+04, -1.9423732817821805E+04, 1.2165497483905752E+04, -4.2702512286689680E+03, 6.8296542153908558E+02, 1.9043622268312891E+01, 0.0000000000000000E+00, 0.0000000000000000E+00, 0.0000000000000000E+00};
    FLT c12[] = {-3.0093984465361217E+01, 9.8972865724808671E+01, -9.7437038666761538E+01, -3.5079928405373198E+02, 1.5699250566648977E+03, -3.1287439837941820E+03, 3.8692196309709061E+03, -3.1287462825615335E+03, 1.5699252631958864E+03, -3.5079944793112952E+02, -9.7437041893750632E+01, 9.8972866189610414E+01, -3.0093984465884773E+01, 0.0000000000000000E+00, 0.0000000000000000E+00, 0.0000000000000000E+00};
    FLT c13[] = {-4.3050286009489040E+00, 2.1108975724659501E+01, -6.4297198812570272E+01, 1.2922884632277874E+02, -1.6991812716212596E+02, 1.2655005901719436E+02, 9.2483537895948854E-05, -1.2655066232531748E+02, 1.6991805207569072E+02, -1.2922893667436634E+02, 6.4297198424711908E+01, -2.1108976207523057E+01, 4.3050286009485790E+00, 0.0000000000000000E+00, 0.0000000000000000E+00, 0.0000000000000000E+00};
    FLT c14[] = {-1.0957333716725008E-01, 7.2949317004436565E-01, -3.4300816058693728E+00, 1.0470054474579324E+01, -2.2292134950656113E+01, 3.4570827323582719E+01, -3.9923523442753932E+01, 3.4573264959502886E+01, -2.2292358612963266E+01, 1.0470042004916014E+01, -3.4300810538570281E+00, 7.2949352113279253E-01, -1.0957333740315604E-01, 0.0000000000000000E+00, 0.0000000000000000E+00, 0.0000000000000000E+00};
    for (int i=0; i<16; i++) ker[i] = c0[i] + z*(c1[i] + z*(c2[i] + z*(c3[i] + z*(c4[i] + z*(c5[i] + z*(c6[i] + z*(c7[i] + z*(c8[i] + z*(c9[i] + z*(c10[i] + z*(c11[i] + z*(c12[i] + z*(c13[i] + z*(c14[i]))))))))))))));
  } else if (w==14) {
    FLT c0[] = {1.5499533202966207E+05, 4.4723032442444688E+08, 5.1495083701694740E+10, 1.2904576022918071E+12, 1.1534950432785506E+13, 4.5650102198520484E+13, 8.8830582190032641E+13, 8.8830582190032641E+13, 4.5650102198520492E+13, 1.1534950432785527E+13, 1.2904576022918074E+12, 5.1495083701695107E+10, 4.4723032442444855E+08, 1.5499533202970232E+05, 0.0000000000000000E+00, 0.0000000000000000E+00};
    FLT c1[] = {8.9188339002980455E+05, 1.3065352538728635E+09, 9.9400185225815567E+10, 1.7136059013402405E+12, 1.0144146621675832E+13, 2.3034036018490715E+13, 1.4630967270448871E+13, -1.4630967270448855E+13, -2.3034036018490719E+13, -1.0144146621675846E+13, -1.7136059013402405E+12, -9.9400185225815964E+10, -1.3065352538728662E+09, -8.9188339002979454E+05, 0.0000000000000000E+00, 0.0000000000000000E+00};
    FLT c2[] = {2.3170473769379663E+06, 1.7532505043698256E+09, 8.6523535958354309E+10, 9.7455289065487354E+11, 3.2977972139362314E+12, 1.7874626001697781E+12, -6.1480918082633916E+12, -6.1480918082633975E+12, 1.7874626001697690E+12, 3.2977972139362285E+12, 9.7455289065487329E+11, 8.6523535958354630E+10, 1.7532505043698275E+09, 2.3170473769380399E+06, 0.0000000000000000E+00, 0.0000000000000000E+00};
    FLT c3[] = {3.6089249230396422E+06, 1.4278058213962190E+09, 4.4296625537022423E+10, 2.9466624630419781E+11, 3.1903621584503235E+11, -9.8834691411254565E+11, -1.1072264714919226E+12, 1.1072264714919316E+12, 9.8834691411255151E+11, -3.1903621584503467E+11, -2.9466624630419769E+11, -4.4296625537022621E+10, -1.4278058213962219E+09, -3.6089249230396664E+06, 0.0000000000000000E+00, 0.0000000000000000E+00};
    FLT c4[] = {3.7733555140851745E+06, 7.8376718099107409E+08, 1.4443117772349569E+10, 4.3197433307418671E+10, -7.6585042240585556E+10, -1.8569640140763062E+11, 2.0385335192657199E+11, 2.0385335192656519E+11, -1.8569640140762662E+11, -7.6585042240580856E+10, 4.3197433307418686E+10, 1.4443117772349669E+10, 7.8376718099107552E+08, 3.7733555140852560E+06, 0.0000000000000000E+00, 0.0000000000000000E+00};
    FLT c5[] = {2.8079157920112358E+06, 3.0340753492383724E+08, 2.9498136661747241E+09, -6.2820200387919831E+08, -2.2372008390623215E+10, 1.5217518660584890E+10, 4.0682590266891922E+10, -4.0682590266869431E+10, -1.5217518660582748E+10, 2.2372008390625935E+10, 6.2820200387968791E+08, -2.9498136661747637E+09, -3.0340753492383808E+08, -2.8079157920112377E+06, 0.0000000000000000E+00, 0.0000000000000000E+00};
    FLT c6[] = {1.5361613559533111E+06, 8.3513615594416574E+07, 3.0077547202708024E+08, -1.3749596754067802E+09, -6.6733027297557127E+08, 5.9590333632819109E+09, -4.3025685566870070E+09, -4.3025685566872711E+09, 5.9590333632806673E+09, -6.6733027297523963E+08, -1.3749596754067125E+09, 3.0077547202709383E+08, 8.3513615594416171E+07, 1.5361613559533576E+06, 0.0000000000000000E+00, 0.0000000000000000E+00};
    FLT c7[] = {6.2759409419592959E+05, 1.5741723594963098E+07, -1.5632610223406436E+07, -1.9294824907078514E+08, 4.4643806532434595E+08, 1.5178998385244830E+07, -9.6771139891725647E+08, 9.6771139892509627E+08, -1.5178998381042883E+07, -4.4643806533176166E+08, 1.9294824907065383E+08, 1.5632610223392555E+07, -1.5741723594963137E+07, -6.2759409419590747E+05, 0.0000000000000000E+00, 0.0000000000000000E+00};
    FLT c8[] = {1.9151404903933613E+05, 1.7156606891563335E+06, -9.7733523156688716E+06, 4.2982266233154163E+06, 5.1660907884347722E+07, -1.1279400211155911E+08, 6.4701089573962681E+07, 6.4701089571562663E+07, -1.1279400211012064E+08, 5.1660907891220264E+07, 4.2982266233826512E+06, -9.7733523157112263E+06, 1.7156606891560503E+06, 1.9151404903936724E+05, 0.0000000000000000E+00, 0.0000000000000000E+00};
    FLT c9[] = {4.2715272622845026E+04, -2.2565910611953568E+03, -1.1769776156959014E+06, 4.0078399907813077E+06, -3.8951858063335596E+06, -5.0944610754510267E+06, 1.6765992446914168E+07, -1.6765992426657490E+07, 5.0944610781778870E+06, 3.8951858062361716E+06, -4.0078399907326135E+06, 1.1769776157141617E+06, 2.2565910606306688E+03, -4.2715272622820135E+04, 0.0000000000000000E+00, 0.0000000000000000E+00};
    FLT c10[] = {6.4806786522793900E+03, -3.5474227032974472E+04, 1.8237100709385861E+04, 3.0934714629696816E+05, -1.0394703931686131E+06, 1.4743920333143482E+06, -7.3356882447856572E+05, -7.3356882916658197E+05, 1.4743920305501707E+06, -1.0394703929917105E+06, 3.0934714631908614E+05, 1.8237100665157792E+04, -3.5474227033406372E+04, 6.4806786523010323E+03, 0.0000000000000000E+00, 0.0000000000000000E+00};
    FLT c11[] = {4.9913632908459954E+02, -5.5416668524952684E+03, 2.0614058717617296E+04, -3.2285139072943130E+04, -5.3099550821623425E+03, 1.1559000502166932E+05, -2.2569743259261423E+05, 2.2569743616896842E+05, -1.1559000130545651E+05, 5.3099543129458480E+03, 3.2285139142872020E+04, -2.0614058670790018E+04, 5.5416668533342381E+03, -4.9913632906195977E+02, 0.0000000000000000E+00, 0.0000000000000000E+00};
    FLT c12[] = {-3.3076333188134086E+01, -1.8970588563697331E+02, 1.8160423493164808E+03, -6.3715703355644328E+03, 1.2525624574329036E+04, -1.4199806452802783E+04, 6.4441892296909591E+03, 6.4441909537524216E+03, -1.4199808176873401E+04, 1.2525626154733827E+04, -6.3715704433222418E+03, 1.8160422729911850E+03, -1.8970588700495102E+02, -3.3076333168231550E+01, 0.0000000000000000E+00, 0.0000000000000000E+00};
    FLT c13[] = {-1.4394533627743886E+01, 5.7000699089242815E+01, -1.0101142663923416E+02, -3.2954197414395189E+01, 6.1417879182394654E+02, -1.6177283846697430E+03, 2.4593386157454975E+03, -2.4593322941165261E+03, 1.6177291239900730E+03, -6.1417952013923764E+02, 3.2954100943010943E+01, 1.0101142710333265E+02, -5.7000699100179844E+01, 1.4394533639240331E+01, 0.0000000000000000E+00, 0.0000000000000000E+00};
    FLT c14[] = {-1.5925952284027161E+00, 8.5113930215357829E+00, -2.8993523187012922E+01, 6.6373454994590404E+01, -1.0329574518449559E+02, 1.0280184257681817E+02, -4.3896094875192006E+01, -4.3899302208087086E+01, 1.0280039795628096E+02, -1.0329511291885207E+02, 6.6373435700858948E+01, -2.8993536490606409E+01, 8.5113924808491728E+00, -1.5925952194145006E+00, 0.0000000000000000E+00, 0.0000000000000000E+00};
    FLT c15[] = {1.5984868520881029E-02, 1.2876175212962959E-01, -9.8358742969175483E-01, 3.7711523389360830E+00, -9.4305498095765508E+00, 1.6842854581416674E+01, -2.2308566502972713E+01, 2.2308940200151390E+01, -1.6841512668820517E+01, 9.4313524091989347E+00, -3.7710716543179599E+00, 9.8361025494556609E-01, -1.2876100566420701E-01, -1.5984859433053292E-02, 0.0000000000000000E+00, 0.0000000000000000E+00};
    for (int i=0; i<16; i++) ker[i] = c0[i] + z*(c1[i] + z*(c2[i] + z*(c3[i] + z*(c4[i] + z*(c5[i] + z*(c6[i] + z*(c7[i] + z*(c8[i] + z*(c9[i] + z*(c10[i] + z*(c11[i] + z*(c12[i] + z*(c13[i] + z*(c14[i] + z*(c15[i])))))))))))))));
  } else if (w==15) {
    FLT c0[] = {2.3939707792241839E+05, 9.7700272582690191E+08, 1.4715933396485257E+11, 4.7242424833337158E+12, 5.3987426629953594E+13, 2.7580474290566078E+14, 7.0693378336533400E+14, 9.6196578554477775E+14, 7.0693378336533400E+14, 2.7580474290566125E+14, 5.3987426629953766E+13, 4.7242424833337246E+12, 1.4715933396485263E+11, 9.7700272582690215E+08, 2.3939707792242285E+05, 0.0000000000000000E+00};
    FLT c1[] = {1.4314487885226035E+06, 2.9961416925358453E+09, 3.0273361232748438E+11, 6.8507333793903584E+12, 5.4192702756911000E+13, 1.7551587948105309E+14, 2.1874615668430150E+14, 3.4316191014053393E-02, -2.1874615668430150E+14, -1.7551587948105334E+14, -5.4192702756911180E+13, -6.8507333793903701E+12, -3.0273361232748438E+11, -2.9961416925358458E+09, -1.4314487885226049E+06, 0.0000000000000000E+00};
    FLT c2[] = {3.8829497354762917E+06, 4.2473082696966448E+09, 2.8414312556015540E+11, 4.3688281331121411E+12, 2.1823119508000543E+13, 3.2228098609392094E+13, -2.1833085454691789E+13, -7.3750710225100812E+13, -2.1833085454691820E+13, 3.2228098609392055E+13, 2.1823119508000594E+13, 4.3688281331121479E+12, 2.8414312556015527E+11, 4.2473082696966434E+09, 3.8829497354762889E+06, 0.0000000000000000E+00};
    FLT c3[] = {6.3495763451755755E+06, 3.6841035003733950E+09, 1.5965774278321045E+11, 1.5630338683778201E+12, 3.8749058615819268E+12, -2.7319740087723574E+12, -1.3233342822865402E+13, 6.1642230420317079E-02, 1.3233342822865449E+13, 2.7319740087723975E+12, -3.8749058615819365E+12, -1.5630338683778203E+12, -1.5965774278321042E+11, -3.6841035003733935E+09, -6.3495763451755764E+06, 0.0000000000000000E+00};
    FLT c4[] = {7.0146619045520434E+06, 2.1782897863065763E+09, 5.8897780310148087E+10, 3.1953009601770325E+11, 4.0651527029737198E+08, -1.6379148273276064E+12, -1.1568753137013029E+11, 2.7451653250460508E+12, -1.1568753137012485E+11, -1.6379148273277261E+12, 4.0651527029819238E+08, 3.1953009601770361E+11, 5.8897780310148087E+10, 2.1782897863065763E+09, 7.0146619045520443E+06, 0.0000000000000000E+00};
    FLT c5[] = {5.5580012413990172E+06, 9.2345162185944164E+08, 1.4522950934020109E+10, 2.7025952371212009E+10, -1.2304576967641914E+11, -1.0116752717202786E+11, 3.8517418245458325E+11, 1.0918347404432817E-01, -3.8517418245444312E+11, 1.0116752717221135E+11, 1.2304576967643665E+11, -2.7025952371214943E+10, -1.4522950934020079E+10, -9.2345162185944211E+08, -5.5580012413990181E+06, 0.0000000000000000E+00};
    FLT c6[] = {3.2693972344231778E+06, 2.8610260147425205E+08, 2.2348528403750563E+09, -3.4574515574242272E+09, -1.7480626463583939E+10, 3.1608597465540653E+10, 1.9879262560072273E+10, -6.6148013553772224E+10, 1.9879262560085339E+10, 3.1608597465515747E+10, -1.7480626463576942E+10, -3.4574515574198236E+09, 2.2348528403750110E+09, 2.8610260147425193E+08, 3.2693972344231787E+06, 0.0000000000000000E+00};
    FLT c7[] = {1.4553539959296256E+06, 6.4136842048384041E+07, 1.3622336582062906E+08, -1.2131510424644001E+09, 6.4322366984221375E+08, 4.5078753872047586E+09, -7.1689413746930647E+09, 3.2906916833662987E-02, 7.1689413746724453E+09, -4.5078753875009747E+09, -6.4322366985365331E+08, 1.2131510424608817E+09, -1.3622336582067037E+08, -6.4136842048384242E+07, -1.4553539959296256E+06, 0.0000000000000000E+00};
    FLT c8[] = {4.9358776531681651E+05, 9.7772970960585065E+06, -2.3511574237987626E+07, -1.0142613816641946E+08, 3.9421144218035364E+08, -2.8449115593052310E+08, -5.7549243243741119E+08, 1.1608781631182449E+09, -5.7549243240763104E+08, -2.8449115600447333E+08, 3.9421144214381480E+08, -1.0142613816429654E+08, -2.3511574237995699E+07, 9.7772970960588697E+06, 4.9358776531681546E+05, 0.0000000000000000E+00};
    FLT c9[] = {1.2660319987326677E+05, 7.7519511328119377E+05, -6.5244610661450895E+06, 9.0878257488052379E+06, 2.3116605621149920E+07, -8.7079594462079599E+07, 9.5542733739275128E+07, 6.0548970733798724E-02, -9.5542733661364838E+07, 8.7079594608550951E+07, -2.3116605559600785E+07, -9.0878257522138134E+06, 6.5244610661298726E+06, -7.7519511328133650E+05, -1.2660319987326639E+05, 0.0000000000000000E+00};
    FLT c10[] = {2.3793325531458529E+04, -4.2305332803808597E+04, -5.2884156985535356E+05, 2.5307340127864038E+06, -4.0404175271559842E+06, -1.7519992360184138E+05, 1.0146438805818636E+07, -1.5828545480742473E+07, 1.0146438778928882E+07, -1.7520004389869148E+05, -4.0404175770437294E+06, 2.5307340149977510E+06, -5.2884156989405944E+05, -4.2305332803937294E+04, 2.3793325531459184E+04, 0.0000000000000000E+00};
    FLT c11[] = {2.9741655196834722E+03, -2.0687056403786246E+04, 3.3295507799709936E+04, 1.0661145730323243E+05, -5.6644238105382060E+05, 1.0874811616841732E+06, -9.6561270266008016E+05, 1.5626594062671070E-02, 9.6561272951271443E+05, -1.0874812528712249E+06, 5.6644243308078672E+05, -1.0661145838213131E+05, -3.3295507812197495E+04, 2.0687056403630129E+04, -2.9741655196846405E+03, 0.0000000000000000E+00};
    FLT c12[] = {1.5389176594899303E+02, -2.3864418511494741E+03, 1.0846266954249364E+04, -2.2940053396478714E+04, 1.4780106121058996E+04, 4.2663651769852157E+04, -1.3047648013242516E+05, 1.7468401314164279E+05, -1.3047645484607235E+05, 4.2663541429144650E+04, 1.4780036296018619E+04, -2.2940053180976502E+04, 1.0846266927315819E+04, -2.3864418517113058E+03, 1.5389176594779781E+02, 0.0000000000000000E+00};
    FLT c13[] = {-2.3857631312588978E+01, -1.9651606133609231E+01, 6.4183083829803820E+02, -2.8648433109641578E+03, 6.8249243722518859E+03, -9.7944325124827701E+03, 7.6177757600121276E+03, 1.8034307737205296E-02, -7.6177559127722052E+03, 9.7944326623113047E+03, -6.8249058342322496E+03, 2.8648407117981119E+03, -6.4183085438795774E+02, 1.9651605969778377E+01, 2.3857631312809222E+01, 0.0000000000000000E+00};
    FLT c14[] = {-6.1348505739169541E+00, 2.7872915855267404E+01, -6.5819942538871970E+01, 5.1366231962952028E+01, 1.7213955398158618E+02, -6.9658621010000411E+02, 1.3192236112353403E+03, -1.6054106225233884E+03, 1.3192031991952242E+03, -6.9663961216547739E+02, 1.7211403815802629E+02, 5.1367579954366171E+01, -6.5819957939661379E+01, 2.7872915947616441E+01, -6.1348505735855374E+00, 0.0000000000000000E+00};
    FLT c15[] = {-4.9671584513490097E-01, 3.0617550953446115E+00, -1.1650665638578070E+01, 3.0081586723089057E+01, -5.4028356726202020E+01, 6.6077203078498044E+01, -4.7145500171928198E+01, 4.2118837140985958E-03, 4.7167106663349848E+01, -6.6048394423269173E+01, 5.4062906728994193E+01, -3.0081603709324451E+01, 1.1650672008416343E+01, -3.0617551285208524E+00, 4.9671584437353217E-01, 0.0000000000000000E+00};
    FLT c16[] = {4.3460786767313729E-03, -1.3199600771767199E-02, -1.9412688562910244E-01, 1.1329433700669471E+00, -3.4442045795063887E+00, 7.1737626956468912E+00, -1.1098109271625262E+01, 1.2385772358881393E+01, -1.1101471316239516E+01, 7.0913926025978853E+00, -3.4845491148773502E+00, 1.1323523856621058E+00, -1.9414904754428672E-01, -1.3200165079792004E-02, 4.3460782759443158E-03, 0.0000000000000000E+00};
    for (int i=0; i<16; i++) ker[i] = c0[i] + z*(c1[i] + z*(c2[i] + z*(c3[i] + z*(c4[i] + z*(c5[i] + z*(c6[i] + z*(c7[i] + z*(c8[i] + z*(c9[i] + z*(c10[i] + z*(c11[i] + z*(c12[i] + z*(c13[i] + z*(c14[i] + z*(c15[i] + z*(c16[i]))))))))))))))));
  } else if (w==16) {
    FLT c0[] = {3.6434551345570839E+05, 2.0744705928579483E+09, 4.0355760945669995E+11, 1.6364575388763029E+13, 2.3514830376056538E+14, 1.5192201717462528E+15, 4.9956173084674090E+15, 8.9287666945127360E+15, 8.9287666945127390E+15, 4.9956173084674090E+15, 1.5192201717462528E+15, 2.3514830376056538E+14, 1.6364575388763035E+13, 4.0355760945670026E+11, 2.0744705928579524E+09, 3.6434551345571183E+05};
    FLT c1[] = {2.2576246485480359E+06, 6.6499571180086451E+09, 8.7873753526056287E+11, 2.5606844387131066E+13, 2.6313738449330153E+14, 1.1495095100701460E+15, 2.1932582707747560E+15, 1.2860244365132595E+15, -1.2860244365132600E+15, -2.1932582707747578E+15, -1.1495095100701465E+15, -2.6313738449330159E+14, -2.5606844387131062E+13, -8.7873753526056299E+11, -6.6499571180086451E+09, -2.2576246485480373E+06};
    FLT c2[] = {6.3730995546265077E+06, 9.9060026035198078E+09, 8.8097248605449023E+11, 1.7953384130753688E+13, 1.2398425545001662E+14, 3.0749346493041262E+14, 1.0259777520247159E+14, -5.5291976457534325E+14, -5.5291976457534325E+14, 1.0259777520247186E+14, 3.0749346493041219E+14, 1.2398425545001659E+14, 1.7953384130753676E+13, 8.8097248605448950E+11, 9.9060026035198040E+09, 6.3730995546265030E+06};
    FLT c3[] = {1.0896915393078227E+07, 9.0890343524593849E+09, 5.3565169504010010E+11, 7.3004206720038701E+12, 2.9692333044160066E+13, 1.6051737468109549E+13, -9.1273329108089906E+13, -8.5999306918502953E+13, 8.5999306918502422E+13, 9.1273329108089984E+13, -1.6051737468109510E+13, -2.9692333044160082E+13, -7.3004206720038701E+12, -5.3565169504010022E+11, -9.0890343524593849E+09, -1.0896915393078227E+07};
    FLT c4[] = {1.2655725616100594E+07, 5.7342804054544210E+09, 2.1822836608899570E+11, 1.8300700858999690E+12, 2.7770431049857676E+12, -8.5034969223852568E+12, -1.2846668467423438E+13, 1.6519076896571838E+13, 1.6519076896572182E+13, -1.2846668467423555E+13, -8.5034969223850703E+12, 2.7770431049857896E+12, 1.8300700858999678E+12, 2.1822836608899567E+11, 5.7342804054544210E+09, 1.2655725616100591E+07};
    FLT c5[] = {1.0609303958036326E+07, 2.6255609052371716E+09, 6.1673589426039413E+10, 2.6044432099085333E+11, -3.5431628074578204E+11, -1.6077602129636348E+12, 1.5534405614728977E+12, 2.8019935380857432E+12, -2.8019935380841978E+12, -1.5534405614724106E+12, 1.6077602129635625E+12, 3.5431628074580896E+11, -2.6044432099084848E+11, -6.1673589426039429E+10, -2.6255609052371716E+09, -1.0609303958036322E+07};
    FLT c6[] = {6.6544809363384582E+06, 8.9490403680928326E+08, 1.1882638725190845E+10, 8.1552898137823076E+09, -1.2575562817886868E+11, 2.7074695075907585E+10, 3.9453789461955023E+11, -3.1679644857468066E+11, -3.1679644857392346E+11, 3.9453789461966650E+11, 2.7074695075992649E+10, -1.2575562817884555E+11, 8.1552898137788668E+09, 1.1882638725190889E+10, 8.9490403680928278E+08, 6.6544809363384554E+06};
    FLT c7[] = {3.1906872142825006E+06, 2.2785946180651775E+08, 1.3744578972809248E+09, -4.3997172592883167E+09, -9.2011130754043922E+09, 3.4690551711832901E+10, -9.4227043395047741E+09, -5.9308465070198639E+10, 5.9308465069336540E+10, 9.4227043396350136E+09, -3.4690551711738396E+10, 9.2011130753567543E+09, 4.3997172592879610E+09, -1.3744578972813025E+09, -2.2785946180651844E+08, -3.1906872142825015E+06};
    FLT c8[] = {1.1821527096621769E+06, 4.2281234059839502E+07, 2.8723226058712766E+07, -8.3553955857628822E+08, 1.2447304828823066E+09, 2.1955280943585949E+09, -7.0514195726908512E+09, 4.3745141239718714E+09, 4.3745141233600502E+09, -7.0514195728029747E+09, 2.1955280943510208E+09, 1.2447304828590808E+09, -8.3553955857879233E+08, 2.8723226058761366E+07, 4.2281234059838109E+07, 1.1821527096621762E+06};
    FLT c9[] = {3.3854610744280310E+05, 5.2176984975081543E+06, -2.0677283565079328E+07, -3.5831818968518838E+07, 2.6599346106412742E+08, -3.7992777977357000E+08, -1.3426914417466179E+08, 9.1752051229224503E+08, -9.1752051129499328E+08, 1.3426914497246322E+08, 3.7992777991069216E+08, -2.6599346104854536E+08, 3.5831818968908392E+07, 2.0677283564896725E+07, -5.2176984975075833E+06, -3.3854610744279937E+05};
    FLT c10[] = {7.3893334077310064E+04, 2.6983804209559254E+05, -3.6415998561101072E+06, 8.4025485849181097E+06, 4.9278860779345948E+06, -5.1437033846752726E+07, 8.7603898676325440E+07, -4.6199498412402093E+07, -4.6199498208604209E+07, 8.7603898435731798E+07, -5.1437033863736227E+07, 4.9278861005789889E+06, 8.4025485831489991E+06, -3.6415998560990733E+06, 2.6983804209473461E+05, 7.3893334077307401E+04};
    FLT c11[] = {1.1778892113375481E+04, -4.0077190108724200E+04, -1.8372552175909068E+05, 1.3262878399160223E+06, -2.9738539927520575E+06, 1.9493509709529271E+06, 4.1881949951139782E+06, -1.1066749616505133E+07, 1.1066749327519676E+07, -4.1881946843906553E+06, -1.9493507810665092E+06, 2.9738539818831389E+06, -1.3262878384774840E+06, 1.8372552162922107E+05, 4.0077190107319519E+04, -1.1778892113376129E+04};
    FLT c12[] = {1.2019749667923656E+03, -1.0378455844500613E+04, 2.6333352653155256E+04, 1.7117060106301305E+04, -2.5133287443653666E+05, 6.4713914262131555E+05, -8.1634942572553246E+05, 3.8623935281825601E+05, 3.8623876433339820E+05, -8.1634960962672008E+05, 6.4713900469564367E+05, -2.5133289627502396E+05, 1.7117057951236206E+04, 2.6333352581335013E+04, -1.0378455846609291E+04, 1.2019749667911419E+03};
    FLT c13[] = {3.1189837632471693E+01, -8.9083493807061564E+02, 4.9454293649337906E+03, -1.3124693635095375E+04, 1.5834784331991095E+04, 6.9607870364081436E+03, -5.9789871879430451E+04, 1.0841726514394575E+05, -1.0841709685990328E+05, 5.9790206615067997E+04, -6.9607049368128291E+03, -1.5834783935893831E+04, 1.3124692974990443E+04, -4.9454295091588992E+03, 8.9083493794871868E+02, -3.1189837631106176E+01};
    FLT c14[] = {-1.2975319073401824E+01, 1.8283698218710011E+01, 1.7684015393859755E+02, -1.1059917445033070E+03, 3.1998168298121523E+03, -5.5988200120063057E+03, 5.9248751921324047E+03, -2.5990022806343668E+03, -2.5990962125709430E+03, 5.9247537039895724E+03, -5.5988835070734467E+03, 3.1998292349030621E+03, -1.1059926481090836E+03, 1.7684013881079576E+02, 1.8283698123134819E+01, -1.2975319073977776E+01};
    FLT c15[] = {-2.3155118729954247E+00, 1.1938503634469159E+01, -3.4150562973753665E+01, 4.8898615554511437E+01, 1.5853185548633874E+01, -2.4272678107130790E+02, 6.0151276286907887E+02, -8.8751856926690448E+02, 8.8742942550355474E+02, -6.0136491467620624E+02, 2.4282489356694586E+02, -1.5850195971204462E+01, -4.8897392545563044E+01, 3.4150562973753665E+01, -1.1938504430698943E+01, 2.3155118723150525E+00};
    FLT c16[] = {-1.5401723686076832E-01, 9.8067823888634464E-01, -4.1900843552415639E+00, 1.2150534299778382E+01, -2.4763139606227178E+01, 3.6068014621628578E+01, -3.4346647779134791E+01, 1.3259903958585387E+01, 1.2937147675617604E+01, -3.4454233206790519E+01, 3.6027670086257579E+01, -2.4769863695455662E+01, 1.2149431128889342E+01, -4.1901615115388706E+00, 9.8067695636810759E-01, -1.5401723756214594E-01};
    FLT c17[] = {1.1808835093099178E-02, -2.5444299558662394E-02, -1.5661344238792723E-04, 2.5820071204205225E-01, -1.0930950485268096E+00, 2.6408492552008669E+00, -4.4415763059111955E+00, 6.8227366238712817E+00, -6.8186662643534008E+00, 4.4887924763186051E+00, -2.6327085361651021E+00, 1.0918739406714428E+00, -2.5844238963842503E-01, 1.2680123888735934E-04, 2.5444206395526567E-02, -1.1808834826225629E-02};
    for (int i=0; i<16; i++) ker[i] = c0[i] + z*(c1[i] + z*(c2[i] + z*(c3[i] + z*(c4[i] + z*(c5[i] + z*(c6[i] + z*(c7[i] + z*(c8[i] + z*(c9[i] + z*(c10[i] + z*(c11[i] + z*(c12[i] + z*(c13[i] + z*(c14[i] + z*(c15[i] + z*(c16[i] + z*(c17[i])))))))))))))))));
  } else
    printf("width not implemented!\n");
    } else if (opts.upsampfac==1.25) {
// Code generated by gen_all_horner_C_code.m in finufft/devel
// Authors: Alex Barnett & Ludvig af Klinteberg.
// (C) The Simons Foundation, Inc.
  if (w==2) {
    FLT c0[] = {2.3711015472112514E+01, 2.3711015472112514E+01, 0.0000000000000000E+00, 0.0000000000000000E+00};
    FLT c1[] = {2.5079742199350562E+01, -2.5079742199350562E+01, 0.0000000000000000E+00, 0.0000000000000000E+00};
    FLT c2[] = {-3.5023281580177050E+00, -3.5023281580177086E+00, 0.0000000000000000E+00, 0.0000000000000000E+00};
    FLT c3[] = {-7.3894949249195587E+00, 7.3894949249195632E+00, 0.0000000000000000E+00, 0.0000000000000000E+00};
    for (int i=0; i<4; i++) ker[i] = c0[i] + z*(c1[i] + z*(c2[i] + z*(c3[i])));
  } else if (w==3) {
    FLT c0[] = {5.9620016143346824E+01, 2.4110216701187497E+02, 5.9620016148621815E+01, 0.0000000000000000E+00};
    FLT c1[] = {9.7575520958604258E+01, 9.4807967775797928E-16, -9.7575520952908519E+01, 0.0000000000000000E+00};
    FLT c2[] = {3.5838417859768512E+01, -7.3472145274965371E+01, 3.5838417865129472E+01, 0.0000000000000000E+00};
    FLT c3[] = {-1.0721643298166471E+01, -2.1299978194824344E-16, 1.0721643303220413E+01, 0.0000000000000000E+00};
    FLT c4[] = {-7.0570630207138318E+00, 9.1538553399011260E+00, -7.0570630151506633E+00, 0.0000000000000000E+00};
    for (int i=0; i<4; i++) ker[i] = c0[i] + z*(c1[i] + z*(c2[i] + z*(c3[i] + z*(c4[i]))));
  } else if (w==4) {
    FLT c0[] = {1.2612470018753689E+02, 1.1896204292999116E+03, 1.1896204292999118E+03, 1.2612470018753696E+02};
    FLT c1[] = {2.6158034850676626E+02, 5.6161104654809810E+02, -5.6161104654809844E+02, -2.6158034850676620E+02};
    FLT c2[] = {1.7145379463699527E+02, -1.6695967127766517E+02, -1.6695967127766514E+02, 1.7145379463699527E+02};
    FLT c3[] = {2.3525961965887870E+01, -1.0057439659768858E+02, 1.0057439659768873E+02, -2.3525961965887827E+01};
    FLT c4[] = {-1.5608307370340880E+01, 9.5627412100260845E+00, 9.5627412100260205E+00, -1.5608307370340908E+01};
    FLT c5[] = {-4.5715207776748699E+00, 7.9904373067895493E+00, -7.9904373067893877E+00, 4.5715207776749462E+00};
    for (int i=0; i<4; i++) ker[i] = c0[i] + z*(c1[i] + z*(c2[i] + z*(c3[i] + z*(c4[i] + z*(c5[i])))));
  } else if (w==5) {
    FLT c0[] = {2.4106943677442615E+02, 4.3538384278025542E+03, 9.3397486707381995E+03, 4.3538384278025515E+03, 2.4106943677442607E+02, 0.0000000000000000E+00, 0.0000000000000000E+00, 0.0000000000000000E+00};
    FLT c1[] = {5.8781364250328272E+02, 3.4742855804122028E+03, -7.3041306797303120E-14, -3.4742855804122009E+03, -5.8781364250328249E+02, 0.0000000000000000E+00, 0.0000000000000000E+00, 0.0000000000000000E+00};
    FLT c2[] = {5.1234107167555862E+02, 3.5219546517037116E+02, -1.7076861141633149E+03, 3.5219546517037247E+02, 5.1234107167555862E+02, 0.0000000000000000E+00, 0.0000000000000000E+00, 0.0000000000000000E+00};
    FLT c3[] = {1.7540956907856057E+02, -3.5792356187777074E+02, -4.9888896652511712E-13, 3.5792356187777165E+02, -1.7540956907856059E+02, 0.0000000000000000E+00, 0.0000000000000000E+00, 0.0000000000000000E+00};
    FLT c4[] = {-2.1768066955094961E-01, -7.8322173187697558E+01, 1.3904039464934516E+02, -7.8322173187697842E+01, -2.1768066955103071E-01, 0.0000000000000000E+00, 0.0000000000000000E+00, 0.0000000000000000E+00};
    FLT c5[] = {-1.4207955403641256E+01, 1.6019466986221790E+01, 5.4386376890865855E-13, -1.6019466986220916E+01, 1.4207955403641320E+01, 0.0000000000000000E+00, 0.0000000000000000E+00, 0.0000000000000000E+00};
    FLT c6[] = {-2.1966493586753826E+00, 5.0672636163194582E+00, -6.7340544905090631E+00, 5.0672636163189448E+00, -2.1966493586753089E+00, 0.0000000000000000E+00, 0.0000000000000000E+00, 0.0000000000000000E+00};
    for (int i=0; i<8; i++) ker[i] = c0[i] + z*(c1[i] + z*(c2[i] + z*(c3[i] + z*(c4[i] + z*(c5[i] + z*(c6[i]))))));
  } else if (w==6) {
    FLT c0[] = {4.3011762559089101E+02, 1.3368828836127070E+04, 4.9861340433371224E+04, 4.9861340433371253E+04, 1.3368828836127073E+04, 4.3011762559835148E+02, 0.0000000000000000E+00, 0.0000000000000000E+00};
    FLT c1[] = {1.1857225840065141E+03, 1.4112553227730617E+04, 1.5410005180819440E+04, -1.5410005180819426E+04, -1.4112553227730616E+04, -1.1857225839984601E+03, 0.0000000000000000E+00, 0.0000000000000000E+00};
    FLT c2[] = {1.2460481448413077E+03, 4.3127030215084960E+03, -5.5438591621431169E+03, -5.5438591621431306E+03, 4.3127030215084960E+03, 1.2460481448488902E+03, 0.0000000000000000E+00, 0.0000000000000000E+00};
    FLT c3[] = {6.0825549344387753E+02, -3.4106010789547094E+02, -1.9775725023673197E+03, 1.9775725023673208E+03, 3.4106010789547116E+02, -6.0825549343673094E+02, 0.0000000000000000E+00, 0.0000000000000000E+00};
    FLT c4[] = {1.1264961069783706E+02, -3.9740822717991142E+02, 2.7557540616463064E+02, 2.7557540616462472E+02, -3.9740822717991210E+02, 1.1264961070570448E+02, 0.0000000000000000E+00, 0.0000000000000000E+00};
    FLT c5[] = {-1.5387906304333878E+01, -3.2640579296387394E+01, 1.1683718215647470E+02, -1.1683718215646800E+02, 3.2640579296390861E+01, 1.5387906311562851E+01, 0.0000000000000000E+00, 0.0000000000000000E+00};
    FLT c6[] = {-9.3947198873910249E+00, 1.5069930500881778E+01, -8.0900452409597179E+00, -8.0900452409538364E+00, 1.5069930500884301E+01, -9.3947198802581902E+00, 0.0000000000000000E+00, 0.0000000000000000E+00};
    FLT c7[] = {-5.6048841964539509E-01, 2.3377422080924530E+00, -4.2391567591836514E+00, 4.2391567591841817E+00, -2.3377422080928629E+00, 5.6048842664294984E-01, 0.0000000000000000E+00, 0.0000000000000000E+00};
    for (int i=0; i<8; i++) ker[i] = c0[i] + z*(c1[i] + z*(c2[i] + z*(c3[i] + z*(c4[i] + z*(c5[i] + z*(c6[i] + z*(c7[i])))))));
  } else if (w==7) {
    FLT c0[] = {7.2950392616203249E+02, 3.6439117038309480E+04, 2.1220891582018422E+05, 3.6180058567561524E+05, 2.1220891582018445E+05, 3.6439117038309487E+04, 7.2950392617434545E+02, 0.0000000000000000E+00};
    FLT c1[] = {2.2197790785452576E+03, 4.6392067080426248E+04, 1.1568051746995670E+05, -1.1902861988308852E-11, -1.1568051746995671E+05, -4.6392067080426241E+04, -2.2197790785319785E+03, 0.0000000000000000E+00};
    FLT c2[] = {2.6796845075663955E+03, 2.0921129984587249E+04, 3.9399551345574849E+01, -4.7251335435527435E+04, 3.9399551345580633E+01, 2.0921129984587245E+04, 2.6796845075789142E+03, 0.0000000000000000E+00};
    FLT c3[] = {1.6253748990844499E+03, 2.6138488347211564E+03, -1.0037546705421508E+04, 2.6823166126907972E-11, 1.0037546705421508E+04, -2.6138488347211546E+03, -1.6253748990726619E+03, 0.0000000000000000E+00};
    FLT c4[] = {4.9106375852553418E+02, -8.6668269315416171E+02, -1.0513434716618249E+03, 2.8444456471590756E+03, -1.0513434716618387E+03, -8.6668269315416057E+02, 4.9106375853851472E+02, 0.0000000000000000E+00};
    FLT c5[] = {4.0739167949763157E+01, -2.8515155742293922E+02, 3.9930326803801455E+02, 2.4847312048933061E-11, -3.9930326803798215E+02, 2.8515155742293899E+02, -4.0739167937835738E+01, 0.0000000000000000E+00};
    FLT c6[] = {-1.7148987139838667E+01, 7.5799002551700223E-01, 6.3260304953160343E+01, -1.0529869309160161E+02, 6.3260304953194023E+01, 7.5799002552709915E-01, -1.7148987128069749E+01, 0.0000000000000000E+00};
    FLT c7[] = {-4.5424411501060264E+00, 9.8749254058318616E+00, -9.6456179777547195E+00, 2.0621161109877312E-11, 9.6456179778118027E+00, -9.8749254058319202E+00, 4.5424411616514604E+00, 0.0000000000000000E+00};
    FLT c8[] = {-5.0793946806832954E-02, 7.3273813711856639E-01, -2.0117140544738263E+00, 2.6999257940856816E+00, -2.0117140545416512E+00, 7.3273813711318592E-01, -5.0793935653327994E-02, 0.0000000000000000E+00};
    for (int i=0; i<8; i++) ker[i] = c0[i] + z*(c1[i] + z*(c2[i] + z*(c3[i] + z*(c4[i] + z*(c5[i] + z*(c6[i] + z*(c7[i] + z*(c8[i]))))))));
  } else if (w==8) {
    FLT c0[] = {1.1895823653767145E+03, 9.0980236725236929E+04, 7.7438826909537485E+05, 2.0077596413122697E+06, 2.0077596413122697E+06, 7.7438826909537497E+05, 9.0980236725236929E+04, 1.1895823653767147E+03};
    FLT c1[] = {3.9313191526977798E+03, 1.3318570706800820E+05, 5.7275848637687636E+05, 4.6250273225257988E+05, -4.6250273225257976E+05, -5.7275848637687659E+05, -1.3318570706800820E+05, -3.9313191526977798E+03};
    FLT c2[] = {5.2976026193612370E+03, 7.5628970871188430E+04, 1.0073339198368321E+05, -1.8165150843791291E+05, -1.8165150843791291E+05, 1.0073339198368321E+05, 7.5628970871188460E+04, 5.2976026193612397E+03};
    FLT c3[] = {3.7552239608473842E+03, 1.8376340228970901E+04, -2.3878081117551585E+04, -4.6296734056047833E+04, 4.6296734056048226E+04, 2.3878081117551632E+04, -1.8376340228970901E+04, -3.7552239608473833E+03};
    FLT c4[] = {1.4742862505418652E+03, 1.2842168112178376E+02, -9.1969665138398723E+03, 7.5990739935234687E+03, 7.5990739935234151E+03, -9.1969665138399178E+03, 1.2842168112178072E+02, 1.4742862505418645E+03};
    FLT c5[] = {2.8158981009344416E+02, -8.8613607108855206E+02, 5.3457145342334378E+01, 2.1750989694614777E+03, -2.1750989694609211E+03, -5.3457145342173561E+01, 8.8613607108856843E+02, -2.8158981009344393E+02};
    FLT c6[] = {-1.4786862436240726E+00, -1.3935442261830281E+02, 3.2599325739083491E+02, -1.9541889343332295E+02, -1.9541889343339443E+02, 3.2599325739083696E+02, -1.3935442261827953E+02, -1.4786862436237442E+00};
    FLT c7[] = {-1.1542034522902307E+01, 1.2000512051397084E+01, 1.9687328710129744E+01, -6.3962883082482271E+01, 6.3962883082874910E+01, -1.9687328710101575E+01, -1.2000512051407391E+01, 1.1542034522902124E+01};
    FLT c8[] = {-1.7448292513542445E+00, 4.8577330433956609E+00, -6.8794163043773890E+00, 3.4611708987408365E+00, 3.4611708985348386E+00, -6.8794163043605385E+00, 4.8577330433771184E+00, -1.7448292513550807E+00};
    FLT c9[] = {1.5044951479021193E-01, 9.6230159355094713E-02, -7.0399250398052082E-01, 1.3251401132916929E+00, -1.3251401128795544E+00, 7.0399250407339709E-01, -9.6230159355094713E-02, -1.5044951479003055E-01};
    for (int i=0; i<8; i++) ker[i] = c0[i] + z*(c1[i] + z*(c2[i] + z*(c3[i] + z*(c4[i] + z*(c5[i] + z*(c6[i] + z*(c7[i] + z*(c8[i] + z*(c9[i])))))))));
  } else if (w==9) {
    FLT c0[] = {1.8793738965776997E+03, 2.1220891582018419E+05, 2.5233246441351641E+06, 9.2877384983420596E+06, 1.4015330434461458E+07, 9.2877384983420689E+06, 2.5233246441351632E+06, 2.1220891582018507E+05, 1.8793738965777015E+03, 0.0000000000000000E+00, 0.0000000000000000E+00, 0.0000000000000000E+00};
    FLT c1[] = {6.6675066501609344E+03, 3.4704155240986997E+05, 2.2890184838322559E+06, 3.8705035445351214E+06, -1.6037058324963857E-09, -3.8705035445351251E+06, -2.2890184838322555E+06, -3.4704155240987107E+05, -6.6675066501609363E+03, 0.0000000000000000E+00, 0.0000000000000000E+00, 0.0000000000000000E+00};
    FLT c2[] = {9.8412775404612330E+03, 2.3171563090202375E+05, 6.8167589492092200E+05, -2.1140963571671984E+05, -1.4236515118873848E+06, -2.1140963571672366E+05, 6.8167589492092165E+05, 2.3171563090202425E+05, 9.8412775404612312E+03, 0.0000000000000000E+00, 0.0000000000000000E+00, 0.0000000000000000E+00};
    FLT c3[] = {7.8762358364031033E+03, 7.6500585979636104E+04, 1.2434778984075023E+04, -2.8572091469430045E+05, 1.5952874106327477E-09, 2.8572091469430359E+05, -1.2434778984075045E+04, -7.6500585979636220E+04, -7.8762358364031052E+03, 0.0000000000000000E+00, 0.0000000000000000E+00, 0.0000000000000000E+00};
    FLT c4[] = {3.6941911906762084E+03, 9.9232929169975941E+03, -3.3472877669902169E+04, -1.4082384858052235E+04, 6.7911966136972551E+04, -1.4082384858047793E+04, -3.3472877669902322E+04, 9.9232929169976087E+03, 3.6941911906762070E+03, 0.0000000000000000E+00, 0.0000000000000000E+00, 0.0000000000000000E+00};
    FLT c5[] = {9.8900189723050266E+02, -1.2736589324621855E+03, -5.0407308390126955E+03, 9.8914296140171609E+03, 1.0742991696587890E-09, -9.8914296140222541E+03, 5.0407308390134704E+03, 1.2736589324621880E+03, -9.8900189723050198E+02, 0.0000000000000000E+00, 0.0000000000000000E+00, 0.0000000000000000E+00};
    FLT c6[] = {1.1165868717715853E+02, -5.9057035448564977E+02, 5.5860705835603983E+02, 9.1996097522959656E+02, -2.0290255886377897E+03, 9.1996097523001129E+02, 5.5860705835622480E+02, -5.9057035448564693E+02, 1.1165868717715870E+02, 0.0000000000000000E+00, 0.0000000000000000E+00, 0.0000000000000000E+00};
    FLT c7[] = {-1.3142584300868881E+01, -4.2852762793304592E+01, 1.8188640945795066E+02, -2.1362000457567430E+02, 6.1024810759112463E-10, 2.1362000457722939E+02, -1.8188640945795305E+02, 4.2852762793363922E+01, 1.3142584300866494E+01, 0.0000000000000000E+00, 0.0000000000000000E+00, 0.0000000000000000E+00};
    FLT c8[] = {-5.8088068374878068E+00, 1.0201832931362965E+01, -3.5220973519213472E-01, -2.6632420896811951E+01, 4.2737607182672249E+01, -2.6632420895534445E+01, -3.5220973562147767E-01, 1.0201832931230712E+01, -5.8088068374901178E+00, 0.0000000000000000E+00, 0.0000000000000000E+00, 0.0000000000000000E+00};
    FLT c9[] = {-4.0642645973308456E-01, 1.8389772328416343E+00, -3.5549484953682806E+00, 3.2273562233914270E+00, 1.3413454081272250E-09, -3.2273562258526494E+00, 3.5549484959023196E+00, -1.8389772328242200E+00, 4.0642645973371377E-01, 0.0000000000000000E+00, 0.0000000000000000E+00, 0.0000000000000000E+00};
    for (int i=0; i<12; i++) ker[i] = c0[i] + z*(c1[i] + z*(c2[i] + z*(c3[i] + z*(c4[i] + z*(c5[i] + z*(c6[i] + z*(c7[i] + z*(c8[i] + z*(c9[i])))))))));
  } else if (w==10) {
    FLT c0[] = {2.8923571298063562E+03, 4.6856831608341925E+05, 7.5304732752870023E+06, 3.7576537584215783E+07, 7.9591606307847857E+07, 7.9591606307847857E+07, 3.7576537584215745E+07, 7.5304732752870042E+06, 4.6856831608341780E+05, 2.8923571298063575E+03, 0.0000000000000000E+00, 0.0000000000000000E+00};
    FLT c1[] = {1.0919387804943191E+04, 8.3976685277206497E+05, 7.9494027659552367E+06, 2.1606786285174552E+07, 1.4625897641453246E+07, -1.4625897641453277E+07, -2.1606786285174549E+07, -7.9494027659552367E+06, -8.3976685277206241E+05, -1.0919387804943171E+04, 0.0000000000000000E+00, 0.0000000000000000E+00};
    FLT c2[] = {1.7418455635504150E+04, 6.3489952164419880E+05, 3.1358985409389879E+06, 2.2547438801903646E+06, -6.0429762783920728E+06, -6.0429762783920513E+06, 2.2547438801903692E+06, 3.1358985409389860E+06, 6.3489952164419706E+05, 1.7418455635504110E+04, 0.0000000000000000E+00, 0.0000000000000000E+00};
    FLT c3[] = {1.5396188098732160E+04, 2.5490607173283451E+05, 4.2818880748176615E+05, -9.5435463094349275E+05, -1.2004850139039254E+06, 1.2004850139039545E+06, 9.5435463094349345E+05, -4.2818880748176581E+05, -2.5490607173283395E+05, -1.5396188098732138E+04, 0.0000000000000000E+00, 0.0000000000000000E+00};
    FLT c4[] = {8.2616700456447434E+03, 5.2880641964112285E+04, -6.1165055141131161E+04, -2.1590299490711108E+05, 2.1595822052157650E+05, 2.1595822052157007E+05, -2.1590299490713840E+05, -6.1165055141131197E+04, 5.2880641964112183E+04, 8.2616700456447306E+03, 0.0000000000000000E+00, 0.0000000000000000E+00};
    FLT c5[] = {2.7267169079066489E+03, 2.4572549134030801E+03, -2.6065821571078384E+04, 1.3919259807559451E+04, 4.6802084705699206E+04, -4.6802084705714289E+04, -1.3919259807536537E+04, 2.6065821571078890E+04, -2.4572549134029036E+03, -2.7267169079066425E+03, 0.0000000000000000E+00, 0.0000000000000000E+00};
    FLT c6[] = {5.0402062537834070E+02, -1.3640153425625381E+03, -1.4063198459019245E+03, 7.0858129627834105E+03, -4.8375233777605163E+03, -4.8375233777670810E+03, 7.0858129627894641E+03, -1.4063198459014579E+03, -1.3640153425626913E+03, 5.0402062537833700E+02, 0.0000000000000000E+00, 0.0000000000000000E+00};
    FLT c7[] = {2.4199726682542348E+01, -2.8393731159249540E+02, 5.1652001352543709E+02, 7.4578914842705018E+01, -1.1556759026365337E+03, 1.1556759026651935E+03, -7.4578914839714216E+01, -5.1652001352595710E+02, 2.8393731159268043E+02, -2.4199726682540959E+01, 0.0000000000000000E+00, 0.0000000000000000E+00};
    FLT c8[] = {-1.0545675122360885E+01, -3.0306758891224317E+00, 7.2305523762173834E+01, -1.3808908570221064E+02, 7.6293213403386517E+01, 7.6293213419205742E+01, -1.3808908572505672E+02, 7.2305523760424833E+01, -3.0306758894244412E+00, -1.0545675122369961E+01, 0.0000000000000000E+00, 0.0000000000000000E+00};
    FLT c9[] = {-2.1836930570474395E+00, 5.4992367509081630E+00, -4.5624617253163446E+00, -6.6492709819863256E+00, 2.0339240341691568E+01, -2.0339240351164950E+01, 6.6492710020476089E+00, 4.5624617253163446E+00, -5.4992367508501152E+00, 2.1836930570530630E+00, 0.0000000000000000E+00, 0.0000000000000000E+00};
    FLT c10[] = {-9.1748741459757727E-02, 5.2562451739588611E-01, -1.4144257958835973E+00, 1.8629578990262812E+00, -9.0169874554123419E-01, -9.0169876258108816E-01, 1.8629579026113960E+00, -1.4144257947447987E+00, 5.2562451738534777E-01, -9.1748741464373396E-02, 0.0000000000000000E+00, 0.0000000000000000E+00};
    for (int i=0; i<12; i++) ker[i] = c0[i] + z*(c1[i] + z*(c2[i] + z*(c3[i] + z*(c4[i] + z*(c5[i] + z*(c6[i] + z*(c7[i] + z*(c8[i] + z*(c9[i] + z*(c10[i]))))))))));
  } else if (w==11) {
    FLT c0[] = {4.3537972057094357E+03, 9.8872306817881018E+05, 2.0938056062983289E+07, 1.3701428307175827E+08, 3.8828289972017348E+08, 5.4292197128519189E+08, 3.8828289972017324E+08, 1.3701428307175821E+08, 2.0938056062983286E+07, 9.8872306817881099E+05, 4.3537972057093830E+03, 0.0000000000000000E+00};
    FLT c1[] = {1.7371472778611496E+04, 1.9155790709433770E+06, 2.4914432724618733E+07, 9.7792160665338323E+07, 1.3126779387874992E+08, 1.1003518489948497E-08, -1.3126779387874992E+08, -9.7792160665338367E+07, -2.4914432724618725E+07, -1.9155790709433774E+06, -1.7371472778611387E+04, 0.0000000000000000E+00};
    FLT c2[] = {2.9650558537745437E+04, 1.6014973065836846E+06, 1.1867448782239100E+07, 2.0812212822540633E+07, -1.1749875870571069E+07, -4.5121922350041404E+07, -1.1749875870571032E+07, 2.0812212822540659E+07, 1.1867448782239093E+07, 1.6014973065836851E+06, 2.9650558537745299E+04, 0.0000000000000000E+00};
    FLT c3[] = {2.8505604980264394E+04, 7.4166660874053277E+05, 2.5711466441825330E+06, -1.2146931938153899E+06, -8.3931576510116160E+06, -1.5221113764487218E-08, 8.3931576510117017E+06, 1.2146931938154220E+06, -2.5711466441825316E+06, -7.4166660874053324E+05, -2.8505604980264285E+04, 0.0000000000000000E+00};
    FLT c4[] = {1.7045632829988481E+04, 1.9785834209758078E+05, 8.6361403553701501E+04, -1.0584472412326147E+06, -1.3367486018960556E+05, 1.7818009619467217E+06, -1.3367486018952832E+05, -1.0584472412326441E+06, 8.6361403553699885E+04, 1.9785834209758087E+05, 1.7045632829988419E+04, 0.0000000000000000E+00};
    FLT c5[] = {6.5462464716912918E+03, 2.5347576368078855E+04, -7.5810878908805942E+04, -8.0774039751690128E+04, 2.5492801112955116E+05, 3.6655592491345995E-08, -2.5492801112950110E+05, 8.0774039751702396E+04, 7.5810878908810162E+04, -2.5347576368078677E+04, -6.5462464716912700E+03, 0.0000000000000000E+00};
    FLT c6[] = {1.5684149291082115E+03, -1.0302687059852267E+03, -1.3446845770824435E+04, 2.0814393480320545E+04, 1.4366994276523908E+04, -4.4581342385955380E+04, 1.4366994276463982E+04, 2.0814393480325110E+04, -1.3446845770824308E+04, -1.0302687059850016E+03, 1.5684149291082128E+03, 0.0000000000000000E+00};
    FLT c7[] = {1.9398419323286222E+02, -8.7329293867281388E+02, 2.4796533428938184E+02, 3.2905701326623416E+03, -4.8989871768459579E+03, 2.8861239463615327E-09, 4.8989871768722078E+03, -3.2905701326312101E+03, -2.4796533429068171E+02, 8.7329293867237629E+02, -1.9398419323287882E+02, 0.0000000000000000E+00};
    FLT c8[] = {-4.2288232505124679E+00, -9.9574929618003850E+01, 2.9563077146126534E+02, -1.9453049352240328E+02, -4.0107401572039475E+02, 7.9532514195009401E+02, -4.0107401576942334E+02, -1.9453049354949908E+02, 2.9563077145563869E+02, -9.9574929618160851E+01, -4.2288232505049734E+00, 0.0000000000000000E+00};
    FLT c9[] = {-5.3741131162167548E+00, 5.5350606003782072E+00, 1.9153744596147156E+01, -6.3189447483342484E+01, 6.6921287710344444E+01, 2.6543499136172006E-08, -6.6921287588490713E+01, 6.3189447458080132E+01, -1.9153744593546620E+01, -5.5350606004478644E+00, 5.3741131162113120E+00, 0.0000000000000000E+00};
    FLT c10[] = {-7.0359426508237854E-01, 2.2229112757468452E+00, -3.2054079720618520E+00, 8.3392526913327172E-02, 6.8879260281453520E+00, -1.0795498333352139E+01, 6.8879260220718077E+00, 8.3392507342704467E-02, -3.2054079702060019E+00, 2.2229112757257625E+00, -7.0359426507941902E-01, 0.0000000000000000E+00};
    FLT c11[] = {5.2648094861126392E-02, 9.9912561389764148E-02, -4.3913938527232693E-01, 7.9792987484770361E-01, -6.9191816827427566E-01, -1.2022534526020762E-09, 6.9191820562024531E-01, -7.9792984883890594E-01, 4.3913938443394634E-01, -9.9912561446925147E-02, -5.2648094869462925E-02, 0.0000000000000000E+00};
    for (int i=0; i<12; i++) ker[i] = c0[i] + z*(c1[i] + z*(c2[i] + z*(c3[i] + z*(c4[i] + z*(c5[i] + z*(c6[i] + z*(c7[i] + z*(c8[i] + z*(c9[i] + z*(c10[i] + z*(c11[i])))))))))));
  } else if (w==12) {
    FLT c0[] = {6.4299692685485315E+03, 2.0077596413122714E+06, 5.4904521978991628E+07, 4.5946106674819350E+08, 1.6835469840840104E+09, 3.1308386544851556E+09, 3.1308386544851556E+09, 1.6835469840840099E+09, 4.5946106674819458E+08, 5.4904521978991754E+07, 2.0077596413122730E+06, 6.4299692685634491E+03};
    FLT c1[] = {2.6965848540274073E+04, 4.1625245902732178E+06, 7.2097002594596952E+07, 3.8505085985474640E+08, 7.9479013671674240E+08, 4.7870231281824082E+08, -4.7870231281824046E+08, -7.9479013671674252E+08, -3.8505085985474682E+08, -7.2097002594597101E+07, -4.1625245902732178E+06, -2.6965848540258085E+04};
    FLT c2[] = {4.8869694409905111E+04, 3.7863371066322513E+06, 3.9530526716552719E+07, 1.1475134266581042E+08, 4.6311261797930710E+07, -2.0442837194260675E+08, -2.0442837194260725E+08, 4.6311261797930680E+07, 1.1475134266581020E+08, 3.9530526716552787E+07, 3.7863371066322504E+06, 4.8869694409920470E+04};
    FLT c3[] = {5.0530564260114021E+04, 1.9615784087727289E+06, 1.1044597342441007E+07, 7.9812418612436540E+06, -3.4042228324588493E+07, -3.3301805987927791E+07, 3.3301805987928167E+07, 3.4042228324588671E+07, -7.9812418612435497E+06, -1.1044597342440993E+07, -1.9615784087727286E+06, -5.0530564260099913E+04};
    FLT c4[] = {3.3081876469965493E+04, 6.2011956881368335E+05, 1.3086001239863748E+06, -3.1165484297367339E+06, -5.1982996003442882E+06, 6.3530947749618590E+06, 6.3530947749616513E+06, -5.1982996003444213E+06, -3.1165484297366543E+06, 1.3086001239863599E+06, 6.2011956881368288E+05, 3.3081876469981333E+04};
    FLT c5[] = {1.4308966168506788E+04, 1.1375573205951916E+05, -1.0318195403424598E+05, -6.6892418721462542E+05, 5.9223570255461533E+05, 1.1093685152673351E+06, -1.1093685152666988E+06, -5.9223570255418238E+05, 6.6892418721489178E+05, 1.0318195403424004E+05, -1.1375573205951886E+05, -1.4308966168492358E+04};
    FLT c6[] = {4.0848961919700960E+03, 7.5033277163528910E+03, -5.2578904182711594E+04, 6.3431596329919275E+03, 1.5984798504282799E+05, -1.2521363434070408E+05, -1.2521363434057294E+05, 1.5984798504289921E+05, 6.3431596327853522E+03, -5.2578904182714803E+04, 7.5033277163530738E+03, 4.0848961919843541E+03};
    FLT c7[] = {7.1658797373677544E+02, -1.5499947984100402E+03, -4.5490740453241297E+03, 1.4520122796414065E+04, -3.7896465826366048E+03, -2.3597107892645658E+04, 2.3597107892708405E+04, 3.7896465828577311E+03, -1.4520122796272850E+04, 4.5490740453326107E+03, 1.5499947984094520E+03, -7.1658797372277388E+02};
    FLT c8[] = {5.2022749592533359E+01, -4.0624258132650436E+02, 5.2256582980122801E+02, 9.3282469962834807E+02, -2.8710622267611107E+03, 1.7594166903207245E+03, 1.7594166904840572E+03, -2.8710622269566602E+03, 9.3282469973848731E+02, 5.2256582976889342E+02, -4.0624258132718376E+02, 5.2022749606062760E+01};
    FLT c9[] = {-7.0341875498860729E+00, -2.3043166229077922E+01, 1.2279331781679724E+02, -1.6714687548507158E+02, -4.4746498424591195E+01, 3.6060906024962412E+02, -3.6060905985137049E+02, 4.4746498852565225E+01, 1.6714687549695972E+02, -1.2279331779599295E+02, 2.3043166228938606E+01, 7.0341875614861786E+00};
    FLT c10[] = {-2.1556100132617875E+00, 4.1361104009993737E+00, 1.8107701723532290E+00, -2.1223400322208619E+01, 3.5820961861882218E+01, -1.8782945665578143E+01, -1.8782945409136026E+01, 3.5820961915195049E+01, -2.1223400242576908E+01, 1.8107701298380314E+00, 4.1361104007462801E+00, -2.1556100021452793E+00};
    FLT c11[] = {-1.1440899376747954E-01, 7.0567641591060326E-01, -1.4530217904770133E+00, 1.0571984613482723E+00, 1.4389002957406878E+00, -4.2241732762744180E+00, 4.2241733421252539E+00, -1.4389000664821670E+00, -1.0571984509828731E+00, 1.4530218285851431E+00, -7.0567641613924970E-01, 1.1440900438178304E-01};
    FLT c12[] = {-1.4486009663463860E-02, 2.9387825785034223E-03, -1.0265969715607470E-01, 2.6748267835596640E-01, -3.3606430399849180E-01, 1.5850148085005597E-01, 1.5850183161365292E-01, -3.3606448814949358E-01, 2.6748281866164947E-01, -1.0265975004478733E-01, 2.9387817050372631E-03, -1.4486000369842612E-02};
    for (int i=0; i<12; i++) ker[i] = c0[i] + z*(c1[i] + z*(c2[i] + z*(c3[i] + z*(c4[i] + z*(c5[i] + z*(c6[i] + z*(c7[i] + z*(c8[i] + z*(c9[i] + z*(c10[i] + z*(c11[i] + z*(c12[i]))))))))))));
  } else if (w==13) {
    FLT c0[] = {9.3397060605267689E+03, 3.9447202186643109E+06, 1.3701428307175812E+08, 1.4375660883001409E+09, 6.6384519128895693E+09, 1.5848048271166529E+10, 2.1031560281976665E+10, 1.5848048271166502E+10, 6.6384519128895674E+09, 1.4375660883001378E+09, 1.3701428307175812E+08, 3.9447202186642843E+06, 9.3397060605268125E+03, 0.0000000000000000E+00, 0.0000000000000000E+00, 0.0000000000000000E+00};
    FLT c1[] = {4.0984512931817764E+04, 8.6828943763566799E+06, 1.9558432133067656E+08, 1.3674961320373521E+09, 3.9251291128182430E+09, 4.5116631434426517E+09, 4.8375356630808043E-07, -4.5116631434426460E+09, -3.9251291128182402E+09, -1.3674961320373492E+09, -1.9558432133067656E+08, -8.6828943763566278E+06, -4.0984512931817771E+04, 0.0000000000000000E+00, 0.0000000000000000E+00, 0.0000000000000000E+00};
    FLT c2[] = {7.8379538318778985E+04, 8.4928073133582603E+06, 1.1992091153966437E+08, 5.0561697705436689E+08, 6.1845897311593950E+08, -5.1306326495404470E+08, -1.4790096327029374E+09, -5.1306326495404077E+08, 6.1845897311593986E+08, 5.0561697705436659E+08, 1.1992091153966436E+08, 8.4928073133582156E+06, 7.8379538318778927E+04, 0.0000000000000000E+00, 0.0000000000000000E+00, 0.0000000000000000E+00};
    FLT c3[] = {8.6417670227040013E+04, 4.8250267333349697E+06, 3.9836803808039002E+07, 7.5026052902191013E+07, -7.7565422849560052E+07, -2.5393835488011825E+08, 5.1202971235247489E-07, 2.5393835488012013E+08, 7.7565422849558711E+07, -7.5026052902191967E+07, -3.9836803808039002E+07, -4.8250267333349511E+06, -8.6417670227039998E+04, 0.0000000000000000E+00, 0.0000000000000000E+00, 0.0000000000000000E+00};
    FLT c4[] = {6.1161604972829380E+04, 1.7331203720075535E+06, 7.0216196997558968E+06, -3.6027138646117523E+06, -3.1775875626364492E+07, 1.6544480876790185E+06, 4.9816566960114852E+07, 1.6544480876808946E+06, -3.1775875626363728E+07, -3.6027138646113039E+06, 7.0216196997558847E+06, 1.7331203720075490E+06, 6.1161604972829351E+04, 0.0000000000000000E+00, 0.0000000000000000E+00, 0.0000000000000000E+00};
    FLT c5[] = {2.9177164557155938E+04, 3.9318079134661221E+05, 3.1307448297760956E+05, -2.7571366584957433E+06, -9.8421840747392306E+05, 6.8469173866731795E+06, 2.9232946975263515E-06, -6.8469173866698397E+06, 9.8421840747792379E+05, 2.7571366584955421E+06, -3.1307448297758284E+05, -3.9318079134660971E+05, -2.9177164557155946E+04, 0.0000000000000000E+00, 0.0000000000000000E+00, 0.0000000000000000E+00};
    FLT c6[] = {9.5097815505886610E+03, 4.8799940773716655E+04, -1.2734023162441862E+05, -2.5472337176564379E+05, 6.3596049196278059E+05, 2.2361868201841635E+05, -1.0716559939651759E+06, 2.2361868202218774E+05, 6.3596049196161982E+05, -2.5472337176485342E+05, -1.2734023162441724E+05, 4.8799940773713337E+04, 9.5097815505886447E+03, 0.0000000000000000E+00, 0.0000000000000000E+00, 0.0000000000000000E+00};
    FLT c7[] = {2.0601715730545379E+03, 1.9365931141472569E+02, -2.5304303117518622E+04, 2.9151392447034210E+04, 5.9055020355306144E+04, -1.1784846181665688E+05, 1.1400011168699383E-06, 1.1784846181507374E+05, -5.9055020356297522E+04, -2.9151392447480976E+04, 2.5304303117520958E+04, -1.9365931141621550E+02, -2.0601715730545466E+03, 0.0000000000000000E+00, 0.0000000000000000E+00, 0.0000000000000000E+00};
    FLT c8[] = {2.5975061893404052E+02, -1.0025387650583972E+03, -6.8642481194759603E+02, 6.7515314205452096E+03, -7.0772939650079616E+03, -6.5444514139847633E+03, 1.6566898963381227E+04, -6.5444514164662887E+03, -7.0772939638053231E+03, 6.7515314202341915E+03, -6.8642481198706810E+02, -1.0025387650556635E+03, 2.5975061893403893E+02, 0.0000000000000000E+00, 0.0000000000000000E+00, 0.0000000000000000E+00};
    FLT c9[] = {5.8705282128634133E+00, -1.4424362302822419E+02, 3.3390627215295177E+02, 4.8151337640374301E+01, -1.1431733953039347E+03, 1.4557114789663567E+03, 1.9301282133401762E-06, -1.4557114797747520E+03, 1.1431733969207255E+03, -4.8151337212400264E+01, -3.3390627213809154E+02, 1.4424362302302313E+02, -5.8705282128808269E+00, 0.0000000000000000E+00, 0.0000000000000000E+00, 0.0000000000000000E+00};
    FLT c10[] = {-4.0954969508936898E+00, -1.2634947188543673E+00, 3.8134139835466350E+01, -8.4115524781317148E+01, 4.2766848228448069E+01, 1.0573434411021174E+02, -1.9636661067694894E+02, 1.0573435394677749E+02, 4.2766846813968300E+01, -8.4115525213218916E+01, 3.8134139824669184E+01, -1.2634947158177201E+00, -4.0954969509055461E+00, 0.0000000000000000E+00, 0.0000000000000000E+00, 0.0000000000000000E+00};
    FLT c11[] = {-6.2702735486285888E-01, 1.8595467772479546E+00, -1.3027978470952948E+00, -4.9265265903267785E+00, 1.3906831953385087E+01, -1.3753762586104637E+01, 1.0604155239584518E-06, 1.3753756761963198E+01, -1.3906831509501583E+01, 4.9265273268806409E+00, 1.3027978586801867E+00, -1.8595467797630916E+00, 6.2702735486047489E-01, 0.0000000000000000E+00, 0.0000000000000000E+00, 0.0000000000000000E+00};
    FLT c12[] = {-4.8290636703364975E-02, 1.7531876505199090E-01, -5.0041292774701596E-01, 6.3665145473474949E-01, -1.2476811514471326E-02, -1.2061603189510861E+00, 1.8595308638696268E+00, -1.2061633355215959E+00, -1.2475969680262359E-02, 6.3665088474340670E-01, -5.0041295405456876E-01, 1.7531876799797264E-01, -4.8290636708721864E-02, 0.0000000000000000E+00, 0.0000000000000000E+00, 0.0000000000000000E+00};
    FLT c13[] = {2.2894665617766322E-02, -7.1358257229878720E-03, -1.4950743217821900E-02, 7.0611745711086651E-02, -1.2311302279978055E-01, 1.0342573392772816E-01, 5.7346192890547669E-07, -1.0342709034448951E-01, 1.2311300937219723E-01, -7.0611830251417942E-02, 1.4950741891648016E-02, 7.1358203725587141E-03, -2.2894665628191136E-02, 0.0000000000000000E+00, 0.0000000000000000E+00, 0.0000000000000000E+00};
    for (int i=0; i<16; i++) ker[i] = c0[i] + z*(c1[i] + z*(c2[i] + z*(c3[i] + z*(c4[i] + z*(c5[i] + z*(c6[i] + z*(c7[i] + z*(c8[i] + z*(c9[i] + z*(c10[i] + z*(c11[i] + z*(c12[i] + z*(c13[i])))))))))))));
  } else if (w==14) {
    FLT c0[] = {1.3368785683552904E+04, 7.5304732752870144E+06, 3.2765764524434990E+08, 4.2418096936485257E+09, 2.4197690538177525E+10, 7.2227640697189651E+10, 1.2261475327356714E+11, 1.2261475327356711E+11, 7.2227640697189682E+10, 2.4197690538177582E+10, 4.2418096936485257E+09, 3.2765764524435169E+08, 7.5304732752870200E+06, 1.3368785683578039E+04, 0.0000000000000000E+00, 0.0000000000000000E+00};
    FLT c1[] = {6.1154444023081669E+04, 1.7488686085101541E+07, 5.0279014009863263E+08, 4.4777867842655849E+09, 1.6916819861812059E+10, 2.8971884004562843E+10, 1.6054555293734524E+10, -1.6054555293734529E+10, -2.8971884004562843E+10, -1.6916819861812090E+10, -4.4777867842655830E+09, -5.0279014009863406E+08, -1.7488686085101560E+07, -6.1154444023056145E+04, 0.0000000000000000E+00, 0.0000000000000000E+00};
    FLT c2[] = {1.2279790808348049E+05, 1.8230319600271538E+07, 3.3815815633683985E+08, 1.9369899011251254E+09, 3.9743454154781203E+09, 7.4954544638351786E+08, -7.0173920607395000E+09, -7.0173920607395000E+09, 7.4954544638351130E+08, 3.9743454154781117E+09, 1.9369899011251252E+09, 3.3815815633684093E+08, 1.8230319600271557E+07, 1.2279790808350699E+05, 0.0000000000000000E+00, 0.0000000000000000E+00};
    FLT c3[] = {1.4339321200624766E+05, 1.1200899688172188E+07, 1.2799140125169712E+08, 4.0176966726270604E+08, 7.9146174555810899E+07, -1.1719748245183561E+09, -9.6919138198233843E+08, 9.6919138198235476E+08, 1.1719748245183618E+09, -7.9146174555819452E+07, -4.0176966726270568E+08, -1.2799140125169776E+08, -1.1200899688172201E+07, -1.4339321200622554E+05, 0.0000000000000000E+00, 0.0000000000000000E+00};
    FLT c4[] = {1.0866548538632700E+05, 4.4565213401510641E+06, 2.8354150929531462E+07, 2.2805067924009934E+07, -1.2058223609889300E+08, -1.2775415620368913E+08, 1.9261201640091014E+08, 1.9261201640090343E+08, -1.2775415620368628E+08, -1.2058223609888241E+08, 2.2805067924009915E+07, 2.8354150929531943E+07, 4.4565213401510660E+06, 1.0866548538635390E+05, 0.0000000000000000E+00, 0.0000000000000000E+00};
    FLT c5[] = {5.6346565047794407E+04, 1.1743908345502375E+06, 3.0601086667309003E+06, -7.2274020134796975E+06, -1.6220595157143334E+07, 2.0773587344466623E+07, 2.8183198298701070E+07, -2.8183198298682313E+07, -2.0773587344454899E+07, 1.6220595157147046E+07, 7.2274020134809064E+06, -3.0601086667310768E+06, -1.1743908345502312E+06, -5.6346565047771022E+04, 0.0000000000000000E+00, 0.0000000000000000E+00};
    FLT c6[] = {2.0435142564639598E+04, 1.9450977300078847E+05, -1.1234667576926883E+05, -1.5205767549240857E+06, 1.0515640561047094E+06, 3.7458351782500809E+06, -3.3794074240119159E+06, -3.3794074240111569E+06, 3.7458351782506104E+06, 1.0515640561079446E+06, -1.5205767549239916E+06, -1.1234667576914738E+05, 1.9450977300078212E+05, 2.0435142564663307E+04, 0.0000000000000000E+00, 0.0000000000000000E+00};
    FLT c7[] = {5.1491366053560478E+03, 1.4735748500440239E+04, -8.1689482343683034E+04, -3.5176894225644079E+04, 3.7034248410400847E+05, -1.9109669530460562E+05, -5.2637978465735121E+05, 5.2637978465564619E+05, 1.9109669530912716E+05, -3.7034248412078863E+05, 3.5176894225852200E+04, 8.1689482343699274E+04, -1.4735748500439855E+04, -5.1491366053330485E+03, 0.0000000000000000E+00, 0.0000000000000000E+00};
    FLT c8[] = {8.5138795113645585E+02, -1.2978618911733427E+03, -8.7500873646623440E+03, 2.1319159613970569E+04, 7.6586611605801199E+03, -6.2424139811455236E+04, 4.2620771487921840E+04, 4.2620771491440872E+04, -6.2424139815176597E+04, 7.6586611693937375E+03, 2.1319159613447209E+04, -8.7500873648877496E+03, -1.2978618911701635E+03, 8.5138795115875257E+02, 0.0000000000000000E+00, 0.0000000000000000E+00};
    FLT c9[] = {7.2176142041616245E+01, -4.5543406155008586E+02, 2.8301959891624585E+02, 2.1994171513769957E+03, -4.5082500677203352E+03, 4.7658016853354945E+02, 7.1044827209848581E+03, -7.1044827023442112E+03, -4.7658015978385805E+02, 4.5082500694322307E+03, -2.1994171506161529E+03, -2.8301959873197922E+02, 4.5543406154525627E+02, -7.2176142022451799E+01, 0.0000000000000000E+00, 0.0000000000000000E+00};
    FLT c10[] = {-3.1135380163286266E+00, -3.8554406982628045E+01, 1.4396028111579378E+02, -1.1260050352192819E+02, -3.0073665460436297E+02, 7.2079162225452933E+02, -4.1195308319958349E+02, -4.1195308907344031E+02, 7.2079162228692246E+02, -3.0073665296314113E+02, -1.1260050391063737E+02, 1.4396028095922969E+02, -3.8554406981953719E+01, -3.1135379980309104E+00, 0.0000000000000000E+00, 0.0000000000000000E+00};
    FLT c11[] = {-1.6022934776950781E+00, 1.8678197421257499E+00, 8.3368944138930576E+00, -3.0791578217513287E+01, 3.4749712345962102E+01, 1.2322522680262193E+01, -7.3924006859338746E+01, 7.3924005395986399E+01, -1.2322518095091780E+01, -3.4749717239655702E+01, 3.0791578812609753E+01, -8.3368942651188451E+00, -1.8678197375527952E+00, 1.6022934952009980E+00, 0.0000000000000000E+00, 0.0000000000000000E+00};
    FLT c12[] = {-1.9362061840948824E-01, 6.3024467669748396E-01, -9.3262278519229969E-01, -4.8908749318740480E-01, 4.0479376609320967E+00, -6.2829712900962678E+00, 3.1767825933699174E+00, 3.1767865219197975E+00, -6.2829777441520323E+00, 4.0479394849078085E+00, -4.8908801933495105E-01, -9.3262306580362497E-01, 6.3024467258732675E-01, -1.9362060312142931E-01, 0.0000000000000000E+00, 0.0000000000000000E+00};
    FLT c13[] = {1.8785913718903639E-02, 3.1605271252714680E-02, -1.3655798291459853E-01, 2.5016547139148904E-01, -1.6654308552073466E-01, -2.1682598043284024E-01, 6.1786085249849709E-01, -6.1785470804340159E-01, 2.1682794765059335E-01, 1.6654258378326353E-01, -2.5016523395036322E-01, 1.3655803190024704E-01, -3.1605272440421092E-02, -1.8785905282938619E-02, 0.0000000000000000E+00, 0.0000000000000000E+00};
    FLT c14[] = {-1.2896545140952162E-02, -3.7106972352948116E-03, 5.8857860695711909E-04, 1.3987176343065890E-02, -3.5714007561179102E-02, 4.3401590960273219E-02, -2.0034532372716081E-02, -2.0038454375630149E-02, 4.3401322628411031E-02, -3.5713348533616053E-02, 1.3987046090052241E-02, 5.8856319054218355E-04, -3.7106979912720915E-03, -1.2896537385752806E-02, 0.0000000000000000E+00, 0.0000000000000000E+00};
    for (int i=0; i<16; i++) ker[i] = c0[i] + z*(c1[i] + z*(c2[i] + z*(c3[i] + z*(c4[i] + z*(c5[i] + z*(c6[i] + z*(c7[i] + z*(c8[i] + z*(c9[i] + z*(c10[i] + z*(c11[i] + z*(c12[i] + z*(c13[i] + z*(c14[i]))))))))))))));
  } else if (w==15) {
    FLT c0[] = {1.8887777774374499E+04, 1.4015330434461417E+07, 7.5498683300180018E+08, 1.1900937739619951E+10, 8.2530965279375351E+10, 3.0178246269069604E+11, 6.3775691457119104E+11, 8.1471473119305554E+11, 6.3775691457119116E+11, 3.0178246269069641E+11, 8.2530965279375519E+10, 1.1900937739619963E+10, 7.5498683300180054E+08, 1.4015330434461435E+07, 1.8887777774374488E+04, 0.0000000000000000E+00};
    FLT c1[] = {8.9780907163796335E+04, 3.4167636285297148E+07, 1.2346880033823481E+09, 1.3719272724135921E+10, 6.5858241494816696E+10, 1.5266999939989539E+11, 1.5687794513790723E+11, -2.8523584844088883E-05, -1.5687794513790732E+11, -1.5266999939989545E+11, -6.5858241494816811E+10, -1.3719272724135933E+10, -1.2346880033823476E+09, -3.4167636285297163E+07, -8.9780907163796335E+04, 0.0000000000000000E+00};
    FLT c2[] = {1.8850321233130712E+05, 3.7693640983013541E+07, 8.9846818051570034E+08, 6.7094088040439653E+09, 1.9743296615199215E+10, 1.8072727219391140E+10, -2.0634615374559410E+10, -4.9654335197177498E+10, -2.0634615374559414E+10, 1.8072727219391048E+10, 1.9743296615199223E+10, 6.7094088040439672E+09, 8.9846818051570022E+08, 3.7693640983013526E+07, 1.8850321233130703E+05, 0.0000000000000000E+00};
    FLT c3[] = {2.3185006533495727E+05, 2.4789475362741601E+07, 3.7751696829092383E+08, 1.7167916788178182E+09, 1.9832401267745295E+09, -3.4881359830884194E+09, -7.8785602379628601E+09, 6.6906528952995499E-05, 7.8785602379629536E+09, 3.4881359830884261E+09, -1.9832401267745163E+09, -1.7167916788178096E+09, -3.7751696829092425E+08, -2.4789475362741597E+07, -2.3185006533495730E+05, 0.0000000000000000E+00};
    FLT c4[] = {1.8672970114818285E+05, 1.0741068109706732E+07, 9.8017949708492473E+07, 2.0291084954252145E+08, -2.7857869294214898E+08, -9.4112677968756318E+08, 1.7886520649334356E+08, 1.4579673547891481E+09, 1.7886520649344125E+08, -9.4112677968753338E+08, -2.7857869294217581E+08, 2.0291084954251301E+08, 9.8017949708492488E+07, 1.0741068109706739E+07, 1.8672970114818282E+05, 0.0000000000000000E+00};
    FLT c5[] = {1.0411891611891470E+05, 3.1771463075269456E+06, 1.4880104152842037E+07, -6.8136965447538150E+06, -8.7072998215422541E+07, 1.8024116530863210E+06, 1.9067730799615666E+08, 1.2078175959365315E-04, -1.9067730799603686E+08, -1.8024116529155241E+06, 8.7072998215445980E+07, 6.8136965447565373E+06, -1.4880104152841812E+07, -3.1771463075269484E+06, -1.0411891611891470E+05, 0.0000000000000000E+00};
    FLT c6[] = {4.1300641422694731E+04, 6.3217168592497683E+05, 7.7343707634845132E+05, -5.4575962381476769E+06, -3.7387211063063843E+06, 1.8451583614082869E+07, 3.0480804948189310E+06, -2.7500445095872246E+07, 3.0480804948457484E+06, 1.8451583614064269E+07, -3.7387211062890980E+06, -5.4575962381450543E+06, 7.7343707634841127E+05, 6.3217168592497602E+05, 4.1300641422694724E+04, 0.0000000000000000E+00};
    FLT c7[] = {1.1710443348523711E+04, 7.5405449195716908E+04, -1.6634736996487752E+05, -5.6069290801842115E+05, 1.1540571563940533E+06, 1.0209821660925965E+06, -2.9641921942009293E+06, -7.3770236318814628E-06, 2.9641921942630685E+06, -1.0209821662946860E+06, -1.1540571563987043E+06, 5.6069290801928868E+05, 1.6634736996459437E+05, -7.5405449195719295E+04, -1.1710443348523739E+04, 0.0000000000000000E+00};
    FLT c8[] = {2.3142324239350210E+03, 2.1710560541703007E+03, -3.6929625713151705E+04, 2.6143898219588682E+04, 1.4046980090353978E+05, -2.1033190114896413E+05, -1.1132269819276403E+05, 3.7491447373940505E+05, -1.1132269820720138E+05, -2.1033190120894444E+05, 1.4046980085134835E+05, 2.6143898217223435E+04, -3.6929625713258414E+04, 2.1710560541651053E+03, 2.3142324239349791E+03, 0.0000000000000000E+00};
    FLT c9[] = {2.8879718294281940E+02, -9.2801372612866078E+02, -1.9817144428357562E+03, 9.9004179214302640E+03, -5.7928268996319048E+03, -2.1083466266548403E+04, 3.3285502001854453E+04, 1.3615676123196788E-04, -3.3285501884684672E+04, 2.1083466388283239E+04, 5.7928269528908959E+03, -9.9004179214302640E+03, 1.9817144428357562E+03, 9.2801372612624596E+02, -2.8879718294281940E+02, 0.0000000000000000E+00};
    FLT c10[] = {1.3121871131759899E+01, -1.5978845118014243E+02, 2.7429718889479011E+02, 4.4598059431432415E+02, -1.8917609556521720E+03, 1.5303002256342920E+03, 1.7542368404254241E+03, -3.9411530187890685E+03, 1.7542368839611659E+03, 1.5303002335812619E+03, -1.8917609760379448E+03, 4.4598059250034765E+02, 2.7429718872202716E+02, -1.5978845118149314E+02, 1.3121871131760223E+01, 0.0000000000000000E+00};
    FLT c11[] = {-2.4286151057622600E+00, -6.7839829150137421E+00, 4.6999223003107119E+01, -7.4896070454665107E+01, -3.2010110856873055E+01, 2.5022929107925501E+02, -2.8786053481345135E+02, 1.4424367379967129E-05, 2.8786057555317575E+02, -2.5022937123192844E+02, 3.2010139421505684E+01, 7.4896073537460509E+01, -4.6999223012862650E+01, 6.7839829186720362E+00, 2.4286151057336860E+00, 0.0000000000000000E+00};
    FLT c12[] = {-5.4810555665671257E-01, 1.1436870859674571E+00, 8.2471504792547190E-01, -8.5602131787584241E+00, 1.5631631237511966E+01, -6.4979395997142886E+00, -1.8737629118679905E+01, 3.3283673647767003E+01, -1.8737705444926284E+01, -6.4980552114725620E+00, 1.5631576798962341E+01, -8.5602158445716778E+00, 8.2471481116140977E-01, 1.1436870769250529E+00, -5.4810555667406624E-01, 0.0000000000000000E+00};
    FLT c13[] = {-1.4554612891837512E-02, 1.7022157398269799E-01, -3.7563892964814216E-01, 2.0131145240492249E-01, 8.3554123561642435E-01, -2.1191317631421946E+00, 1.9961007770939201E+00, 5.0230495487029605E-05, -1.9960655197919825E+00, 2.1191435815870405E+00, -8.3552330614378623E-01, -2.0131363341395125E-01, 3.7563890238546094E-01, -1.7022157734534860E-01, 1.4554612875194470E-02, 0.0000000000000000E+00};
    FLT c14[] = {-1.2348455978815665E-02, 2.6143485494326945E-03, -2.9252290291144727E-02, 7.5392101552106419E-02, -8.7986538697867239E-02, 1.3073120666751545E-03, 1.5251801232957554E-01, -2.3235618419546245E-01, 1.5253703942622115E-01, 1.3217162898956957E-03, -8.7999818995735196E-02, 7.5391507930594778E-02, -2.9252395603998178E-02, 2.6143483927929994E-03, -1.2348455970768767E-02, 0.0000000000000000E+00};
    FLT c15[] = {1.4214685591273772E-02, -1.2364346992375923E-03, 1.2892328724708124E-03, 1.6178725688327468E-03, -8.2104229475896996E-03, 1.3914679473447157E-02, -1.1426959041713501E-02, 1.6590583007947697E-05, 1.1446333966460217E-02, -1.3912124902889801E-02, 8.2298310485774198E-03, -1.6155336438419190E-03, -1.2892162843503102E-03, 1.2364372911314208E-03, -1.4214685607473108E-02, 0.0000000000000000E+00};
    for (int i=0; i<16; i++) ker[i] = c0[i] + z*(c1[i] + z*(c2[i] + z*(c3[i] + z*(c4[i] + z*(c5[i] + z*(c6[i] + z*(c7[i] + z*(c8[i] + z*(c9[i] + z*(c10[i] + z*(c11[i] + z*(c12[i] + z*(c13[i] + z*(c14[i] + z*(c15[i])))))))))))))));
  } else if (w==16) {
    FLT c0[] = {2.6374086784014689E+04, 2.5501413681212645E+07, 1.6835469840840099E+09, 3.1953580806547867E+10, 2.6584910126662766E+11, 1.1715858191494619E+12, 3.0181658330343120E+12, 4.7888775408612773E+12, 4.7888775408612764E+12, 3.0181658330343125E+12, 1.1715858191494619E+12, 2.6584910126662772E+11, 3.1953580806547874E+10, 1.6835469840840104E+09, 2.5501413681212656E+07, 2.6374086784014886E+04};
    FLT c1[] = {1.2991568388123445E+05, 6.4986154651133664E+07, 2.9142305012947259E+09, 3.9748054433728149E+10, 2.3649443248440247E+11, 7.0471088240421252E+11, 1.0533888905987031E+12, 5.4832304482297632E+11, -5.4832304482297687E+11, -1.0533888905987034E+12, -7.0471088240421265E+11, -2.3649443248440250E+11, -3.9748054433728149E+10, -2.9142305012947259E+09, -6.4986154651133649E+07, -1.2991568388123448E+05};
    FLT c2[] = {2.8421223836872831E+05, 7.5448503558118582E+07, 2.2710828032883868E+09, 2.1491603403163826E+10, 8.4299374042308136E+10, 1.3384457365769528E+11, 1.8630012765531485E+09, -2.4384536789321179E+11, -2.4384536789321094E+11, 1.8630012765532806E+09, 1.3384457365769531E+11, 8.4299374042308090E+10, 2.1491603403163826E+10, 2.2710828032883863E+09, 7.5448503558118552E+07, 2.8421223836872820E+05};
    FLT c3[] = {3.6653021243297518E+05, 5.2693428548387080E+07, 1.0410094433021281E+09, 6.3986267576853533E+09, 1.3313926739756302E+10, -2.7909761561128025E+09, -3.9911638977027977E+10, -2.9236947704012939E+10, 2.9236947704012939E+10, 3.9911638977028267E+10, 2.7909761561128430E+09, -1.3313926739756279E+10, -6.3986267576853561E+09, -1.0410094433021276E+09, -5.2693428548387088E+07, -3.6653021243297518E+05};
    FLT c4[] = {3.1185660915838118E+05, 2.4564274645530280E+07, 3.0509279143241835E+08, 1.0432225146182569E+09, 6.4966284440222360E+07, -4.2483903608016477E+09, -3.1778261722524829E+09, 5.9880587942832708E+09, 5.9880587942832832E+09, -3.1778261722526174E+09, -4.2483903608017979E+09, 6.4966284440235756E+07, 1.0432225146182607E+09, 3.0509279143241805E+08, 2.4564274645530272E+07, 3.1185660915838124E+05};
    FLT c5[] = {1.8544733523229562E+05, 7.9824949938292839E+06, 5.6880943382648192E+07, 5.4097201999258779E+07, -3.0776449202833223E+08, -3.7659931821867347E+08, 6.8797698944719648E+08, 7.5429896889866996E+08, -7.5429896889781320E+08, -6.8797698944658160E+08, 3.7659931821898031E+08, 3.0776449202837497E+08, -5.4097201999252096E+07, -5.6880943382647842E+07, -7.9824949938292857E+06, -1.8544733523229562E+05};
    FLT c6[] = {7.9472339236673259E+04, 1.8159676553648398E+06, 5.7259818806751696E+06, -1.2786136236423338E+07, -3.8677490873147681E+07, 4.7651450515707508E+07, 9.0723760109202415E+07, -9.4532949239946112E+07, -9.4532949239604995E+07, 9.0723760109522834E+07, 4.7651450515667401E+07, -3.8677490873160362E+07, -1.2786136236416934E+07, 5.7259818806752721E+06, 1.8159676553648538E+06, 7.9472339236673215E+04};
    FLT c7[] = {2.4831718998299857E+04, 2.7536301841716090E+05, -5.1045953356025166E+04, -2.6996387880239477E+06, 1.1656554632125401E+06, 9.1521923449522462E+06, -6.8198180925621921E+06, -1.2555197000954127E+07, 1.2555197001087580E+07, 6.8198180925775450E+06, -9.1521923449367471E+06, -1.1656554632051867E+06, 2.6996387880183556E+06, 5.1045953355832869E+04, -2.7536301841717580E+05, -2.4831718998299897E+04};
    FLT c8[] = {5.6060763597396035E+03, 2.2154740880101843E+04, -1.0243462874810334E+05, -1.1802198892388590E+05, 6.4061699367506150E+05, -1.1166716749369531E+05, -1.4153578101923370E+06, 1.0790712965214122E+06, 1.0790712965802078E+06, -1.4153578102569627E+06, -1.1166716767280686E+05, 6.4061699367841065E+05, -1.1802198892652121E+05, -1.0243462874831920E+05, 2.2154740880096295E+04, 5.6060763597396262E+03};
    FLT c9[] = {8.7271993222049730E+02, -7.0074676859193858E+02, -1.2528372958474913E+04, 2.3643101054370443E+04, 3.1699060146436736E+04, -1.1270133578294520E+05, 3.6872846840416030E+04, 1.5168911768972370E+05, -1.5168911672801850E+05, -3.6872846329129716E+04, 1.1270133600206790E+05, -3.1699060140349993E+04, -2.3643101053229180E+04, 1.2528372958403583E+04, 7.0074676858840917E+02, -8.7271993222049730E+02};
    FLT c10[] = {7.8842259458727298E+01, -4.2070880913717718E+02, -1.0535142166729695E+02, 3.3375056757602101E+03, -4.9426353709826744E+03, -3.6567309465694352E+03, 1.5199085032737788E+04, -9.4972226150681072E+03, -9.4972224492176338E+03, 1.5199085307902486E+04, -3.6567309714471071E+03, -4.9426353751288962E+03, 3.3375056795609726E+03, -1.0535142205602271E+02, -4.2070880913447866E+02, 7.8842259458701932E+01};
    FLT c11[] = {8.9833076760252317E-02, -4.4163371177310189E+01, 1.2880771175011134E+02, 2.8722208980881483E+00, -5.7164632401064989E+02, 9.0417621054583299E+02, 1.1221311957018894E+00, -1.4190922684153286E+03, 1.4190926436578332E+03, -1.1219382673482139E+00, -9.0417616902565715E+02, 5.7164633587355513E+02, -2.8722219907225899E+00, -1.2880771149646372E+02, 4.4163371174871045E+01, -8.9833076793553943E-02};
    FLT c12[] = {-1.0900468357304585E+00, -1.1264666580175993E-01, 1.1810668498718398E+01, -3.0289105594116332E+01, 1.5494599855921946E+01, 6.0130016326899806E+01, -1.2330195579557967E+02, 6.7114292010484860E+01, 6.7114238133033894E+01, -1.2330200967294053E+02, 6.0129899592769000E+01, 1.5494588631452897E+01, -3.0289108821162568E+01, 1.1810668060273379E+01, -1.1264668224327026E-01, -1.0900468357482698E+00};
    FLT c13[] = {-1.1763610124684608E-01, 4.2939195551308978E-01, -2.7950231695310290E-01, -1.7354597875532083E+00, 5.1181749794184972E+00, -5.0538409872852545E+00, -2.1268758321444312E+00, 1.0709572497394593E+01, -1.0709247944735344E+01, 2.1270284132327628E+00, 5.0538814533614023E+00, -5.1181783143082038E+00, 1.7354587260576941E+00, 2.7950208340719496E-01, -4.2939195720020440E-01, 1.1763610121354666E-01};
    FLT c14[] = {-1.8020499708490779E-02, 3.6694576081450124E-02, -1.1331174689418615E-01, 1.3970801507325420E-01, 8.1708800731612838E-02, -5.4465632012605969E-01, 7.9628723318194716E-01, -3.9045387765910361E-01, -3.9034731591396871E-01, 7.9641679205120786E-01, -5.4465236519348836E-01, 8.1709687544577886E-02, 1.3970913694934384E-01, -1.1331198385459386E-01, 3.6694575058947500E-02, -1.8020499699434717E-02};
    FLT c15[] = {1.4589783457723899E-02, -7.8885273589694921E-04, -4.4854775481901451E-03, 1.8117810622567232E-02, -3.0563678378015532E-02, 1.9027105036022670E-02, 2.4778670881552757E-02, -6.7767913155521747E-02, 6.7979444868167399E-02, -2.4638534439549119E-02, -1.8992900331546877E-02, 3.0569915511324409E-02, -1.8117279802711158E-02, 4.4857097818771776E-03, 7.8885377265448060E-04, -1.4589783469873403E-02};
    FLT c16[] = {-1.0467998068898355E-02, -3.2140568385029999E-04, 5.2979866592800886E-04, -1.5800624712947203E-04, -1.4200041949817279E-03, 3.7626007108648857E-03, -3.8348321381240775E-03, 1.6547563335740942E-03, 1.5759584129276946E-03, -3.8873640852216617E-03, 3.7166352571544989E-03, -1.4265706883689335E-03, -1.5923746463956793E-04, 5.2952292450647511E-04, -3.2141610431099765E-04, -1.0467998084554094E-02};
    for (int i=0; i<16; i++) ker[i] = c0[i] + z*(c1[i] + z*(c2[i] + z*(c3[i] + z*(c4[i] + z*(c5[i] + z*(c6[i] + z*(c7[i] + z*(c8[i] + z*(c9[i] + z*(c10[i] + z*(c11[i] + z*(c12[i] + z*(c13[i] + z*(c14[i] + z*(c15[i] + z*(c16[i]))))))))))))))));
  } else
    printf("width not implemented!\n");
    } else
      fprintf(stderr,"%s: unknown upsampfac, failed!\n",__func__);
  }
}

void interp_line(FLT *target,FLT *du, FLT *ker,BIGINT i1,BIGINT N1,int ns)
// 1D interpolate complex values from du array to out, using real weights
// ker[0] through ker[ns-1]. out must be size 2 (real,imag), and du
// of size 2*N1 (alternating real,imag). i1 is the left-most index in [0,N1)
// Periodic wrapping in the du array is applied, assuming N1>=ns.
// dx is index into ker array, j index in complex du (data_uniform) array.
// Barnett 6/15/17
{
  FLT out[] = {0.0, 0.0};
  BIGINT j = i1;
  if (i1<0) {                               // wraps at left
    j+=N1;
    for (int dx=0; dx<-i1; ++dx) {
      out[0] += du[2*j]*ker[dx];
      out[1] += du[2*j+1]*ker[dx];
      ++j;
    }
    j-=N1;
    for (int dx=-i1; dx<ns; ++dx) {
      out[0] += du[2*j]*ker[dx];
      out[1] += du[2*j+1]*ker[dx];
      ++j;
    }
  } else if (i1+ns>=N1) {                    // wraps at right
    for (int dx=0; dx<N1-i1; ++dx) {
      out[0] += du[2*j]*ker[dx];
      out[1] += du[2*j+1]*ker[dx];
      ++j;
    }
    j-=N1;
    for (int dx=N1-i1; dx<ns; ++dx) {
      out[0] += du[2*j]*ker[dx];
      out[1] += du[2*j+1]*ker[dx];
      ++j;
    }
  } else {                                     // doesn't wrap
    for (int dx=0; dx<ns; ++dx) {
      out[0] += du[2*j]*ker[dx];
      out[1] += du[2*j+1]*ker[dx];
      ++j;
    }
  }
  target[0] = out[0];
  target[1] = out[1];
}

void interp_square(FLT *target,FLT *du, FLT *ker1, FLT *ker2, BIGINT i1,BIGINT i2,BIGINT N1,BIGINT N2,int ns)
// 2D interpolate complex values from du (uniform grid data) array to out value,
// using ns*ns square of real weights
// in ker. out must be size 2 (real,imag), and du
// of size 2*N1*N2 (alternating real,imag). i1 is the left-most index in [0,N1)
// and i2 the bottom index in [0,N2).
// Periodic wrapping in the du array is applied, assuming N1,N2>=ns.
// dx,dy indices into ker array, j index in complex du array.
// Barnett 6/16/17
{
  FLT out[] = {0.0, 0.0};
  if (i1>=0 && i1+ns<=N1 && i2>=0 && i2+ns<=N2) {  // no wrapping: avoid ptrs
    for (int dy=0; dy<ns; dy++) {
      BIGINT j = N1*(i2+dy) + i1;
      for (int dx=0; dx<ns; dx++) {
	FLT k = ker1[dx]*ker2[dy];
	out[0] += du[2*j] * k;
	out[1] += du[2*j+1] * k;
	++j;
      }
    }
  } else {                         // wraps somewhere: use ptr list (slower)
    BIGINT j1[MAX_NSPREAD], j2[MAX_NSPREAD];   // 1d ptr lists
    BIGINT x=i1, y=i2;                 // initialize coords
    for (int d=0; d<ns; d++) {         // set up ptr lists
      if (x<0) x+=N1;
      if (x>=N1) x-=N1;
      j1[d] = x++;
      if (y<0) y+=N2;
      if (y>=N2) y-=N2;
      j2[d] = y++;
    }
    for (int dy=0; dy<ns; dy++) {      // use the pts lists
      BIGINT oy = N1*j2[dy];           // offset due to y
      for (int dx=0; dx<ns; dx++) {
	FLT k = ker1[dx]*ker2[dy];
	BIGINT j = oy + j1[dx];
	out[0] += du[2*j] * k;
	out[1] += du[2*j+1] * k;
      }
    }
  }
  target[0] = out[0];
  target[1] = out[1];  
}

void interp_cube(FLT *target,FLT *du, FLT *ker1, FLT *ker2, FLT *ker3,
		 BIGINT i1,BIGINT i2,BIGINT i3, BIGINT N1,BIGINT N2,BIGINT N3,int ns)
// 3D interpolate complex values from du (uniform grid data) array to out value,
// using ns*ns*ns cube of real weights
// in ker. out must be size 2 (real,imag), and du
// of size 2*N1*N2*N3 (alternating real,imag). i1 is the left-most index in
// [0,N1), i2 the bottom index in [0,N2), i3 lowest in [0,N3).
// Periodic wrapping in the du array is applied, assuming N1,N2,N3>=ns.
// dx,dy,dz indices into ker array, j index in complex du array.
// Barnett 6/16/17
{
  FLT out[] = {0.0, 0.0};  
  if (i1>=0 && i1+ns<=N1 && i2>=0 && i2+ns<=N2 && i3>=0 && i3+ns<=N3) {
    // no wrapping: avoid ptrs
    for (int dz=0; dz<ns; dz++) {
      BIGINT oz = N1*N2*(i3+dz);        // offset due to z
      for (int dy=0; dy<ns; dy++) {
	BIGINT j = oz + N1*(i2+dy) + i1;
	FLT ker23 = ker2[dy]*ker3[dz];
	for (int dx=0; dx<ns; dx++) {
	  FLT k = ker1[dx]*ker23;
	  out[0] += du[2*j] * k;
	  out[1] += du[2*j+1] * k;
	  ++j;
	}
      }
    }
  } else {                         // wraps somewhere: use ptr list (slower)
    BIGINT j1[MAX_NSPREAD], j2[MAX_NSPREAD], j3[MAX_NSPREAD];   // 1d ptr lists
    BIGINT x=i1, y=i2, z=i3;         // initialize coords
    for (int d=0; d<ns; d++) {          // set up ptr lists
      if (x<0) x+=N1;
      if (x>=N1) x-=N1;
      j1[d] = x++;
      if (y<0) y+=N2;
      if (y>=N2) y-=N2;
      j2[d] = y++;
      if (z<0) z+=N3;
      if (z>=N3) z-=N3;
      j3[d] = z++;
    }
    for (int dz=0; dz<ns; dz++) {             // use the pts lists
      BIGINT oz = N1*N2*j3[dz];               // offset due to z
      for (int dy=0; dy<ns; dy++) {
	BIGINT oy = oz + N1*j2[dy];           // offset due to y & z
	FLT ker23 = ker2[dy]*ker3[dz];	
	for (int dx=0; dx<ns; dx++) {
	  FLT k = ker1[dx]*ker23;
	  BIGINT j = oy + j1[dx];
	  out[0] += du[2*j] * k;
	  out[1] += du[2*j+1] * k;
	}
      }
    }
  }
  target[0] = out[0];
  target[1] = out[1];  
}

void spread_subproblem_1d(BIGINT N1,FLT *du,BIGINT M,
			  FLT *kx,FLT *dd,
			  const spread_opts& opts)
/* spreader from dd (NU) to du (uniform) in 1D without wrapping.
   kx (size M) are NU locations in [0,N1]
   dd (size M complex) are source strengths
   du (size N1) is uniform output array.

   This a naive loop w/ Ludvig's eval_ker_vec.
*/
{
  int ns=opts.nspread;
  FLT ns2 = (FLT)ns/2;          // half spread width
  for (BIGINT i=0;i<2*N1;++i)
    du[i] = 0.0;
  FLT kernel_args[MAX_NSPREAD];
  FLT ker[MAX_NSPREAD];
  for (BIGINT i=0; i<M; i++) {           // loop over NU pts
    FLT re0 = dd[2*i];
    FLT im0 = dd[2*i+1];
    BIGINT i1 = (BIGINT)std::ceil(kx[i] - ns2);
    FLT x1 = (FLT)i1 - kx[i];            // x1 in [-w/2,-w/2+1]
    if (opts.kerevalmeth==0) {
      set_kernel_args(kernel_args, x1, opts);
      evaluate_kernel_vector(ker, kernel_args, opts, ns);
    } else
      eval_kernel_vec_Horner(ker,x1,ns,opts);
    // critical inner loop: 
    BIGINT j=i1;
    for (int dx=0; dx<ns; ++dx) {
      FLT k = ker[dx];
      du[2*j] += re0*k;
      du[2*j+1] += im0*k;
      ++j;
    }
  }
}

void spread_subproblem_2d(BIGINT N1,BIGINT N2,FLT *du,BIGINT M,
			  FLT *kx,FLT *ky,FLT *dd,
			  const spread_opts& opts)
/* spreader from dd (NU) to du (uniform) in 2D without wrapping.
   kx,ky (size M) are NU locations in [0,N1],[0,N2]
   dd (size M complex) are source strengths
   du (size N1*N2) is uniform output array
 */
{
  int ns=opts.nspread;
  FLT ns2 = (FLT)ns/2;          // half spread width
  for (BIGINT i=0;i<2*N1*N2;++i)
    du[i] = 0.0;
  FLT kernel_args[2*MAX_NSPREAD];
  FLT kernel_values[2*MAX_NSPREAD];
  FLT *ker1 = kernel_values;
  FLT *ker2 = kernel_values + ns;  
  for (BIGINT i=0; i<M; i++) {           // loop over NU pts
    FLT re0 = dd[2*i];
    FLT im0 = dd[2*i+1];
    BIGINT i1 = (BIGINT)std::ceil(kx[i] - ns2);
    BIGINT i2 = (BIGINT)std::ceil(ky[i] - ns2);
    FLT x1 = (FLT)i1 - kx[i];
    FLT x2 = (FLT)i2 - ky[i];
    if (opts.kerevalmeth==0) {
      set_kernel_args(kernel_args, x1, opts);
      set_kernel_args(kernel_args+ns, x2, opts);
      evaluate_kernel_vector(kernel_values, kernel_args, opts, 2*ns);
    } else {
      eval_kernel_vec_Horner(ker1,x1,ns,opts);
      eval_kernel_vec_Horner(ker2,x2,ns,opts);
    }
    // Combine kernel with complex source value to simplify inner loop
    FLT ker1val[2*MAX_NSPREAD];
    for (int i = 0; i < ns; i++) {
      ker1val[2*i] = re0*ker1[i];
      ker1val[2*i+1] = im0*ker1[i];	
    }    
    // critical inner loop:
    for (int dy=0; dy<ns; ++dy) {
      BIGINT j = N1*(i2+dy) + i1;
      FLT kerval = ker2[dy];
      FLT *trg = du+2*j;
      for (int dx=0; dx<2*ns; ++dx) {
	trg[dx] += kerval*ker1val[dx];
      }	
    }
  }
}

void spread_subproblem_3d(BIGINT N1,BIGINT N2,BIGINT N3,FLT *du,BIGINT M,
			  FLT *kx,FLT *ky,FLT *kz,FLT *dd,
			  const spread_opts& opts)
/* spreader from dd (NU) to du (uniform) in 3D without wrapping.
   kx,ky,kz (size M) are NU locations in [0,N1],[0,N2],[0,N3]
   dd (size M complex) are source strengths
   du (size N1*N2*N3) is uniform output array
 */
{
  int ns=opts.nspread;
  FLT ns2 = (FLT)ns/2;          // half spread width
  for (BIGINT i=0;i<2*N1*N2*N3;++i)
    du[i] = 0.0;
  FLT kernel_args[3*MAX_NSPREAD];
  // Kernel values stored in consecutive memory. This allows us to compute
  // values in all three directions in a single kernel evaluation call.
  FLT kernel_values[3*MAX_NSPREAD];
  FLT *ker1 = kernel_values;
  FLT *ker2 = kernel_values + ns;
  FLT *ker3 = kernel_values + 2*ns;  
  for (BIGINT i=0; i<M; i++) {           // loop over NU pts
    FLT re0 = dd[2*i];
    FLT im0 = dd[2*i+1];
    BIGINT i1 = (BIGINT)std::ceil(kx[i] - ns2);
    BIGINT i2 = (BIGINT)std::ceil(ky[i] - ns2);
    BIGINT i3 = (BIGINT)std::ceil(kz[i] - ns2);
    FLT x1 = (FLT)i1 - kx[i];
    FLT x2 = (FLT)i2 - ky[i];
    FLT x3 = (FLT)i3 - kz[i];
    if (opts.kerevalmeth==0) {
      set_kernel_args(kernel_args, x1, opts);
      set_kernel_args(kernel_args+ns, x2, opts);
      set_kernel_args(kernel_args+2*ns, x3, opts);
      evaluate_kernel_vector(kernel_values, kernel_args, opts, 3*ns);
    } else {
      eval_kernel_vec_Horner(ker1,x1,ns,opts);
      eval_kernel_vec_Horner(ker2,x2,ns,opts);
      eval_kernel_vec_Horner(ker3,x3,ns,opts);
    }
    // Combine kernel with complex source value to simplify inner loop
    FLT ker1val[2*MAX_NSPREAD];
    for (int i = 0; i < ns; i++) {
      ker1val[2*i] = re0*ker1[i];
      ker1val[2*i+1] = im0*ker1[i];	
    }    
    // critical inner loop:
    for (int dz=0; dz<ns; ++dz) {
      BIGINT oz = N1*N2*(i3+dz);        // offset due to z
      for (int dy=0; dy<ns; ++dy) {
	BIGINT j = oz + N1*(i2+dy) + i1;
	FLT kerval = ker2[dy]*ker3[dz];
	FLT *trg = du+2*j;
	for (int dx=0; dx<2*ns; ++dx) {
	  trg[dx] += kerval*ker1val[dx];
	}	
      }
    }
  }
}

void add_wrapped_subgrid(BIGINT offset1,BIGINT offset2,BIGINT offset3,
			 BIGINT size1,BIGINT size2,BIGINT size3,BIGINT N1,
			 BIGINT N2,BIGINT N3,FLT *data_uniform, FLT *du0)
/* Add a large subgrid (du0) to output grid (data_uniform),
   with periodic wrapping to N1,N2,N3 box.
   offset1,2,3 give the offset of the subgrid from the lowest corner of output.
   size1,2,3 give the size of subgrid.
   Works in all dims. Thread-safe since must be called inside omp critical.
   Barnett 3/27/18 made separate routine, tried to speed up inner loop.
*/
{
  std::vector<BIGINT> o2(size2), o3(size3);
  BIGINT y=offset2, z=offset3;    // fill wrapped ptr lists in slower dims y,z...
  for (int i=0; i<size2; ++i) {
    if (y<0) y+=N2;
    if (y>=N2) y-=N2;
    o2[i] = y++;
  }
  for (int i=0; i<size3; ++i) {
    if (z<0) z+=N3;
    if (z>=N3) z-=N3;
    o3[i] = z++;
  }
  BIGINT nlo = (offset1<0) ? -offset1 : 0;          // # wrapping below in x
  BIGINT nhi = (offset1+size1>N1) ? offset1+size1-N1 : 0;    // " above in x
  // this triple loop works in all dims
  for (int dz=0; dz<size3; dz++) {       // use ptr lists in each axis
    BIGINT oz = N1*N2*o3[dz];            // offset due to z (0 in <3D)
    for (int dy=0; dy<size2; dy++) {
      BIGINT oy = oz + N1*o2[dy];        // off due to y & z (0 in 1D)
      FLT *out = data_uniform + 2*oy;
      FLT *in  = du0 + 2*size1*(dy + size2*dz);   // ptr to subgrid array
      BIGINT o = 2*(offset1+N1);         // 1d offset for output
      for (int j=0; j<2*nlo; j++)        // j is really dx/2 (since re,im parts)
	out[j+o] += in[j];
      o = 2*offset1;
      for (int j=2*nlo; j<2*(size1-nhi); j++)
	out[j+o] += in[j];
      o = 2*(offset1-N1);
      for (int j=2*(size1-nhi); j<2*size1; j++)
      	out[j+o] += in[j];
    }
  }
}

void bin_sort_singlethread(BIGINT *ret, BIGINT M, FLT *kx, FLT *ky, FLT *kz,
	      BIGINT N1,BIGINT N2,BIGINT N3,int pirange,
	      double bin_size_x,double bin_size_y,double bin_size_z, int debug)
/* Returns permutation of all nonuniform points with good RAM access,
 * ie less cache misses for spreading, in 1D, 2D, or 3D. Singe-threaded version.
 *
 * This is achieved by binning into cuboids (of given bin_size)
 * then reading out the indices within
 * these boxes in the natural box order (x fastest, y med, z slowest).
 * Finally the permutation is inverted.
 * 
 * Inputs: M - number of input NU points.
 *         kx,ky,kz - length-M arrays of real coords of NU pts, in the domain
 *                    for FOLDRESCALE, which includes [0,N1], [0,N2], [0,N3]
 *                    respectively, if pirange=0; or [-pi,pi] if pirange=1.
 *         N1,N2,N3 - ranges of NU coords (set N2=N3=1 for 1D, N3=1 for 2D)
 *         bin_size_x,y,z - what binning box size to use in each dimension
 *                    (in rescaled coords where ranges are [0,Ni] ).
 *                    For 1D, only bin_size_x is used; for 2D, it and bin_size_y
 * Output:
 *         writes to ret a vector list of indices, each in the range 0,..,M-1.
 *         Thus, ret must have been allocated for M BIGINTs.
 *
 * Notes: I compared RAM usage against declaring an internal vector and passing
 * back; the latter used more RAM and was slower.
 * Avoided the bins array, as in JFM's spreader of 2016,
 * tidied up, early 2017, Barnett.
 *
 * Timings: 3s for M=1e8 NU pts on 1 core of i7; 5s on 1 core of xeon.
 */
{
  bool isky=(N2>1), iskz=(N3>1);  // ky,kz avail? (cannot access if not)
  BIGINT nbins1=N1/bin_size_x+1, nbins2, nbins3;
  nbins2 = isky ? N2/bin_size_y+1 : 1;
  nbins3 = iskz ? N3/bin_size_z+1 : 1;
  BIGINT nbins = nbins1*nbins2*nbins3;

  std::vector<BIGINT> counts(nbins,0);  // count how many pts in each bin
  for (BIGINT i=0; i<M; i++) {
    // find the bin index in however many dims are needed
    BIGINT i1=FOLDRESCALE(kx[i],N1,pirange)/bin_size_x, i2=0, i3=0;
    if (isky) i2 = FOLDRESCALE(ky[i],N2,pirange)/bin_size_y;
    if (iskz) i3 = FOLDRESCALE(kz[i],N3,pirange)/bin_size_z;
    BIGINT bin = i1+nbins1*(i2+nbins2*i3);
    counts[bin]++;
  }
  std::vector<BIGINT> offsets(nbins);   // cumulative sum of bin counts
  offsets[0]=0;     // do: offsets = [0 cumsum(counts(1:end-1)]
  for (BIGINT i=1; i<nbins; i++)
    offsets[i]=offsets[i-1]+counts[i-1];
  
  std::vector<BIGINT> inv(M);           // fill inverse map
  for (BIGINT i=0; i<M; i++) {
    // find the bin index (again! but better than using RAM)
    BIGINT i1=FOLDRESCALE(kx[i],N1,pirange)/bin_size_x, i2=0, i3=0;
    if (isky) i2 = FOLDRESCALE(ky[i],N2,pirange)/bin_size_y;
    if (iskz) i3 = FOLDRESCALE(kz[i],N3,pirange)/bin_size_z;
    BIGINT bin = i1+nbins1*(i2+nbins2*i3);
    BIGINT offset=offsets[bin];
    offsets[bin]++;
    inv[i]=offset;
  }
  // invert the map, writing to output pointer (writing pattern is random)
  for (BIGINT i=0; i<M; i++)
    ret[inv[i]]=i;
}

void bin_sort_multithread(BIGINT *ret, BIGINT M, FLT *kx, FLT *ky, FLT *kz,
	      BIGINT N1,BIGINT N2,BIGINT N3,int pirange,
              double bin_size_x,double bin_size_y,double bin_size_z, int debug,
              int nthr)
/* Mostly-OpenMP'ed version of bin_sort.
   For documentation see: bin_sort_singlethread.
   Caution: when M (# NU pts) << N (# U pts), is SLOWER than single-thread.
   Barnett 2/8/18
   Todo: if debug, print timing breakdowns.
   Explicit #threads control 7/20/20.
 */
{
  bool isky=(N2>1), iskz=(N3>1);  // ky,kz avail? (cannot access if not)
  BIGINT nbins1=N1/bin_size_x+1, nbins2, nbins3;
  nbins2 = isky ? N2/bin_size_y+1 : 1;
  nbins3 = iskz ? N3/bin_size_z+1 : 1;
  BIGINT nbins = nbins1*nbins2*nbins3;
  if (nthr==0)
    fprintf(stderr,"[%s] nthr (%d) must be positive!\n",__func__,nthr);
  int nt = min(M,(BIGINT)nthr);  // handle case of more points than threads
  std::vector<BIGINT> brk(nt+1); // start NU pt indices per thread

  // distribute the M NU pts to threads once & for all...
  for (int t=0; t<=nt; ++t)
    brk[t] = (BIGINT)(0.5 + M*t/(double)nt);   // start index for t'th chunk
  std::vector<BIGINT> counts(nbins,0);  // counts of how many pts in each bin
  std::vector< std::vector<BIGINT> > ot(nt,counts); // offsets per thread, nt * nbins
  {    // scope for ct, the 2d array of counts in bins for each threads's NU pts
    std::vector< std::vector<BIGINT> > ct(nt,counts);   // nt * nbins, init to 0
    
#pragma omp parallel num_threads(nt)
    {                                    // block done once per thread;
      int t = MY_OMP_GET_THREAD_NUM();   // we assume all nt threads created
      //printf("\tt=%d: [%d,%d]\n",t,jlo[t],jhi[t]);
      for (BIGINT i=brk[t]; i<brk[t+1]; i++) {
        // find the bin index in however many dims are needed
        BIGINT i1=FOLDRESCALE(kx[i],N1,pirange)/bin_size_x, i2=0, i3=0;
        if (isky) i2 = FOLDRESCALE(ky[i],N2,pirange)/bin_size_y;
        if (iskz) i3 = FOLDRESCALE(kz[i],N3,pirange)/bin_size_z;
        BIGINT bin = i1+nbins1*(i2+nbins2*i3);
        ct[t][bin]++;               // no clash btw threads
      }
    }
#pragma omp parallel for num_threads(nt) schedule(dynamic,10000) // matters
    for (int t=0; t<nt; ++t)
      for (BIGINT b=0; b<nbins; ++b)
	counts[b] += ct[t][b];
    
    std::vector<BIGINT> offsets(nbins);   // cumulative sum of bin counts
    offsets[0]=0;
    // do: offsets = [0 cumsum(counts(1:end-1))].
    // multithread? (do chunks in 2 pass) but not many bins; don't bother...
    for (BIGINT i=1; i<nbins; i++)
      offsets[i]=offsets[i-1]+counts[i-1];
    
    for (BIGINT b=0; b<nbins; ++b)  // now build offsets for each thread & bin:
      ot[0][b] = offsets[b];                       // init
#pragma omp parallel for num_threads(nt) schedule(dynamic,10000)
    for (int t=1; t<nt; ++t)
      for (BIGINT b=0; b<nbins; ++b)
	ot[t][b] = ot[t-1][b]+ct[t-1][b];        // cumsum along t axis
    
  } // scope frees up ct here
  
  std::vector<BIGINT> inv(M);           // fill inverse map
#pragma omp parallel num_threads(nt)
  {
    int t = MY_OMP_GET_THREAD_NUM();
    for (BIGINT i=brk[t]; i<brk[t+1]; i++) {
      // find the bin index (again! but better than using RAM)
      BIGINT i1=FOLDRESCALE(kx[i],N1,pirange)/bin_size_x, i2=0, i3=0;
      if (isky) i2 = FOLDRESCALE(ky[i],N2,pirange)/bin_size_y;
      if (iskz) i3 = FOLDRESCALE(kz[i],N3,pirange)/bin_size_z;
      BIGINT bin = i1+nbins1*(i2+nbins2*i3);
      inv[i]=ot[t][bin];   // get the offset for this NU pt and thread
      ot[t][bin]++;               // no clash
    }
  }
  // invert the map, writing to output pointer (writing pattern is random)
#pragma omp parallel for num_threads(nt) schedule(dynamic,10000)
  for (BIGINT i=0; i<M; i++)
    ret[inv[i]]=i;
}


void get_subgrid(BIGINT &offset1,BIGINT &offset2,BIGINT &offset3,BIGINT &size1,BIGINT &size2,BIGINT &size3,BIGINT M,FLT* kx,FLT* ky,FLT* kz,int ns,int ndims)
/* Writes out the offsets and sizes of the subgrid defined by the
   nonuniform points and the spreading diameter approx ns/2.
   Requires O(M) effort to find the k array bnds. Works in all dims 1,2,3.
   Must return offset 0 and size 1 for each unused dimension.
   Grid has been made tight to the kernel point choice using identical ceil
   operations.  6/16/17
*/
{
  FLT ns2 = (FLT)ns/2;
  // compute the min/max of the k-space locations of the nonuniform points
  FLT min_kx,max_kx;
  arrayrange(M,kx,&min_kx,&max_kx);
  BIGINT a1=std::ceil(min_kx-ns2);
  BIGINT a2=std::ceil(max_kx-ns2)+ns-1;
  offset1=a1;
  size1=a2-a1+1;
  if (ndims>1) {
    FLT min_ky,max_ky;
    arrayrange(M,ky,&min_ky,&max_ky);
    BIGINT b1=std::ceil(min_ky-ns2);
    BIGINT b2=std::ceil(max_ky-ns2)+ns-1;
    offset2=b1;
    size2=b2-b1+1;
  } else {
    offset2=0;
    size2=1;
  }
  if (ndims>2) {
    FLT min_kz,max_kz;
    arrayrange(M,kz,&min_kz,&max_kz);
    BIGINT c1=std::ceil(min_kz-ns2);
    BIGINT c2=std::ceil(max_kz-ns2)+ns-1;
    offset3=c1;
    size3=c2-c1+1;
  } else {
    offset3=0;
    size3=1;
  }
}
