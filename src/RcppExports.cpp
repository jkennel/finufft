// Generated by using Rcpp::compileAttributes() -> do not edit by hand
// Generator token: 10BE3573-1514-4C36-9D1C-5A225CD40393

#include <RcppArmadillo.h>
#include <Rcpp.h>

using namespace Rcpp;

// nufft_1d1
arma::cx_vec nufft_1d1(arma::vec xj, arma::cx_vec cj, int n1, double tol, int iflag);
RcppExport SEXP _finufft_nufft_1d1(SEXP xjSEXP, SEXP cjSEXP, SEXP n1SEXP, SEXP tolSEXP, SEXP iflagSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::vec >::type xj(xjSEXP);
    Rcpp::traits::input_parameter< arma::cx_vec >::type cj(cjSEXP);
    Rcpp::traits::input_parameter< int >::type n1(n1SEXP);
    Rcpp::traits::input_parameter< double >::type tol(tolSEXP);
    Rcpp::traits::input_parameter< int >::type iflag(iflagSEXP);
    rcpp_result_gen = Rcpp::wrap(nufft_1d1(xj, cj, n1, tol, iflag));
    return rcpp_result_gen;
END_RCPP
}
// nufft_2d1
arma::cx_mat nufft_2d1(arma::vec xj, arma::vec yj, arma::cx_vec cj, int n1, int n2, double tol, int iflag);
RcppExport SEXP _finufft_nufft_2d1(SEXP xjSEXP, SEXP yjSEXP, SEXP cjSEXP, SEXP n1SEXP, SEXP n2SEXP, SEXP tolSEXP, SEXP iflagSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::vec >::type xj(xjSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type yj(yjSEXP);
    Rcpp::traits::input_parameter< arma::cx_vec >::type cj(cjSEXP);
    Rcpp::traits::input_parameter< int >::type n1(n1SEXP);
    Rcpp::traits::input_parameter< int >::type n2(n2SEXP);
    Rcpp::traits::input_parameter< double >::type tol(tolSEXP);
    Rcpp::traits::input_parameter< int >::type iflag(iflagSEXP);
    rcpp_result_gen = Rcpp::wrap(nufft_2d1(xj, yj, cj, n1, n2, tol, iflag));
    return rcpp_result_gen;
END_RCPP
}
// nufft_3d1
arma::cx_cube nufft_3d1(arma::vec xj, arma::vec yj, arma::vec zj, arma::cx_vec cj, int n1, int n2, int n3, double tol, int iflag);
RcppExport SEXP _finufft_nufft_3d1(SEXP xjSEXP, SEXP yjSEXP, SEXP zjSEXP, SEXP cjSEXP, SEXP n1SEXP, SEXP n2SEXP, SEXP n3SEXP, SEXP tolSEXP, SEXP iflagSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::vec >::type xj(xjSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type yj(yjSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type zj(zjSEXP);
    Rcpp::traits::input_parameter< arma::cx_vec >::type cj(cjSEXP);
    Rcpp::traits::input_parameter< int >::type n1(n1SEXP);
    Rcpp::traits::input_parameter< int >::type n2(n2SEXP);
    Rcpp::traits::input_parameter< int >::type n3(n3SEXP);
    Rcpp::traits::input_parameter< double >::type tol(tolSEXP);
    Rcpp::traits::input_parameter< int >::type iflag(iflagSEXP);
    rcpp_result_gen = Rcpp::wrap(nufft_3d1(xj, yj, zj, cj, n1, n2, n3, tol, iflag));
    return rcpp_result_gen;
END_RCPP
}
// nufft_1d2
arma::cx_vec nufft_1d2(arma::vec xj, arma::cx_vec fk, double tol, int iflag);
RcppExport SEXP _finufft_nufft_1d2(SEXP xjSEXP, SEXP fkSEXP, SEXP tolSEXP, SEXP iflagSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::vec >::type xj(xjSEXP);
    Rcpp::traits::input_parameter< arma::cx_vec >::type fk(fkSEXP);
    Rcpp::traits::input_parameter< double >::type tol(tolSEXP);
    Rcpp::traits::input_parameter< int >::type iflag(iflagSEXP);
    rcpp_result_gen = Rcpp::wrap(nufft_1d2(xj, fk, tol, iflag));
    return rcpp_result_gen;
END_RCPP
}
// nufft_2d2
arma::cx_vec nufft_2d2(arma::vec xj, arma::vec yj, arma::cx_mat fk, double tol, int iflag);
RcppExport SEXP _finufft_nufft_2d2(SEXP xjSEXP, SEXP yjSEXP, SEXP fkSEXP, SEXP tolSEXP, SEXP iflagSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::vec >::type xj(xjSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type yj(yjSEXP);
    Rcpp::traits::input_parameter< arma::cx_mat >::type fk(fkSEXP);
    Rcpp::traits::input_parameter< double >::type tol(tolSEXP);
    Rcpp::traits::input_parameter< int >::type iflag(iflagSEXP);
    rcpp_result_gen = Rcpp::wrap(nufft_2d2(xj, yj, fk, tol, iflag));
    return rcpp_result_gen;
END_RCPP
}
// nufft_3d2
arma::cx_vec nufft_3d2(arma::vec xj, arma::vec yj, arma::vec zj, arma::cx_cube fk, double tol, int iflag);
RcppExport SEXP _finufft_nufft_3d2(SEXP xjSEXP, SEXP yjSEXP, SEXP zjSEXP, SEXP fkSEXP, SEXP tolSEXP, SEXP iflagSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::vec >::type xj(xjSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type yj(yjSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type zj(zjSEXP);
    Rcpp::traits::input_parameter< arma::cx_cube >::type fk(fkSEXP);
    Rcpp::traits::input_parameter< double >::type tol(tolSEXP);
    Rcpp::traits::input_parameter< int >::type iflag(iflagSEXP);
    rcpp_result_gen = Rcpp::wrap(nufft_3d2(xj, yj, zj, fk, tol, iflag));
    return rcpp_result_gen;
END_RCPP
}
// nufft_1d3
arma::cx_vec nufft_1d3(arma::vec xj, arma::cx_vec cj, arma::vec sk, double tol, int iflag);
RcppExport SEXP _finufft_nufft_1d3(SEXP xjSEXP, SEXP cjSEXP, SEXP skSEXP, SEXP tolSEXP, SEXP iflagSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::vec >::type xj(xjSEXP);
    Rcpp::traits::input_parameter< arma::cx_vec >::type cj(cjSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type sk(skSEXP);
    Rcpp::traits::input_parameter< double >::type tol(tolSEXP);
    Rcpp::traits::input_parameter< int >::type iflag(iflagSEXP);
    rcpp_result_gen = Rcpp::wrap(nufft_1d3(xj, cj, sk, tol, iflag));
    return rcpp_result_gen;
END_RCPP
}
// nufft_2d3
arma::cx_vec nufft_2d3(arma::vec xj, arma::vec yj, arma::cx_vec cj, arma::vec sk, arma::vec tk, double tol, int iflag);
RcppExport SEXP _finufft_nufft_2d3(SEXP xjSEXP, SEXP yjSEXP, SEXP cjSEXP, SEXP skSEXP, SEXP tkSEXP, SEXP tolSEXP, SEXP iflagSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::vec >::type xj(xjSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type yj(yjSEXP);
    Rcpp::traits::input_parameter< arma::cx_vec >::type cj(cjSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type sk(skSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type tk(tkSEXP);
    Rcpp::traits::input_parameter< double >::type tol(tolSEXP);
    Rcpp::traits::input_parameter< int >::type iflag(iflagSEXP);
    rcpp_result_gen = Rcpp::wrap(nufft_2d3(xj, yj, cj, sk, tk, tol, iflag));
    return rcpp_result_gen;
END_RCPP
}
// nufft_3d3
arma::cx_vec nufft_3d3(arma::vec xj, arma::vec yj, arma::vec zj, arma::cx_vec cj, arma::vec sk, arma::vec tk, arma::vec uk, double tol, int iflag);
RcppExport SEXP _finufft_nufft_3d3(SEXP xjSEXP, SEXP yjSEXP, SEXP zjSEXP, SEXP cjSEXP, SEXP skSEXP, SEXP tkSEXP, SEXP ukSEXP, SEXP tolSEXP, SEXP iflagSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::vec >::type xj(xjSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type yj(yjSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type zj(zjSEXP);
    Rcpp::traits::input_parameter< arma::cx_vec >::type cj(cjSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type sk(skSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type tk(tkSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type uk(ukSEXP);
    Rcpp::traits::input_parameter< double >::type tol(tolSEXP);
    Rcpp::traits::input_parameter< int >::type iflag(iflagSEXP);
    rcpp_result_gen = Rcpp::wrap(nufft_3d3(xj, yj, zj, cj, sk, tk, uk, tol, iflag));
    return rcpp_result_gen;
END_RCPP
}
// fftshift
arma::cx_vec fftshift(arma::cx_vec x);
RcppExport SEXP _finufft_fftshift(SEXP xSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::cx_vec >::type x(xSEXP);
    rcpp_result_gen = Rcpp::wrap(fftshift(x));
    return rcpp_result_gen;
END_RCPP
}
// ifftshift
arma::vec ifftshift(arma::vec x);
RcppExport SEXP _finufft_ifftshift(SEXP xSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::vec >::type x(xSEXP);
    rcpp_result_gen = Rcpp::wrap(ifftshift(x));
    return rcpp_result_gen;
END_RCPP
}

static const R_CallMethodDef CallEntries[] = {
    {"_finufft_nufft_1d1", (DL_FUNC) &_finufft_nufft_1d1, 5},
    {"_finufft_nufft_2d1", (DL_FUNC) &_finufft_nufft_2d1, 7},
    {"_finufft_nufft_3d1", (DL_FUNC) &_finufft_nufft_3d1, 9},
    {"_finufft_nufft_1d2", (DL_FUNC) &_finufft_nufft_1d2, 4},
    {"_finufft_nufft_2d2", (DL_FUNC) &_finufft_nufft_2d2, 5},
    {"_finufft_nufft_3d2", (DL_FUNC) &_finufft_nufft_3d2, 6},
    {"_finufft_nufft_1d3", (DL_FUNC) &_finufft_nufft_1d3, 5},
    {"_finufft_nufft_2d3", (DL_FUNC) &_finufft_nufft_2d3, 7},
    {"_finufft_nufft_3d3", (DL_FUNC) &_finufft_nufft_3d3, 9},
    {"_finufft_fftshift", (DL_FUNC) &_finufft_fftshift, 1},
    {"_finufft_ifftshift", (DL_FUNC) &_finufft_ifftshift, 1},
    {NULL, NULL, 0}
};

RcppExport void R_init_finufft(DllInfo *dll) {
    R_registerRoutines(dll, NULL, CallEntries, NULL, NULL);
    R_useDynamicSymbols(dll, FALSE);
}
