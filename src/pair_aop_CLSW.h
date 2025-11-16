#ifndef PAIR_AOP_CLSWH
#define PAIR_AOP_CLSWH

/* Keep C++ compilers happy */
#ifdef __cplusplus
extern "C" {
#endif

  /* ---- Setup / teardown ---- */

  /* Allocate all internal buffers based on dimensions previously read. */
  void allocate(void);

  /* Free all internal buffers (safe to call multiple times). */
  void deallocate(void);

  /* Set problem dimensions.
   num_categ = K (number of ordinal categories)
   num_regr  = p (number of regressors, incl. intercept)
   num_time  = T (length of series) */
  void read_dimensions(const int *num_categ,
                       const int *num_regr,
                       const int *num_time);

  /* Wire in the observed category sequence (length T, 1..K integers). */
  void read_data(const int *data_input);

  /* Wire in the wide indicator matrix Xw (T x K), column-major as integers 0/1. */
  void read_data_matrix(const int *data_matrix_input);

  /* Wire in the design matrix (T x p), column-major doubles. */
  void read_covariates(const double *design_matrix_input);


  /* ---- Objective (CLS on probabilities) ---- */

  /* Compute CLS objective matching AopCLSW (lag=1).
   param layout: [tau_2, ..., tau_{K-1}, beta_1..beta_p, phi]
   Returns scalar objective in *CLSEW. */
  void CLSEW_aop_Rcall(const double *param, double *CLSEW);


  /* ---- Gradient of the CLS objective ---- */

  /* Compute gradient d/dparam of the CLS objective.
   Output length is (K-2) + p + 1 in the same order as 'param'. */
  void gradient_CLSEW_aop(const double *param, double *gradient);


  /* ---- Fortran bivariate normal rectangle (provided by mvndstpack.f) ---- */
  /* Declared here in case other compilation units need it. */
  double F77_NAME(bvnmvn)(double *lower, double *upper, int *infin, double *correl);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* PAIR_AOP_CLSWH */
