// src/init_CLSW.c
#include <R.h>
#include <Rinternals.h>
#include <R_ext/Rdynload.h>

/* ---- prototypes that MATCH your .C() calls in R ---- */
void read_dimensions(int *K, int *p, int *Ts);
void allocate(void);
void deallocate(void);
void read_data(int *Xl);
void read_data_matrix(int *Xw);
void read_covariates(double *DesignX);
void CLSEW_aop_Rcall(double *par, double *out);
void gradient_CLSEW_aop(double *par, double *grad);

/* ---- register .C routines ---- */
static const R_CMethodDef CEntries[] = {
  {"read_dimensions",    (DL_FUNC) &read_dimensions,    3},
  {"allocate",           (DL_FUNC) &allocate,           0},
  {"deallocate",         (DL_FUNC) &deallocate,         0},
  {"read_data",          (DL_FUNC) &read_data,          1},
  {"read_data_matrix",   (DL_FUNC) &read_data_matrix,   1},
  {"read_covariates",    (DL_FUNC) &read_covariates,    1},
  {"CLSEW_aop_Rcall",    (DL_FUNC) &CLSEW_aop_Rcall,    5},
  {"gradient_CLSEW_aop", (DL_FUNC) &gradient_CLSEW_aop, 2},
  {NULL, NULL, 0}
};

/* ---- PACKAGE-NAME-SPECIFIC init hook ---- */
void R_init_TSAOP(DllInfo *dll) {
  R_registerRoutines(dll, CEntries, NULL, NULL, NULL);
  R_useDynamicSymbols(dll, FALSE);
}
