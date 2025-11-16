// src/init_CLSW.c
#include <R.h>
#include <Rinternals.h>
#include <R_ext/Rdynload.h>

/* --- prototypes matching your C definitions (all .C args are pointers) --- */
void read_dimensions(int *K, int *p, int *Ts);
void allocate(void);
void read_data(int *Xl);
void read_data_matrix(int *Xw);
void read_covariates(double *DesignX);
void deallocate(void);
void CLSEW_aop_Rcall(double *par, double *out);
void gradient_CLSEW_aop(double *par, double *grad);

/* --- .C table (name, pointer, number_of_arguments) --- */
static const R_CMethodDef CEntries[] = {
  {"read_dimensions",      (DL_FUNC) &read_dimensions,      3},
  {"allocate",             (DL_FUNC) &allocate,             0},
  {"read_data",            (DL_FUNC) &read_data,            1},
  {"read_data_matrix",     (DL_FUNC) &read_data_matrix,     1},
  {"read_covariates",      (DL_FUNC) &read_covariates,      1},
  {"deallocate",           (DL_FUNC) &deallocate,           0},
  {"CLSEW_aop_Rcall",      (DL_FUNC) &CLSEW_aop_Rcall,      2},
  {"gradient_CLSEW_aop",   (DL_FUNC) &gradient_CLSEW_aop,   2},
  {NULL, NULL, 0}
};

void R_init_aopts(DllInfo *dll) {
  R_registerRoutines(dll, CEntries, NULL, NULL, NULL);
  R_useDynamicSymbols(dll, FALSE);
}
