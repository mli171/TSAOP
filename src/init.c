#include <R.h>
#include <Rinternals.h>
#include <R_ext/Rdynload.h>

/* ---- CLSEW prototypes (match real signatures) ---- */
void read_dimensions(int *K, int *p, int *Ts);
void allocate(void);
void deallocate(void);
void read_data(int *Xl);
void read_data_matrix(int *Xw);
void read_covariates(double *DesignX);
void CLSEW_aop_Rcall(double *par, double *out);
void gradient_CLSEW_aop(double *par, double *grad);
void score_by_t_CLSEW_aop(double *par, double *Wout);

/* ---- PL-AR(p) prototypes (match aop_PL.c) ---- */
void read_dimensions_pl_arp(int *K, int *p, int *Ts);
void read_pl_ar_order_pl_arp(int *q);
void read_order_pair_lik_pl_arp(int *L);
void allocate_pl_arp(void);
void deallocate_pl_arp(void);
void read_data_pl_arp(int *Xl);
void read_covariates_pl_arp(double *DesignX);
void logPair_aop_ARp_Rcall_pl_arp(double *par, double *out);
void gradient_logPair_aop_ARp_pl_arp(double *par, double *grad);
void score_by_t_logPair_aop_ARp_pl_arp(double *par, double *scores);

/* ---- innovations (.Call) prototypes (match innovations_uni.c) ---- */
SEXP innovations_uni_gamma(SEXP gammaSEXP, SEXP uSEXP,
                           SEXP jitterSEXP, SEXP lag_maxSEXP, SEXP lag_tolSEXP, SEXP aheadSEXP);
SEXP innovations_uni_K(SEXP KSEXP, SEXP uSEXP,
                       SEXP jitterSEXP, SEXP lag_maxSEXP, SEXP lag_tolSEXP, SEXP aheadSEXP);
SEXP innovations_uni_kappa(SEXP kappaFunSEXP, SEXP uSEXP,
                           SEXP jitterSEXP, SEXP lag_maxSEXP, SEXP lag_tolSEXP, SEXP aheadSEXP);

/* ---- innovations (.Call) prototypes (match innovations_mv.c) ---- */
SEXP innovations_mv_gamma(SEXP GammaSEXP, SEXP uSEXP,
                          SEXP jitterSEXP, SEXP symmetrizeSEXP,
                          SEXP lag_maxSEXP, SEXP lag_tolSEXP, SEXP aheadSEXP);
SEXP innovations_mv_Kbig(SEXP KbigSEXP, SEXP uSEXP,
                         SEXP jitterSEXP, SEXP symmetrizeSEXP,
                         SEXP lag_maxSEXP, SEXP lag_tolSEXP, SEXP aheadSEXP);
SEXP innovations_mv_kappa(SEXP kappaFunSEXP, SEXP uSEXP,
                          SEXP jitterSEXP, SEXP symmetrizeSEXP,
                          SEXP lag_maxSEXP, SEXP lag_tolSEXP, SEXP aheadSEXP);

static const R_CMethodDef CEntries[] = {
  /* CLSEW-AR(1) */
  {"read_dimensions",      (DL_FUNC) &read_dimensions,    3},
  {"allocate",             (DL_FUNC) &allocate,           0},
  {"deallocate",           (DL_FUNC) &deallocate,         0},
  {"read_data",            (DL_FUNC) &read_data,          1},
  {"read_data_matrix",     (DL_FUNC) &read_data_matrix,   1},
  {"read_covariates",      (DL_FUNC) &read_covariates,    1},
  {"CLSEW_aop_Rcall",      (DL_FUNC) &CLSEW_aop_Rcall,    2},
  {"gradient_CLSEW_aop",   (DL_FUNC) &gradient_CLSEW_aop, 2},
  {"score_by_t_CLSEW_aop", (DL_FUNC) &score_by_t_CLSEW_aop, 2},

  /* PL-AR(p) */
  {"read_dimensions_pl_arp",          (DL_FUNC) &read_dimensions_pl_arp,          3},
  {"read_pl_ar_order_pl_arp",         (DL_FUNC) &read_pl_ar_order_pl_arp,         1},
  {"read_order_pair_lik_pl_arp",      (DL_FUNC) &read_order_pair_lik_pl_arp,      1},
  {"allocate_pl_arp",                 (DL_FUNC) &allocate_pl_arp,                 0},
  {"deallocate_pl_arp",               (DL_FUNC) &deallocate_pl_arp,               0},
  {"read_data_pl_arp",                (DL_FUNC) &read_data_pl_arp,                1},
  {"read_covariates_pl_arp",          (DL_FUNC) &read_covariates_pl_arp,          1},
  {"logPair_aop_ARp_Rcall_pl_arp",    (DL_FUNC) &logPair_aop_ARp_Rcall_pl_arp,    2},
  {"gradient_logPair_aop_ARp_pl_arp", (DL_FUNC) &gradient_logPair_aop_ARp_pl_arp, 2},
  {"score_by_t_logPair_aop_ARp_pl_arp",(DL_FUNC) &score_by_t_logPair_aop_ARp_pl_arp, 2},

  {NULL, NULL, 0}
};

static const R_CallMethodDef CallEntries[] = {
  {"innovations_uni_gamma", (DL_FUNC) &innovations_uni_gamma, 6},
  {"innovations_uni_K",     (DL_FUNC) &innovations_uni_K,     6},
  {"innovations_uni_kappa", (DL_FUNC) &innovations_uni_kappa, 6},

  {"innovations_mv_gamma", (DL_FUNC) &innovations_mv_gamma, 7},
  {"innovations_mv_Kbig",  (DL_FUNC) &innovations_mv_Kbig,  7},
  {"innovations_mv_kappa", (DL_FUNC) &innovations_mv_kappa, 7},

  {NULL, NULL, 0}
};

void R_init_TSAOP(DllInfo *dll) {
  R_registerRoutines(dll, CEntries, CallEntries, NULL, NULL);
  R_useDynamicSymbols(dll, FALSE);
}
