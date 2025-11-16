#include <R.h>
#include <Rmath.h>
#include <R_ext/RS.h>
#include <R_ext/Applic.h>
#include <R_ext/Memory.h>
#include "pair_aop_CLSW.h"

#define MISSING_VALUE -999
#define IS_MISSING(data) ((data) == MISSING_VALUE ? 1 : 0)
#define SMALL 1e-16
#define BIG  1e+16

/********************************************************************************
 STATIC VARIABLES
 ********************************************************************************/

static int is_allocated = 0;

/* indicator for the adopted model (unused here but kept for parity) */
static int model;

/* variables containing data and design matrix */
static int    *data           = NULL;
static double **data_matrix   = NULL;
static double **design_matrix = NULL;

static double One_Rho2;
static double sqrt_One_Rho2;
static double Inv_TwoPiOne_Rho2;

/* forward decls (objective) */
static double CLSEW_aop(const double *param);
static double CLSEW_aop_single(const double *param);
static void   trans_prob_aop(const int position_t, const int k, double *result);

/* forward decls (gradient) */
void         gradient_CLSEW_aop(const double *param, double *gradient);
static void  gradient_CLSEW_aop_single(const double *param);
void         gradient_CLSEW_pair_aop(const int position1, const int position2, double *gradient_this_pair);

/* bivariate standard normal pdf (no integration) — forward decl */
static double mydnorm2(const double x1, const double x2, const double rho);

/* parameters */
struct Parameters { double *tau; double *beta; double phi; };
static struct Parameters Param = { NULL, NULL, 0.0 };

/* dimensions */
struct Dimensions {
  int num_param;   /* (K-2) + p + 1 */
int num_categ;   /* K */
int num_regr;    /* p */
int num_time;    /* T */
};
static struct Dimensions Dim = {0,0,0,0};

/* state */
static int current_index;
static int current_index_g;
static int end_index;

static double *means = NULL;

/* gradient use */
static double *gradient_single    = NULL;
static double *gradient_this_pair = NULL;

static double *density_Ct_Mut   = NULL;
static double *density_Ctm1_Mut = NULL;
static double *Ct_Mut   = NULL;
static double *Ctm1_Mut = NULL;

/* Fortran BVNMVN is a FUNCTION returning double */
extern double F77_NAME(bvnmvn)(double *lower, double *upper, int *infin, double *correl);

/* --------------------------------- memory ---------------------------------- */

void allocate()
{
  if (is_allocated) return;
  
  int i,j;
  data = R_Calloc(Dim.num_time, int);
  data_matrix   = R_Calloc(Dim.num_time, double*);
  design_matrix = R_Calloc(Dim.num_time, double*);
  
  Param.tau  = R_Calloc((Dim.num_categ - 1), double);
  Param.beta = R_Calloc(Dim.num_regr,       double);
  
  means = R_Calloc((Dim.num_time + 1), double);
  
  gradient_single    = R_Calloc(Dim.num_param, double);
  gradient_this_pair = R_Calloc(Dim.num_param, double);
  
  for (i = 0; i < Dim.num_time; i++) {
    design_matrix[i] = R_Calloc(Dim.num_regr, double);
    for (j = 0; j < Dim.num_regr; j++) design_matrix[i][j] = 0.0;
    
    data_matrix[i] = R_Calloc(Dim.num_categ, double);
    for (j = 0; j < Dim.num_categ; j++) data_matrix[i][j] = 0.0;
  }
  
  Ct_Mut           = R_Calloc(Dim.num_time, double);
  Ctm1_Mut         = R_Calloc(Dim.num_time, double);
  density_Ct_Mut   = R_Calloc(Dim.num_time, double);
  density_Ctm1_Mut = R_Calloc(Dim.num_time, double);
  
  is_allocated = 1;
}

void deallocate()
{
  int i;
  if (!is_allocated) return;
  
  if (data)        { R_Free(data);        data = NULL; }
  if (Param.beta)  { R_Free(Param.beta);  Param.beta = NULL; }
  if (Param.tau)   { R_Free(Param.tau);   Param.tau  = NULL; }
  
  if (design_matrix) {
    for (i = 0; i < Dim.num_time; i++)
      if (design_matrix[i]) R_Free(design_matrix[i]);
      R_Free(design_matrix); design_matrix = NULL;
  }
  
  if (data_matrix) {
    for (i = 0; i < Dim.num_time; i++)
      if (data_matrix[i]) R_Free(data_matrix[i]);
      R_Free(data_matrix); data_matrix = NULL;
  }
  
  if (means)            { R_Free(means);            means = NULL; }
  if (Ct_Mut)           { R_Free(Ct_Mut);           Ct_Mut = NULL; }
  if (Ctm1_Mut)         { R_Free(Ctm1_Mut);         Ctm1_Mut = NULL; }
  if (density_Ct_Mut)   { R_Free(density_Ct_Mut);   density_Ct_Mut = NULL; }
  if (density_Ctm1_Mut) { R_Free(density_Ctm1_Mut); density_Ctm1_Mut = NULL; }
  if (gradient_single)  { R_Free(gradient_single);  gradient_single = NULL; }
  if (gradient_this_pair){R_Free(gradient_this_pair);gradient_this_pair = NULL; }
  
  is_allocated = 0;
}

/* ------------------------------ readers ------------------------------------ */

void read_dimensions(const int *num_categ,
                     const int *num_regr,
                     const int *num_time)
{
  Dim.num_param = (*num_categ - 2) + *num_regr + 1;
  Dim.num_categ = *num_categ;
  Dim.num_regr  = *num_regr;
  Dim.num_time  = *num_time;
}

void read_data(const int *data_input)
{
  int i; for (i = 0; i < Dim.num_time; i++) data[i] = data_input[i];
}

void read_data_matrix(const int *data_matrix_input)
{
  int i, j;
  for (j = 0; j < Dim.num_categ; j++)
    for (i = 0; i < Dim.num_time; i++)
      data_matrix[i][j] = (double) data_matrix_input[i + j*Dim.num_time];
}

void read_covariates(const double *design_matrix_input)
{
  int i,j;
  for (i = 0; i < Dim.num_time; i++)
    for (j = 0; j < Dim.num_regr; j++)
      design_matrix[i][j] = design_matrix_input[i + j*Dim.num_time];
}

/* ------------------------------ parameters --------------------------------- */
/* add a file-scope cache */
static double scale_std = 1.0;  /* = sqrt(1 - rho^2) */

/* after you read parameters: */
static void read_parameters(const double *param) {
  int i;
  Param.tau[0] = 0.0;
  for (i = 1; i < (Dim.num_categ - 1); i++)
    Param.tau[i] = param[i-1];
  for (i = 0; i < Dim.num_regr; i++)
    Param.beta[i] = param[i + Dim.num_categ - 2];
  Param.phi = param[Dim.num_categ + Dim.num_regr - 2];
  
  /* scale to convert non-standard marginals to N(0,1) */
  double one_minus_rho2 = 1.0 - Param.phi * Param.phi;
  if (one_minus_rho2 < 1e-15) one_minus_rho2 = 1e-15;
  scale_std = sqrt(one_minus_rho2);  /* NOTE: multiply bounds by this */
}

/* standardize bounds here */
static void build_limits(const int data_val,
                         const double mu,
                         double* lim_inf,
                         double* lim_sup,
                         int* infin)
{
  int x = data_val;
  if (x > 1 && x < Dim.num_categ) {
    *lim_inf = scale_std * (Param.tau[x-2] - mu);
    *lim_sup = scale_std * (Param.tau[x-1] - mu);
    *infin = 2;
  } else if (x == 1) {
    *lim_inf = -BIG;
    *lim_sup = scale_std * (Param.tau[0] - mu);  /* = -scale_std*mu */
*infin = 0;
  } else { /* x == K */
*lim_inf = scale_std * (Param.tau[Dim.num_categ - 2] - mu);
    *lim_sup = BIG;
    *infin = 1;
  }
}

/* ------------------------------ Objective --------------------------------- */

void CLSEW_aop_Rcall(const double *param, double *CLSEW)
{
  *CLSEW = CLSEW_aop(param);
}

static double CLSEW_aop(const double *param)
{
  int i, j;
  double sum = 0.0;
  
  read_parameters(param);
  
  /* ---- compute mu_t under the reparameterization ----
   mu_t = x_t' beta + phi * mu_{t-1},   with mu_0 = 0
   (equivalent to sum_{j=0}^{t-1} phi^j x_{t-j}' beta)
   */
  {
    double mu_prev = 0.0;                  /* μ_0 */
  for (i = 0; i < Dim.num_time; ++i) {
    double xb = 0.0;
    for (j = 0; j < Dim.num_regr; ++j)
      xb += Param.beta[j] * design_matrix[i][j];
    double mu_t = xb + Param.phi * mu_prev;
    means[i] = mu_t;
    mu_prev  = mu_t;
  }
  }
  
  /* ---- t = 1: marginal probabilities (USE STANDARDIZED BOUNDS) ---- */
  // {
  //   for (int k = 1; k <= Dim.num_categ; ++k) {
  //     double a, b; int inf;
  //     /* returns z-scores: scale_std * (tau - mu) */
  //     build_limits(k, means[0], &a, &b, &inf);
  //     
  //     double pk  = pnorm(b, 0.0, 1.0, 1, 0) - pnorm(a, 0.0, 1.0, 1, 0);
  //     double Ik  = data_matrix[0][k-1];
  //     double dif = Ik - pk;
  //     sum += dif * dif;
  //   }
  // }
  
  /* ---- t >= 2: conditional probabilities, lag = 1 ---- */
  for (i = 0; i < (Dim.num_time - 1); i++) {
    current_index = i;               /* i = t-1 */
  sum += CLSEW_aop_single(param);
  }
  
  return sum;
}

/* contributes sum_k ( I_{t,k} - P(Y_t=k | Y_{t-1}) )^2 for t = current_index+1 */
static double CLSEW_aop_single(const double *param)
{
  (void)param;
  int tprev = current_index;       /* Y_{t-1} */
int t     = current_index + 1;   /* Y_t     */

double s = 0.0;
for (int k = 1; k <= Dim.num_categ; k++) {
  double gk;
  trans_prob_aop(tprev, k, &gk);           /* P(Y_t=k | Y_{t-1}) */
double Ik = data_matrix[t][k-1];         /* I_{t,k} */
double diff = Ik - gk;
s += diff * diff;
}

return s;
}

/* P(X_{t+1}=k | X_t) using lag-1 BVN rectangle */
static void trans_prob_aop(const int position_t, const int k, double *result)
{
  int t = position_t;        /* time t   -> X_t is observed data[t] */
int l = position_t + 1;    /* time t+1 -> X_{t+1} target category k */

double corr = Param.phi;   /* lag-1 only */

double lower[2], upper[2];
int    infin[2];

/* dim1: X_{t+1} in category k, shifted by mean means[l] */
build_limits(k,       means[l], &lower[0], &upper[0], &infin[0]);

/* dim2: X_t in its observed category data[t], shifted by mean means[t] */
build_limits(data[t], means[t], &lower[1], &upper[1], &infin[1]);

/* denominator is the marginal probability for dim2 (the conditioning event X_t) */
double denom = pnorm(upper[1], 0.0, 1.0, 1, 0) - pnorm(lower[1], 0.0, 1.0, 1, 0);
if (denom < 1e-14) denom = 1e-14;

if (fabs(corr) > 1e-15) {
  double nom = F77_NAME(bvnmvn)(lower, upper, infin, &corr);
  if (!R_finite(nom)) nom = 0.0;
  *result = nom / denom;
} else {
  /* independence fallback => unconditional P(X_{t+1}=k) */
  *result = pnorm(upper[0], 0.0, 1.0, 1, 0) - pnorm(lower[0], 0.0, 1.0, 1, 0);
}
}
