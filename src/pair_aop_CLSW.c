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
void         score_by_t_CLSEW_aop(const double *param, double *Wout);

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

static void read_parameters(const double *param)
{
  int i;
  Param.tau[0] = 0.0;
for (i = 1; i < (Dim.num_categ - 1); i++)
  Param.tau[i] = param[i-1];

for (i = 0; i < Dim.num_regr; i++)
  Param.beta[i] = param[i + Dim.num_categ - 2];
  Param.phi = param[Dim.num_categ + Dim.num_regr - 2];
}

static void build_limits(const int data_val,
                         const double regr,
                         double* lim_inf,
                         double* lim_sup,
                         int* infin)
{
  int x = data_val;
  if (x > 1 && x < Dim.num_categ) {
    *lim_inf = Param.tau[x-2] - regr;
    *lim_sup = Param.tau[x-1] - regr;
    *infin = 2;
  } else if (x == 1) {
    *lim_inf = -BIG;
    *lim_sup = Param.tau[0] - regr;
    *infin = 0;
  } else {
    *lim_inf = Param.tau[Dim.num_categ - 2] - regr;
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

  /* mu_t = X_t' beta */
  for (i = 0; i < Dim.num_time; i++) {
    means[i] = 0.0;
    for (j = 0; j < Dim.num_regr; j++) means[i] += Param.beta[j] * design_matrix[i][j];
  }

  /* ---- t = 1: marginal probabilities ---- */
  {
    double mst = means[0];
    for (int k = 1; k <= Dim.num_categ; k++) {
      double a = (k == 1) ? -BIG : Param.tau[k-2];
      double b = (k == Dim.num_categ) ? BIG : Param.tau[k-1];
      double pk = pnorm(b - mst, 0.0, 1.0, 1, 0) - pnorm(a - mst, 0.0, 1.0, 1, 0);
      double Ik = data_matrix[0][k-1];  /* I_{1,k} */
  double diff = Ik - pk;
  sum += diff * diff;
    }
  }

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
int t     = current_index + 1;     /* Y_t     */

double s = 0.0;
for (int k = 1; k <= Dim.num_categ; k++) {
  double gk;
  trans_prob_aop(tprev, k, &gk);           /* P(Y_t=k | Y_{t-1}) */
double Ik = data_matrix[t][k-1];           /* I_{t,k} */
double diff = Ik - gk;
s += diff * diff;
}

return s;
}

/* P(X_{t+1}=k | X_t) using lag-1 BVN rectangle */
static void trans_prob_aop(const int position_t, const int k, double *result)
{
  int t = position_t;        /* time t   -> X_t is observed data[t] */
int l = position_t + 1;    /* time t+1   -> X_{t+1} target category k */

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

/* ------------------------------ Gradient ---------------------------------- */

void gradient_CLSEW_aop(const double *param,
                        double *gradient)
{
  int i,t;

  read_parameters(param);

  /* reset */
  for (i = 0; i < Dim.num_param; i++) gradient[i] = 0.0;

  /* precompute rho helpers */
  One_Rho2          = 1.0 - Param.phi*Param.phi;
  if (One_Rho2 < 1e-15) One_Rho2 = 1e-15;
  sqrt_One_Rho2     = sqrt(One_Rho2);
  Inv_TwoPiOne_Rho2 = 1.0 / (M_2PI * sqrt_One_Rho2);

  /* clear caches */
  for (t = 0; t < Dim.num_time; t++) {
    means[t]          = 0.0;
    Ct_Mut[t]         = 0.0;
    Ctm1_Mut[t]       = 0.0;
    density_Ct_Mut[t] = 0.0;
    density_Ctm1_Mut[t]= 0.0;
  }

  /* compute means and boundaries */
  for (t = 0; t < Dim.num_time; t++) {
    for (i = 0; i < Dim.num_regr; i++)
      means[t] += Param.beta[i]*design_matrix[t][i];

    if (data[t] == 1) {
      Ct_Mut[t]   = Param.tau[data[t]-1] - means[t];
      Ctm1_Mut[t] = -BIG;
    } else if (data[t] < Dim.num_categ && data[t] > 1) {
      Ct_Mut[t]   = Param.tau[data[t]-1] - means[t];
      Ctm1_Mut[t] = Param.tau[data[t]-2] - means[t];
    } else if (data[t] == Dim.num_categ) {
      Ct_Mut[t]   = BIG;
      Ctm1_Mut[t] = Param.tau[data[t]-2] - means[t];
    }

    density_Ct_Mut[t]   = dnorm(Ct_Mut[t],   0.0, 1.0, 0);
    density_Ctm1_Mut[t] = dnorm(Ctm1_Mut[t], 0.0, 1.0, 0);
  }

  /* ---------- include t = 1 marginal term ---------- */
  current_index_g = -1;                  /* triggers marginal branch */
  gradient_CLSEW_aop_single(param);
  for (i = 0; i < Dim.num_param; i++)
    gradient[i] += gradient_single[i];

  /* ---------- conditional terms for t >= 2 ---------- */
  for (t = 0; t < (Dim.num_time - 1); t++) {
    current_index_g = t;
    gradient_CLSEW_aop_single(param);
    for (i = 0; i < Dim.num_param; i++)
      gradient[i] += gradient_single[i];
  }

  /* chain rule for (Ik - Pk)^2 */
  for (i = 0; i < Dim.num_param; ++i)
    gradient[i] *= -2.0;
}

/* one step: sum over k the contributions (Ik - Pk) dPk/dtheta into gradient_single
 current_index_g == -1  => t = 1 marginal branch
 current_index_g >= 0   => conditional term at t = current_index_g + 1       */
static void gradient_CLSEW_aop_single(const double *param)
{
  (void)param; /* params already read in driver */
int i, k;

/* initialise */
for (i = 0; i < Dim.num_param; ++i)
  gradient_single[i] = 0.0;

/* ---------------- t = 1 (marginal) ---------------- */
if (current_index_g < 0) {

  const int t0 = 0;
  const double mu = means[t0];

  /* precompute residuals r_k = I_{1,k} - p_{1,k} and reuse later */
  /* using gradient_this_pair as temporary scratch for residuals */
  for (k = 1; k <= Dim.num_categ; ++k) {
    double a, b; int inf;
    build_limits(k, mu, &a, &b, &inf);
    double pk = pnorm(b, 0.0, 1.0, 1, 0) - pnorm(a, 0.0, 1.0, 1, 0);
    gradient_this_pair[k-1] = data_matrix[t0][k-1] - pk;  /* residual */
  }

  /* beta gradient: d pk / d mu = -(phi(b) - phi(a)); mu = X beta */
  for (k = 1; k <= Dim.num_categ; ++k) {
    double a, b; int inf;
    build_limits(k, mu, &a, &b, &inf);

    const double phi_a = (inf != 0) ? dnorm(a, 0.0, 1.0, 0) : 0.0;
    const double phi_b = (inf != 1) ? dnorm(b, 0.0, 1.0, 0) : 0.0;
    const double dpk_dmu = -(phi_b - phi_a);
    const double res_k   = gradient_this_pair[k-1];

    for (i = 0; i < Dim.num_regr; ++i) {
      const int idx_beta = (Dim.num_categ - 2) + i;
      gradient_single[idx_beta] += res_k * dpk_dmu * design_matrix[t0][i];
    }
  }

  /* tau_j gradient (j = 2..K-1): (res_j - res_{j+1}) * phi(τ_j - μ) */
  if (Dim.num_categ >= 3) {
    int j;
    for (j = 2; j <= Dim.num_categ - 1; ++j) {
      const double zj = Param.tau[j-1] - mu;  /* τ_j - μ */
  const double ph = dnorm(zj, 0.0, 1.0, 0);

  const double res_j   = gradient_this_pair[j-1];   /* I_{1,j}   - p_{1,j}   */
  const double res_j1  = gradient_this_pair[j];     /* I_{1,j+1} - p_{1,j+1} */

  const int par_idx = j - 2; /* tau2..tau_{K-1} -> indices 0..K-3 */
  gradient_single[par_idx] += (res_j - res_j1) * ph;
    }
  }

  /* no phi contribution at t=1 marginal */
  return;
}

/* ---------------- conditional term at t = current_index_g + 1 ---------------- */
{
  const int tprev = current_index_g;
  for (k = 1; k <= Dim.num_categ; ++k) {
    gradient_CLSEW_pair_aop(tprev, k, gradient_this_pair);
    for (i = 0; i < Dim.num_param; ++i)
      gradient_single[i] += gradient_this_pair[i];
  }
}
}

/* position1 = tprev, position2 = k
 Write (Ik - Pk) * dPk/dtheta into gradient_this_pair */
void gradient_CLSEW_pair_aop(const int position1,
                             const int position2,
                             double *gradient_this_pair)
{
  int i;

  const int tprev = position1;
  const int t     = position1 + 1;
  const int k     = position2;

  for (i = 0; i < Dim.num_param; ++i) gradient_this_pair[i] = 0.0;

  const double mu1 = means[t];       /* μ_t */
const double mu2 = means[tprev];   /* μ_{t-1} */

/* rectangle bounds via build_limits */
double ll[2], uu[2]; int inf[2];
build_limits(k,            mu1, &ll[0], &uu[0], &inf[0]);
build_limits(data[tprev],  mu2, &ll[1], &uu[1], &inf[1]);

const double a1 = ll[0], b1 = uu[0];
const double a2 = ll[1], b2 = uu[1];

double denom = pnorm(b2, 0.0, 1.0, 1, 0) - pnorm(a2, 0.0, 1.0, 1, 0);
if (denom < 1e-14) denom = 1e-14;

const double rho = Param.phi;
/* guard s against roundoff when |rho| ~ 1 */
const double s2 = 1.0 - rho*rho;
const double s  = sqrt(s2 > 1e-15 ? s2 : 1e-15);

double num = F77_NAME(bvnmvn)(ll, uu, inf, (double*)&rho);
if (!R_finite(num)) num = 0.0;

const double Pk    = num / denom;
const double Ik    = data_matrix[t][k-1];
const double resid = Ik - Pk;

const int inf1 = inf[0], inf2 = inf[1];
const double phi_a1 = (inf1 != 0) ? dnorm(a1, 0.0, 1.0, 0) : 0.0;
const double phi_b1 = (inf1 != 1) ? dnorm(b1, 0.0, 1.0, 0) : 0.0;
const double phi_a2 = (inf2 != 0) ? dnorm(a2, 0.0, 1.0, 0) : 0.0;
const double phi_b2 = (inf2 != 1) ? dnorm(b2, 0.0, 1.0, 0) : 0.0;

double dN_da1 = 0.0, dN_db1 = 0.0, dN_da2 = 0.0, dN_db2 = 0.0;

if (inf1 != 0) {
  const double U = (inf2 != 1) ? pnorm((b2 - rho*a1)/s, 0.0, 1.0, 1, 0) : 1.0;
  const double L = (inf2 != 0) ? pnorm((a2 - rho*a1)/s, 0.0, 1.0, 1, 0) : 0.0;
  dN_da1 = -phi_a1 * (U - L);
}

if (inf1 != 1) {
  const double U = (inf2 != 1) ? pnorm((b2 - rho*b1)/s, 0.0, 1.0, 1, 0) : 1.0;
  const double L = (inf2 != 0) ? pnorm((a2 - rho*b1)/s, 0.0, 1.0, 1, 0) : 0.0;
  dN_db1 =  phi_b1 * (U - L);
}

if (inf2 != 0) {
  const double U = (inf1 != 1) ? pnorm((b1 - rho*a2)/s, 0.0, 1.0, 1, 0) : 1.0;
  const double L = (inf1 != 0) ? pnorm((a1 - rho*a2)/s, 0.0, 1.0, 1, 0) : 0.0;
  dN_da2 = -phi_a2 * (U - L);
}

if (inf2 != 1) {
  const double U = (inf1 != 1) ? pnorm((b1 - rho*b2)/s, 0.0, 1.0, 1, 0) : 1.0;
  const double L = (inf1 != 0) ? pnorm((a1 - rho*b2)/s, 0.0, 1.0, 1, 0) : 0.0;
  dN_db2 =  phi_b2 * (U - L);
}

/* dnum/drho via signed corner pdfs */
double dN_drho = 0.0;
if (inf1 != 1 && inf2 != 1) dN_drho += mydnorm2(b1, b2, rho);
if (inf1 != 0 && inf2 != 1) dN_drho -= mydnorm2(a1, b2, rho);
if (inf1 != 1 && inf2 != 0) dN_drho -= mydnorm2(b1, a2, rho);
if (inf1 != 0 && inf2 != 0) dN_drho += mydnorm2(a1, a2, rho);

/* denominator partials wrt mu2 and its bounds */
const double dDen_da2  = (inf2 != 0) ? -phi_a2 : 0.0;
const double dDen_db2  = (inf2 != 1) ?  phi_b2 : 0.0;
const double dDen_dmu2 = -(phi_b2 - phi_a2);

/* betas via mu1, mu2 (quotient rule) */
{
  const double dN_dmu1 = -(dN_da1 + dN_db1);
  const double dN_dmu2 = -(dN_da2 + dN_db2);
  for (i = 0; i < Dim.num_regr; ++i) {
    const double x1 = design_matrix[t][i];
    const double x2 = design_matrix[tprev][i];
    const double dNum = dN_dmu1 * x1 + dN_dmu2 * x2;
    const double dDen = dDen_dmu2 * x2;
    const double dPk  = (dNum * denom - num * dDen) / (denom * denom);
    const int idx_beta = (Dim.num_categ - 2) + i;
    gradient_this_pair[idx_beta] += resid * dPk;
  }
}

if (k > 1) {
  const int par_idx = (k - 2) - 1;
if (par_idx >= 0) {
  const double dPk = dN_da1 / denom;
  gradient_this_pair[par_idx] += resid * dPk;
}
}
if (k < Dim.num_categ) {
  const int par_idx = (k - 1) - 1;
if (par_idx >= 0) {
  const double dPk = dN_db1 / denom;
  gradient_this_pair[par_idx] += resid * dPk;
}
}

{
  const int c = data[tprev];
  if (c >= 3) {
    const int par_idx = (c - 3);
    const double dPk = (dN_da2 * denom - num * dDen_da2) / (denom * denom);
    gradient_this_pair[par_idx] += resid * dPk;
  }
  if (c <= Dim.num_categ - 1) {
    const int par_idx = (c - 2);
    if (par_idx >= 0) {
      const double dPk = (dN_db2 * denom - num * dDen_db2) / (denom * denom);
      gradient_this_pair[par_idx] += resid * dPk;
    }
  }
}

/* phi (rho) — denom independent of rho */
{
  const int idx_phi = Dim.num_categ + Dim.num_regr - 2;
  const double dPk = dN_drho / denom;
  gradient_this_pair[idx_phi] += resid * dPk;
}
}

/* plain BVN pdf using the precomputed Rho helpers */
static double mydnorm2(const double x1,
                       const double x2,
                       const double rho)
{
  return Inv_TwoPiOne_Rho2 * exp(-0.5 * (x1*x1 + x2*x2 - 2.0*rho*x1*x2) / One_Rho2);
}

/* -----------------------------------------------------------------------
 * Moment-based "meat" matrix W_n for the CLS-lag1 objective.
 *
 * On input:
 *   param  : parameter vector θ of length Dim.num_param
 * On output:
 *   Wout   : length Dim.num_param * Dim.num_param (column-major),
 *            containing
 *              W_n = 4 * sum_{t=2}^T L_t^T Cov(Y_t | F_{t-1}) L_t,
 *            where rows of L_t are l_{t-1,k} = dp_{t,k}/dθ.
 *
 * NOTE: we start at t = 2 (index t=1 in C) and ignore t = 1
 *       (marginal term); this has O(1/n) effect on the asymptotics.
 * ----------------------------------------------------------------------- */
void score_by_t_CLSEW_aop(const double *param,
                          double *Wout)
{
  int i, j, t, k, kp;

  const int P = Dim.num_param;   /* number of parameters */
  const int K = Dim.num_categ;   /* number of categories */
  const int T = Dim.num_time;    /* length of series */

  /* 1) Read parameters and precompute rho helpers */
  read_parameters(param);

  One_Rho2          = 1.0 - Param.phi * Param.phi;
  if (One_Rho2 < 1e-15) One_Rho2 = 1e-15;
    sqrt_One_Rho2     = sqrt(One_Rho2);
    Inv_TwoPiOne_Rho2 = 1.0 / (M_2PI * sqrt_One_Rho2);

  /* 2) Compute means: mu_t = X_t' beta */
  for (t = 0; t < T; ++t) {
    means[t] = 0.0;
    for (i = 0; i < Dim.num_regr; ++i)
      means[t] += Param.beta[i] * design_matrix[t][i];
  }

  /* 3) Initialize Wout to zero */
  for (i = 0; i < P * P; ++i)
    Wout[i] = 0.0;

  /* 4) Allocate temporaries:
  *    - L_flat: K x P matrix of l_{t-1,k} in row-major (k = row)
  *    - p_t   : length K vector of p_{t,k}
  *    - saved_row: original indicator row at time t
  *    - grad1, grad0: temporary gradients for Ik=1 and Ik=0
  */
  double *L_flat    = (double*) R_Calloc(K * P, double);
  double *p_t       = (double*) R_Calloc(K, double);
  double *saved_row = (double*) R_Calloc(K, double);
  double *grad1     = (double*) R_Calloc(P, double);
  double *grad0     = (double*) R_Calloc(P, double);

  /* 5) Loop over t = 2,...,T (C indices t=1,...,T-1) */
  for (t = 1; t < T; ++t) {
    int tprev = t - 1;

    /* zero L_flat and p_t for this t */
    for (k = 0; k < K; ++k) {
      p_t[k] = 0.0;
      for (i = 0; i < P; ++i)
        L_flat[k * P + i] = 0.0;
    }

    /* save original row t of data_matrix */
    for (k = 0; k < K; ++k)
      saved_row[k] = data_matrix[t][k];

    /* compute p_{t,k} and l_{t-1,k} = dp_{t,k}/dθ for each category k */
    for (k = 1; k <= K; ++k) {

      /* --- step 1: p_{t,k} = P(Y_t = k | Y_{t-1}) via trans_prob_aop --- */
      double Pk;
      trans_prob_aop(tprev, k, &Pk);
      p_t[k - 1] = Pk;

      /* --- step 2: gradient wrt theta using two calls to gradient_CLSEW_pair_aop --- */

      /* case A: Ik = 1 at time t in category k */
      for (kp = 0; kp < K; ++kp) data_matrix[t][kp] = 0.0;
      data_matrix[t][k - 1] = 1.0;
      gradient_CLSEW_pair_aop(tprev, k, grad1);

      /* case B: Ik = 0 at time t (all zero indicators) */
      for (kp = 0; kp < K; ++kp) data_matrix[t][kp] = 0.0;
      gradient_CLSEW_pair_aop(tprev, k, grad0);

      /* store l_{t-1,k} = grad1 - grad0 (dimension P) */
      for (i = 0; i < P; ++i)
        L_flat[(k - 1) * P + i] = grad1[i] - grad0[i];
    }

    /* restore original data_matrix row t */
    for (k = 0; k < K; ++k)
      data_matrix[t][k] = saved_row[k];

    /* --- step 3: accumulate W_t = 4 * L^T Cov(Y_t|F_{t-1}) L --- */
    for (i = 0; i < P; ++i) {
      for (j = 0; j < P; ++j) {
        double accum = 0.0;

        /* sum over k, k' (Cov = diag(p) - p p^T) */
        for (k = 0; k < K; ++k) {
          double l_ki = L_flat[k * P + i];
          double pk   = p_t[k];

          /* diagonal: k' = k */
          {
            double cov_diag = pk * (1.0 - pk);
            double l_kj     = L_flat[k * P + j];
            accum += cov_diag * l_ki * l_kj;
          }

          /* off-diagonal: k' != k */
          for (kp = 0; kp < K; ++kp) {
            if (kp == k) continue;
            double pk2    = p_t[kp];
            double cov_off = - pk * pk2;
            double l_kp_j  = L_flat[kp * P + j];
            accum += cov_off * l_ki * l_kp_j;
          }
        }

        Wout[i + j * P] += 4.0 * accum;
      }
    }
  }

  R_Free(L_flat);
  R_Free(p_t);
  R_Free(saved_row);
  R_Free(grad1);
  R_Free(grad0);
}
