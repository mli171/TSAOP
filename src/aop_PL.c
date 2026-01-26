/*  pair_aop_pl_arp.c
 *
 *  Pairwise (composite) log-likelihood for ordinal AOP with latent Gaussian AR(p):
 *      Z_t = mu_t + u_t,   u_t stationary AR(p), Var(u_t)=1
 *      Y_t = k  <=>  tau_{k-1} <= Z_t < tau_k, with tau_1 fixed at 0 (as in your code)
 *
 *  This file is SELF-CONTAINED and safe to compile alongside pair_aop_clsw.c
 *  because all exported symbols here are uniquely named (suffix _pl_arp).
 *
 *  Parameter vector (recommended):
 *      param = [ tau_2..tau_{K-1},  beta_1..beta_q,  phi_1..phi_p ]
 *  length = (K-2) + q + p
 *
 *  Pair set:
 *      all pairs (t-h, t) with h=1..L and t=1..T-1 (skipping missing), averaged by #pairs.
 *
 *  Exported C entry points (for .C):
 *      read_dimensions_pl_arp(num_categ, num_regr, num_time)
 *      read_pl_ar_order_pl_arp(p)
 *      read_order_pair_lik_pl_arp(L)              // max lag for pairs
 *      allocate_pl_arp()
 *      read_data_pl_arp(y)
 *      read_covariates_pl_arp(X)                  // column-major T x q
 *
 *      logPair_aop_ARp_Rcall_pl_arp(param, out)    // average log pair prob
 *      gradient_logPair_aop_ARp_pl_arp(param, grad)// average gradient
 *      score_by_t_logPair_aop_ARp_pl_arp(param, scores) // T x num_param, column-major
 *
 *      deallocate_pl_arp()
 */

#include <R.h>
#include <Rmath.h>
#include <R_ext/RS.h>
#include <R_ext/Applic.h>
#include <R_ext/Memory.h>
#include "aop_PL.h"

#define MISSING_VALUE -999
#define IS_MISSING(x) ((x) == MISSING_VALUE ? 1 : 0)
#define SMALL 1e-16
#define BIG   1e+16

/* Fortran BVNMVN is a FUNCTION returning double (as in your CLSEW file) */
extern double F77_NAME(bvnmvn)(double *lower, double *upper, int *infin, double *correl);

/* ----------------------------- static state -------------------------------- */

static int is_allocated_pl = 0;

/* dimensions (single time series, like your CLSEW file) */
struct DimensionsPL {
  int num_categ;   /* K */
  int num_regr;    /* q */
  int num_time;    /* T */
  int num_param;   /* (K-2) + q + p */
};
static struct DimensionsPL Dim = {0,0,0,0};

/* parameters */
struct ParametersPL {
  double *tau;   /* length K-1, tau[0]=0 fixed, tau[1..K-2] estimated */
  double *beta;  /* length q */
  double *phi;   /* length p */
};
static struct ParametersPL Param = {NULL,NULL,NULL};

/* data */
static int    *data           = NULL;  /* length T */
static double **design_matrix = NULL;  /* T x q */
static double *means          = NULL;  /* length T */

/* pair settings */
static int ar_order_p   = 1; /* p */
static int max_lag_L    = 1; /* L (order_pair) */

/* rho(h) and derivatives d rho(h) / d phi_r */
static double *rho      = NULL; /* length L+1, rho[0]=1 */
static double *drho     = NULL; /* length p*(L+1), drho[(r-1)*(L+1)+h] */

/* solver scratch for Yule–Walker */
static double *A_mat    = NULL; /* p*p (column-major) */
static double *b_vec    = NULL; /* p */
static double *x_vec    = NULL; /* p */
static double *v_vec    = NULL; /* p */

/* gradient scratch */
static double *grad_pair = NULL; /* length num_param */

/* ----------------------------- helpers ------------------------------------- */

static inline double bvn_pdf(const double x1, const double x2, const double r)
{
  double one = 1.0 - r*r;
  if (one < 1e-15) one = 1e-15;
  return (1.0/(M_2PI*sqrt(one))) * exp(-0.5*(x1*x1 + x2*x2 - 2.0*r*x1*x2)/one);
}

static void build_limits_pl(const int y,
                            const double mu,
                            double *lim_inf,
                            double *lim_sup,
                            int    *infin)
{
  /* bounds are tau - mu (sd=1) */
  if (y > 1 && y < Dim.num_categ) {
    *lim_inf = Param.tau[y-2] - mu;  /* tau_{y-1} - mu */
    *lim_sup = Param.tau[y-1] - mu;  /* tau_{y}   - mu */
    *infin   = 2;
  } else if (y == 1) {
    *lim_inf = -BIG;
    *lim_sup = Param.tau[0] - mu;    /* tau_1 - mu = -mu */
    *infin   = 0;
  } else { /* y == K */
    *lim_inf = Param.tau[Dim.num_categ-2] - mu; /* tau_{K-1} - mu */
    *lim_sup = BIG;
    *infin   = 1;
  }
}

/* small linear solver: Gauss–Jordan with partial pivoting, column-major A */
static int solve_lin_sys(const int n, double *A, double *b, double *x)
{
  for (int i = 0; i < n; ++i) x[i] = b[i];

  for (int k = 0; k < n; ++k) {
    int piv = k;
    double amax = fabs(A[k + k*n]);
    for (int i = k+1; i < n; ++i) {
      double v = fabs(A[i + k*n]);
      if (v > amax) { amax = v; piv = i; }
    }
    if (amax < 1e-12) return 0;

    if (piv != k) {
      for (int j = k; j < n; ++j) {
        double tmp = A[k + j*n];
        A[k + j*n] = A[piv + j*n];
        A[piv + j*n] = tmp;
      }
      double tb = x[k]; x[k] = x[piv]; x[piv] = tb;
    }

    double diag = A[k + k*n];
    for (int j = k; j < n; ++j) A[k + j*n] /= diag;
    x[k] /= diag;

    for (int i = 0; i < n; ++i) if (i != k) {
      double f = A[i + k*n];
      if (fabs(f) < 1e-18) continue;
      for (int j = k; j < n; ++j) A[i + j*n] -= f * A[k + j*n];
      x[i] -= f * x[k];
    }
  }
  return 1;
}

/* compute rho(1..L) and drho/dphi for AR(p) via Yule–Walker + recursion
   returns 1 if OK, 0 if singular/invalid. */
static int compute_rho_and_drho(void)
{
  const int p = ar_order_p;
  const int L = max_lag_L;

  rho[0] = 1.0;
  for (int h = 1; h <= L; ++h) rho[h] = 0.0;
  for (int i = 0; i < p*(L+1); ++i) drho[i] = 0.0;

  /* Build A r = b for r = (rho(1)..rho(p))
     Equation k=1..p:
       rho(k) - sum_{j!=k} phi_j rho(|k-j|) = phi_k
  */
  for (int i = 0; i < p*p; ++i) A_mat[i] = 0.0;
  for (int k = 1; k <= p; ++k) {
    b_vec[k-1] = Param.phi[k-1];
    A_mat[(k-1) + (k-1)*p] = 1.0;
    for (int j = 1; j <= p; ++j) if (j != k) {
      int m = abs(k - j); /* 1..p-1 */
      A_mat[(k-1) + (m-1)*p] -= Param.phi[j-1];
    }
  }

  /* solve for rho(1..p) */
  double *A0 = (double*) R_alloc((size_t)(p*p), sizeof(double));
  double *b0 = (double*) R_alloc((size_t)p,     sizeof(double));
  for (int i = 0; i < p*p; ++i) A0[i] = A_mat[i];
  for (int i = 0; i < p;   ++i) b0[i] = b_vec[i];

  if (!solve_lin_sys(p, A0, b0, x_vec)) return 0;

  for (int k = 1; k <= p && k <= L; ++k) rho[k] = x_vec[k-1];

  /* extend rho(h) for h>p by recursion rho(h)=sum phi_j rho(h-j) */
  for (int h = p+1; h <= L; ++h) {
    double s = 0.0;
    for (int j = 1; j <= p; ++j) s += Param.phi[j-1] * rho[h-j];
    rho[h] = s;
  }

  /* reject if correlations are invalid/near-singular */
  for (int h = 1; h <= L; ++h) {
    if (!R_finite(rho[h]) || fabs(rho[h]) >= 0.999999) return 0;
  }

  /* derivatives for h=1..p via dr = A^{-1} v_r */
  for (int r = 1; r <= p; ++r) {
    for (int k = 1; k <= p; ++k) {
      double vk = (k == r) ? 1.0 : 0.0;
      if (k != r) vk += rho[abs(k-r)];
      v_vec[k-1] = vk;
    }

    double *A1 = (double*) R_alloc((size_t)(p*p), sizeof(double));
    double *b1 = (double*) R_alloc((size_t)p,     sizeof(double));
    for (int i = 0; i < p*p; ++i) A1[i] = A_mat[i];
    for (int i = 0; i < p;   ++i) b1[i] = v_vec[i];

    if (!solve_lin_sys(p, A1, b1, x_vec)) return 0;

    for (int h = 1; h <= p && h <= L; ++h)
      drho[(r-1)*(L+1) + h] = x_vec[h-1];

    /* extend for h>p:
       d rho(h)/d phi_r = rho(h-r) + sum_j phi_j d rho(h-j)/d phi_r
    */
    for (int h = p+1; h <= L; ++h) {
      double s = rho[h-r];
      for (int j = 1; j <= p; ++j)
        s += Param.phi[j-1] * drho[(r-1)*(L+1) + (h-j)];
      drho[(r-1)*(L+1) + h] = s;
    }
  }

  return 1;
}

/* compute P and rectangle derivatives for pair (t2,t1) with t1>t2, lag=h */
static void pair_prob_and_derivs(const int t1, const int t2, const int h,
                                 double *P,
                                 double *dP_da1, double *dP_db1,
                                 double *dP_da2, double *dP_db2,
                                 double *dP_drho,
                                 double lower[2], double upper[2], int infin[2])
{
  /* dimension 1 = time t1, dimension 2 = time t2 */
  build_limits_pl(data[t1], means[t1], &lower[0], &upper[0], &infin[0]);
  build_limits_pl(data[t2], means[t2], &lower[1], &upper[1], &infin[1]);

  double r = rho[h];
  double s2 = 1.0 - r*r;
  if (s2 < 1e-15) s2 = 1e-15;
  double s = sqrt(s2);

  double prob = F77_NAME(bvnmvn)(lower, upper, infin, &r);
  if (!R_finite(prob) || prob < SMALL) prob = SMALL;
  *P = prob;

  const double a1 = lower[0], b1 = upper[0];
  const double a2 = lower[1], b2 = upper[1];

  const int inf1 = infin[0], inf2 = infin[1];
  const double phi_a1 = (inf1 != 0) ? dnorm(a1, 0.0, 1.0, 0) : 0.0;
  const double phi_b1 = (inf1 != 1) ? dnorm(b1, 0.0, 1.0, 0) : 0.0;
  const double phi_a2 = (inf2 != 0) ? dnorm(a2, 0.0, 1.0, 0) : 0.0;
  const double phi_b2 = (inf2 != 1) ? dnorm(b2, 0.0, 1.0, 0) : 0.0;

  *dP_da1 = 0.0; *dP_db1 = 0.0; *dP_da2 = 0.0; *dP_db2 = 0.0; *dP_drho = 0.0;

  if (inf1 != 0) {
    const double U = (inf2 != 1) ? pnorm((b2 - r*a1)/s, 0.0, 1.0, 1, 0) : 1.0;
    const double L = (inf2 != 0) ? pnorm((a2 - r*a1)/s, 0.0, 1.0, 1, 0) : 0.0;
    *dP_da1 = -phi_a1 * (U - L);
  }
  if (inf1 != 1) {
    const double U = (inf2 != 1) ? pnorm((b2 - r*b1)/s, 0.0, 1.0, 1, 0) : 1.0;
    const double L = (inf2 != 0) ? pnorm((a2 - r*b1)/s, 0.0, 1.0, 1, 0) : 0.0;
    *dP_db1 =  phi_b1 * (U - L);
  }
  if (inf2 != 0) {
    const double U = (inf1 != 1) ? pnorm((b1 - r*a2)/s, 0.0, 1.0, 1, 0) : 1.0;
    const double L = (inf1 != 0) ? pnorm((a1 - r*a2)/s, 0.0, 1.0, 1, 0) : 0.0;
    *dP_da2 = -phi_a2 * (U - L);
  }
  if (inf2 != 1) {
    const double U = (inf1 != 1) ? pnorm((b1 - r*b2)/s, 0.0, 1.0, 1, 0) : 1.0;
    const double L = (inf1 != 0) ? pnorm((a1 - r*b2)/s, 0.0, 1.0, 1, 0) : 0.0;
    *dP_db2 =  phi_b2 * (U - L);
  }

  /* dP/drho via signed corner pdfs */
  double d = 0.0;
  if (inf1 != 1 && inf2 != 1) d += bvn_pdf(b1, b2, r);
  if (inf1 != 0 && inf2 != 1) d -= bvn_pdf(a1, b2, r);
  if (inf1 != 1 && inf2 != 0) d -= bvn_pdf(b1, a2, r);
  if (inf1 != 0 && inf2 != 0) d += bvn_pdf(a1, a2, r);
  *dP_drho = d;
}

/* read params: tau/beta like your CLSEW; phi_1..phi_p appended */
static void read_parameters_pl_arp(const double *param)
{
  Param.tau[0] = 0.0;
  for (int i = 1; i < (Dim.num_categ - 1); ++i)
    Param.tau[i] = param[i-1];

  for (int i = 0; i < Dim.num_regr; ++i)
    Param.beta[i] = param[i + Dim.num_categ - 2];

  for (int i = 0; i < ar_order_p; ++i)
    Param.phi[i] = param[(Dim.num_categ - 2) + Dim.num_regr + i];
}

/* ----------------------------- exported I/O -------------------------------- */

void read_dimensions_pl_arp(const int *num_categ,
                            const int *num_regr,
                            const int *num_time)
{
  Dim.num_categ = *num_categ;
  Dim.num_regr  = *num_regr;
  Dim.num_time  = *num_time;

  /* num_param depends on p; will be finalized after read_pl_ar_order_pl_arp */
  Dim.num_param = (Dim.num_categ - 2) + Dim.num_regr + ar_order_p;
}

void read_pl_ar_order_pl_arp(const int *p)
{
  ar_order_p = *p;
  if (ar_order_p < 1) ar_order_p = 1;
  Dim.num_param = (Dim.num_categ - 2) + Dim.num_regr + ar_order_p;
}

void read_order_pair_lik_pl_arp(const int *L)
{
  max_lag_L = *L;
  if (max_lag_L < 1) max_lag_L = 1;
}

/* allocate everything for PL-AR(p) */
void allocate_pl_arp(void)
{
  if (is_allocated_pl) return;

  const int T = Dim.num_time;
  const int K = Dim.num_categ;
  const int q = Dim.num_regr;
  const int p = ar_order_p;
  const int L = max_lag_L;

  data = R_Calloc(T, int);

  design_matrix = R_Calloc(T, double*);
  for (int t = 0; t < T; ++t) {
    design_matrix[t] = R_Calloc(q, double);
    for (int j = 0; j < q; ++j) design_matrix[t][j] = 0.0;
  }

  Param.tau  = R_Calloc((K - 1), double);
  Param.beta = R_Calloc(q, double);
  Param.phi  = R_Calloc(p, double);

  means = R_Calloc(T, double);

  rho  = R_Calloc(L + 1, double);
  drho = R_Calloc(p * (L + 1), double);

  A_mat = R_Calloc(p * p, double);
  b_vec = R_Calloc(p, double);
  x_vec = R_Calloc(p, double);
  v_vec = R_Calloc(p, double);

  grad_pair = R_Calloc(Dim.num_param, double);

  is_allocated_pl = 1;
}

void deallocate_pl_arp(void)
{
  if (!is_allocated_pl) return;

  if (data) { R_Free(data); data = NULL; }

  if (design_matrix) {
    for (int t = 0; t < Dim.num_time; ++t) {
      if (design_matrix[t]) R_Free(design_matrix[t]);
    }
    R_Free(design_matrix);
    design_matrix = NULL;
  }

  if (Param.tau)  { R_Free(Param.tau);  Param.tau = NULL; }
  if (Param.beta) { R_Free(Param.beta); Param.beta = NULL; }
  if (Param.phi)  { R_Free(Param.phi);  Param.phi = NULL; }

  if (means) { R_Free(means); means = NULL; }

  if (rho)  { R_Free(rho);  rho  = NULL; }
  if (drho) { R_Free(drho); drho = NULL; }

  if (A_mat) { R_Free(A_mat); A_mat = NULL; }
  if (b_vec) { R_Free(b_vec); b_vec = NULL; }
  if (x_vec) { R_Free(x_vec); x_vec = NULL; }
  if (v_vec) { R_Free(v_vec); v_vec = NULL; }

  if (grad_pair) { R_Free(grad_pair); grad_pair = NULL; }

  is_allocated_pl = 0;
}

void read_data_pl_arp(const int *data_input)
{
  for (int t = 0; t < Dim.num_time; ++t) data[t] = data_input[t];
}

void read_covariates_pl_arp(const double *design_matrix_input)
{
  /* input is column-major T x q */
  for (int t = 0; t < Dim.num_time; ++t)
    for (int j = 0; j < Dim.num_regr; ++j)
      design_matrix[t][j] = design_matrix_input[t + j * Dim.num_time];
}

/* ----------------------------- objective ----------------------------------- */

void logPair_aop_ARp_Rcall_pl_arp(const double *param, double *out)
{
  read_parameters_pl_arp(param);

  /* mu_t = X_t' beta */
  for (int t = 0; t < Dim.num_time; ++t) {
    means[t] = 0.0;
    for (int j = 0; j < Dim.num_regr; ++j)
      means[t] += Param.beta[j] * design_matrix[t][j];
  }

  if (!compute_rho_and_drho()) {
    *out = -BIG;
    return;
  }

  double sum = 0.0;
  int count = 0;

  for (int t = 1; t < Dim.num_time; ++t) {
    int hmax = (t < max_lag_L) ? t : max_lag_L;
    for (int h = 1; h <= hmax; ++h) {
      int s = t - h;
      if (IS_MISSING(data[t]) || IS_MISSING(data[s])) continue;

      double lo[2], up[2]; int inf[2];
      build_limits_pl(data[t], means[t], &lo[0], &up[0], &inf[0]);
      build_limits_pl(data[s], means[s], &lo[1], &up[1], &inf[1]);

      double r = rho[h];
      double p2 = F77_NAME(bvnmvn)(lo, up, inf, &r);
      if (!R_finite(p2) || p2 < SMALL) p2 = SMALL;

      sum += log(p2);
      count++;
    }
  }

  *out = (count > 0) ? (sum / (double)count) : 0.0;
}

/* ----------------------------- gradient ------------------------------------ */

void gradient_logPair_aop_ARp_pl_arp(const double *param, double *gradient)
{
  read_parameters_pl_arp(param);

  /* mu_t = X_t' beta */
  for (int t = 0; t < Dim.num_time; ++t) {
    means[t] = 0.0;
    for (int j = 0; j < Dim.num_regr; ++j)
      means[t] += Param.beta[j] * design_matrix[t][j];
  }

  for (int i = 0; i < Dim.num_param; ++i) gradient[i] = 0.0;

  if (!compute_rho_and_drho()) {
    /* invalid region: return zero gradient (objective caller can penalize) */
    return;
  }

  const int idx_beta0 = (Dim.num_categ - 2);
  const int idx_phi0  = (Dim.num_categ - 2) + Dim.num_regr;

  int count = 0;

  for (int t = 1; t < Dim.num_time; ++t) {
    int hmax = (t < max_lag_L) ? t : max_lag_L;
    for (int h = 1; h <= hmax; ++h) {
      int s = t - h;
      if (IS_MISSING(data[t]) || IS_MISSING(data[s])) continue;

      double P, dP_da1, dP_db1, dP_da2, dP_db2, dP_dr;
      double lo[2], up[2]; int inf[2];
      pair_prob_and_derivs(t, s, h, &P, &dP_da1, &dP_db1, &dP_da2, &dP_db2, &dP_dr,
                           lo, up, inf);

      const double invP = 1.0 / P;

      /* beta: dP/dmu = -(dP/da + dP/db), mu=X beta */
      const double dP_dmu1 = -(dP_da1 + dP_db1);
      const double dP_dmu2 = -(dP_da2 + dP_db2);
      for (int j = 0; j < Dim.num_regr; ++j) {
        gradient[idx_beta0 + j] += invP * (dP_dmu1 * design_matrix[t][j] +
                                           dP_dmu2 * design_matrix[s][j]);
      }

      /* tau contributions (only tau_2..tau_{K-1} are estimated) */
      /* time t (dim1): lower uses tau[y-1] => tau index y-2; upper uses tau[y] => tau index y-1 */
      {
        int y = data[t];
        if (y > 1) {
          int tau_arr_idx = y - 2;
          if (tau_arr_idx >= 1) gradient[tau_arr_idx - 1] += invP * dP_da1;
        }
        if (y < Dim.num_categ) {
          int tau_arr_idx = y - 1;
          if (tau_arr_idx >= 1) gradient[tau_arr_idx - 1] += invP * dP_db1;
        }
      }
      /* time s (dim2) */
      {
        int y = data[s];
        if (y > 1) {
          int tau_arr_idx = y - 2;
          if (tau_arr_idx >= 1) gradient[tau_arr_idx - 1] += invP * dP_da2;
        }
        if (y < Dim.num_categ) {
          int tau_arr_idx = y - 1;
          if (tau_arr_idx >= 1) gradient[tau_arr_idx - 1] += invP * dP_db2;
        }
      }

      /* phi_r: via rho(h) */
      for (int r = 1; r <= ar_order_p; ++r) {
        double drh = drho[(r-1)*(max_lag_L + 1) + h];
        gradient[idx_phi0 + (r-1)] += invP * (dP_dr * drh);
      }

      count++;
    }
  }

  if (count > 0) {
    for (int i = 0; i < Dim.num_param; ++i) gradient[i] /= (double)count;
  }
}

/* ----------------------------- per-time score ------------------------------ */
/* scores is length T * num_param, column-major:
     scores[t + i*T] = score contribution at time t for parameter i
   Here: score at time t sums gradients of log P(t-h, t) for h=1..min(L,t).
*/
void score_by_t_logPair_aop_ARp_pl_arp(const double *param, double *scores)
{
  read_parameters_pl_arp(param);

  const int T = Dim.num_time;
  const int Pn = Dim.num_param;

  for (int i = 0; i < Pn*T; ++i) scores[i] = 0.0;

  /* mu_t */
  for (int t = 0; t < T; ++t) {
    means[t] = 0.0;
    for (int j = 0; j < Dim.num_regr; ++j)
      means[t] += Param.beta[j] * design_matrix[t][j];
  }

  if (!compute_rho_and_drho()) return;

  const int idx_beta0 = (Dim.num_categ - 2);
  const int idx_phi0  = (Dim.num_categ - 2) + Dim.num_regr;

  for (int t = 1; t < T; ++t) {
    int hmax = (t < max_lag_L) ? t : max_lag_L;

    for (int i = 0; i < Pn; ++i) grad_pair[i] = 0.0;
    int cnt = 0;

    for (int h = 1; h <= hmax; ++h) {
      int s = t - h;
      if (IS_MISSING(data[t]) || IS_MISSING(data[s])) continue;

      double P, dP_da1, dP_db1, dP_da2, dP_db2, dP_dr;
      double lo[2], up[2]; int inf[2];
      pair_prob_and_derivs(t, s, h, &P, &dP_da1, &dP_db1, &dP_da2, &dP_db2, &dP_dr,
                           lo, up, inf);

      const double invP = 1.0 / P;

      const double dP_dmu1 = -(dP_da1 + dP_db1);
      const double dP_dmu2 = -(dP_da2 + dP_db2);

      /* beta */
      for (int j = 0; j < Dim.num_regr; ++j) {
        grad_pair[idx_beta0 + j] += invP * (dP_dmu1 * design_matrix[t][j] +
                                            dP_dmu2 * design_matrix[s][j]);
      }

      /* tau: t */
      {
        int y = data[t];
        if (y > 1) {
          int tau_arr_idx = y - 2;
          if (tau_arr_idx >= 1) grad_pair[tau_arr_idx - 1] += invP * dP_da1;
        }
        if (y < Dim.num_categ) {
          int tau_arr_idx = y - 1;
          if (tau_arr_idx >= 1) grad_pair[tau_arr_idx - 1] += invP * dP_db1;
        }
      }
      /* tau: s */
      {
        int y = data[s];
        if (y > 1) {
          int tau_arr_idx = y - 2;
          if (tau_arr_idx >= 1) grad_pair[tau_arr_idx - 1] += invP * dP_da2;
        }
        if (y < Dim.num_categ) {
          int tau_arr_idx = y - 1;
          if (tau_arr_idx >= 1) grad_pair[tau_arr_idx - 1] += invP * dP_db2;
        }
      }

      /* phi */
      for (int r = 1; r <= ar_order_p; ++r) {
        double drh = drho[(r-1)*(max_lag_L + 1) + h];
        grad_pair[idx_phi0 + (r-1)] += invP * (dP_dr * drh);
      }

      cnt++;
    }

    /* average over pairs ending at time t */
    if (cnt > 0) {
      for (int i = 0; i < Pn; ++i)
        // scores[t + i*T] = grad_pair[i] / (double)cnt;
        scores[t + i*T] = grad_pair[i]; /* sum over pairs ending at t */
    }
  }
}
