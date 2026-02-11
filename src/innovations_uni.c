#include <R.h>
#include <Rinternals.h>
#include <R_ext/Rdynload.h>
#include <math.h>

typedef enum {
  KAPPA_GAMMA = 1,   // gamma[|i-j|]
  KAPPA_MAT   = 2,   // K[i,j]
  KAPPA_FUN   = 3    // R function kappa(i,j)
} kappa_mode_t;

typedef struct {
  kappa_mode_t mode;

  // gamma backend
  const double *gamma;
  int Lgamma;

  // matrix backend
  const double *K;
  int nrowK, ncolK;

  // function backend
  SEXP kappa_fun;
  SEXP eval_env;
} kappa_ctx_t;

static inline int iabs_int(int x) { return x < 0 ? -x : x; }

static inline double getK(const kappa_ctx_t *ctx, int i, int j)
{
  // i,j are 1-based
  if (ctx->mode == KAPPA_GAMMA) {
    int h = iabs_int(i - j);
    if (h >= ctx->Lgamma) error("gamma too short: need gamma[%d] but length is %d", h, ctx->Lgamma);
    double val = ctx->gamma[h];
    if (!R_FINITE(val)) error("gamma[%d] is not finite", h);
    return val;
  }

  if (ctx->mode == KAPPA_MAT) {
    if (i < 1 || i > ctx->nrowK || j < 1 || j > ctx->ncolK)
      error("K index out of bounds: (%d,%d) with dim (%d,%d)", i, j, ctx->nrowK, ctx->ncolK);
    // column-major
    double val = ctx->K[(i - 1) + ctx->nrowK * (j - 1)];
    if (!R_FINITE(val)) error("K(%d,%d) is not finite", i, j);
    return val;
  }

  // ctx->mode == KAPPA_FUN
  SEXP call = PROTECT(Rf_lang3(ctx->kappa_fun, Rf_ScalarInteger(i), Rf_ScalarInteger(j)));
  SEXP ans  = PROTECT(Rf_eval(call, ctx->eval_env));
  double val = Rf_asReal(ans);
  UNPROTECT(2);
  if (!R_FINITE(val)) error("kappa(%d,%d) returned non-finite", i, j);
  return val;
}

static SEXP innovations_uni_core(const kappa_ctx_t *ctx,
                                 SEXP uSEXP,
                                 double jitter,
                                 int lag_max_is_null, int lag_max,
                                 int lag_tol_is_null, double lag_tol,
                                 int ahead_is_null, int ahead)
{
  int nprot = 0;

  // coerce u to REAL
  SEXP uR = PROTECT(Rf_coerceVector(uSEXP, REALSXP)); nprot++;
  int Tt  = Rf_length(uR);
  if (Tt < 1) error("u must have length >= 1");
  const double *u = REAL(uR);

  int max_n = Tt - 1;

  // initial cap (memory/compute cap)
  int lag_cap0;
  if (lag_max_is_null) {
    lag_cap0 = max_n;
  } else {
    if (lag_max < 1) error("lag_max must be >= 1 or NULL");
    lag_cap0 = (lag_max < max_n) ? lag_max : max_n;
  }
  if (lag_cap0 < 0) lag_cap0 = 0;

  // tol stop
  int do_tol_stop = !lag_tol_is_null;
  if (do_tol_stop) {
    if (!R_FINITE(lag_tol) || lag_tol < 0)
      error("lag_tol must be finite and >= 0, or NULL");
  }

  // ahead
  int compute_ahead = !ahead_is_null;
  if (compute_ahead) {
    if (ahead < 1) error("ahead must be >= 1 or NULL");
  }

  // outputs
  SEXP theta = PROTECT(Rf_allocVector(VECSXP, Tt)); nprot++;
  SEXP v     = PROTECT(Rf_allocVector(REALSXP, Tt)); nprot++;
  double *vR = REAL(v);

  // diagnostics
  SEXP lag_pathS = PROTECT(Rf_allocVector(INTSXP, Tt)); nprot++;
  int *lag_path = INTEGER(lag_pathS);
  for (int tt = 0; tt < Tt; ++tt) lag_path[tt] = 0;

  // t = 1
  SET_VECTOR_ELT(theta, 0, Rf_allocVector(REALSXP, 0));
  double v1 = getK(ctx, 1, 1);
  if (v1 < 0) v1 = 0;
  vR[0] = v1 + jitter;

  // adaptive lag length
  int Lcurr = lag_cap0;
  int lag_fixed = 0;
  lag_path[0] = 0;

  // build theta, v for ALL times t=2..Tt, but only store Lt lags each time
  if (max_n >= 1 && Lcurr >= 1) {
    for (int n = 1; n <= max_n; ++n) {     // n = t-1
      int t = n + 1;                       // 1-based time

      int Lt = (n < Lcurr) ? n : Lcurr;    // number of lags stored at time t
      lag_path[t - 1] = Lt;

      SEXP thR = PROTECT(Rf_allocVector(REALSXP, Lt)); nprot++;
      double *th = REAL(thR);              // th[0..Lt-1] = lags 1..Lt

      // compute only k in {n-Lt, ..., n-1}
      int k_start = n - Lt;

      for (int k = k_start; k <= (n - 1); ++k) {
        int ell = n - k;                   // lag index in 1..Lt
        double term = getK(ctx, t, k + 1);

        if (k >= 1) {
          // theta[[k+1]] has length Lk
          SEXP theta_k1 = VECTOR_ELT(theta, k);     // time k+1
          const double *th_k1 = REAL(theta_k1);
          int Lk = Rf_length(theta_k1);

          int j0 = 0;
          int a = n - Lt; if (a > j0) j0 = a;       // ensure ell_tj = n-j <= Lt
          int b = k - Lk; if (b > j0) j0 = b;       // ensure ell_kj = k-j <= Lk

          if (j0 <= (k - 1)) {
            for (int j = j0; j <= (k - 1); ++j) {
              int ell_tj = n - j;                   // 1..Lt
              int ell_kj = k - j;                   // 1..Lk
              term -= th[ell_tj - 1] * vR[j] * th_k1[ell_kj - 1];
            }
          }
        }

        // th[ell] = term / v[k+1] (vR[k] is v[k+1] in 0-based)
        th[ell - 1] = term / vR[k];
      }

      // update v[t] using only retained lags (j in {n-Lt,...,n-1})
      double vt = getK(ctx, t, t);
      int j0v = n - Lt;
      if (j0v < 0) j0v = 0;

      for (int j = j0v; j <= (n - 1); ++j) {
        int ell = n - j;                   // 1..Lt
        double a = th[ell - 1];
        vt -= a * a * vR[j];
      }
      if (vt < 0) vt = 0;
      vR[t - 1] = vt + jitter;

      // store theta[[t]] = thR
      SET_VECTOR_ELT(theta, t - 1, thR);

      // tol rule: once boundary coef is small, fix lag length forever
      if (!lag_fixed && do_tol_stop && Lt >= 1) {
        if (fabs(th[Lt - 1]) < lag_tol) {
          Lcurr = Lt;
          lag_fixed = 1;
        }
      }

      UNPROTECT(1); nprot--;  // thR
    }
  } else {
    // if no lags allowed, fill theta[[t]] with empty and v[t]=K(t,t)+jitter
    for (int t = 2; t <= Tt; ++t) {
      SET_VECTOR_ELT(theta, t - 1, Rf_allocVector(REALSXP, 0));
      double vt = getK(ctx, t, t);
      if (vt < 0) vt = 0;
      vR[t - 1] = vt + jitter;
      lag_path[t - 1] = 0;
    }
  }

  int lag_max_used = Lcurr;   // what you actually used after fixing

  // innovations and 1-step uhat
  SEXP uhat  = PROTECT(Rf_allocVector(REALSXP, Tt)); nprot++;
  SEXP innov = PROTECT(Rf_allocVector(REALSXP, Tt)); nprot++;
  double *uhR = REAL(uhat);
  double *inR = REAL(innov);

  uhR[0] = 0.0;
  inR[0] = u[0];

  for (int t = 2; t <= Tt; ++t) {
    SEXP thS = VECTOR_ELT(theta, t - 1);
    const double *th = REAL(thS);
    int len = Rf_length(thS);
    int jmax = (len < (t - 1)) ? len : (t - 1);

    double pred = 0.0;
    for (int j = 1; j <= jmax; ++j) {
      pred += th[j - 1] * inR[(t - 1) - j];
    }
    uhR[t - 1] = pred;
    inR[t - 1] = u[t - 1] - pred;
  }

  // h-step-ahead conditional mean
  SEXP uhat_ahead = R_NilValue;
  if (compute_ahead) {
    if (ahead == 1) {
      uhat_ahead = uhat;
    } else {
      uhat_ahead = PROTECT(Rf_allocVector(REALSXP, Tt)); nprot++;
      double *uha = REAL(uhat_ahead);

      for (int t = 1; t <= Tt; ++t) {
        if (t <= ahead) {
          uha[t - 1] = 0.0;
        } else {
          int origin = t - ahead;                 // 1-based
          SEXP thS = VECTOR_ELT(theta, origin);   // theta[[origin+1]] in R
          const double *th = REAL(thS);
          int len = Rf_length(thS);
          int jmax = (len < (t - 1)) ? len : (t - 1);

          double pred = 0.0;
          if (jmax >= ahead) {
            for (int j = ahead; j <= jmax; ++j) {
              pred += th[j - 1] * inR[(t - 1) - j];
            }
          }
          uha[t - 1] = pred;
        }
      }
    }
  }

  // assemble return list (expanded)
  SEXP out = PROTECT(Rf_allocVector(VECSXP, 8)); nprot++;
  SEXP nm  = PROTECT(Rf_allocVector(STRSXP, 8)); nprot++;

  SET_STRING_ELT(nm, 0, Rf_mkChar("theta"));
  SET_STRING_ELT(nm, 1, Rf_mkChar("v"));
  SET_STRING_ELT(nm, 2, Rf_mkChar("uhat"));
  SET_STRING_ELT(nm, 3, Rf_mkChar("innov"));
  SET_STRING_ELT(nm, 4, Rf_mkChar("uhat_ahead"));
  SET_STRING_ELT(nm, 5, Rf_mkChar("lag_max_used"));
  SET_STRING_ELT(nm, 6, Rf_mkChar("lag_used_path"));
  SET_STRING_ELT(nm, 7, Rf_mkChar("mylag")); // alias

  Rf_setAttrib(out, R_NamesSymbol, nm);

  SET_VECTOR_ELT(out, 0, theta);
  SET_VECTOR_ELT(out, 1, v);
  SET_VECTOR_ELT(out, 2, uhat);
  SET_VECTOR_ELT(out, 3, innov);
  SET_VECTOR_ELT(out, 4, uhat_ahead);
  SET_VECTOR_ELT(out, 5, Rf_ScalarInteger(lag_max_used));
  SET_VECTOR_ELT(out, 6, lag_pathS);
  SET_VECTOR_ELT(out, 7, Rf_ScalarInteger(lag_max_used));

  UNPROTECT(nprot);
  return out;
}


// ---- .Call entry points ----

// gamma: numeric vector gamma[0..] such that Cov(i,j)=gamma[|i-j|]
SEXP innovations_uni_gamma(SEXP gammaSEXP, SEXP uSEXP,
                           SEXP jitterSEXP, SEXP lag_maxSEXP, SEXP lag_tolSEXP, SEXP aheadSEXP)
{
  SEXP gR = PROTECT(Rf_coerceVector(gammaSEXP, REALSXP));
  kappa_ctx_t ctx;
  ctx.mode   = KAPPA_GAMMA;
  ctx.gamma  = REAL(gR);
  ctx.Lgamma  = Rf_length(gR);
  ctx.K      = NULL;
  ctx.nrowK  = 0; ctx.ncolK = 0;
  ctx.kappa_fun = R_NilValue;
  ctx.eval_env  = R_GlobalEnv;

  double jitter = Rf_asReal(jitterSEXP);

  int lag_max_is_null = Rf_isNull(lag_maxSEXP);
  int lag_max = lag_max_is_null ? 0 : Rf_asInteger(lag_maxSEXP);

  int lag_tol_is_null = Rf_isNull(lag_tolSEXP);
  double lag_tol = lag_tol_is_null ? 0.0 : Rf_asReal(lag_tolSEXP);

  int ahead_is_null = Rf_isNull(aheadSEXP);
  int ahead = ahead_is_null ? 1 : Rf_asInteger(aheadSEXP);

  SEXP ans = innovations_uni_core(&ctx, uSEXP, jitter,
                                  lag_max_is_null, lag_max,
                                  lag_tol_is_null, lag_tol,
                                  ahead_is_null, ahead);

  UNPROTECT(1); // gR
  return ans;
}

// K: numeric matrix (T x T)
SEXP innovations_uni_K(SEXP KSEXP, SEXP uSEXP,
                       SEXP jitterSEXP, SEXP lag_maxSEXP, SEXP lag_tolSEXP, SEXP aheadSEXP)
{
  if (!Rf_isReal(KSEXP) && !Rf_isInteger(KSEXP))
    error("K must be a numeric matrix");

  SEXP KR = PROTECT(Rf_coerceVector(KSEXP, REALSXP));

  SEXP dim = Rf_getAttrib(KSEXP, R_DimSymbol);
  if (Rf_isNull(dim) || Rf_length(dim) != 2) error("K must be a matrix");

  int nrow = INTEGER(dim)[0];
  int ncol = INTEGER(dim)[1];
  if (nrow != ncol) error("K must be square");

  kappa_ctx_t ctx;
  ctx.mode   = KAPPA_MAT;
  ctx.K      = REAL(KR);
  ctx.nrowK  = nrow;
  ctx.ncolK  = ncol;
  ctx.gamma  = NULL;
  ctx.Lgamma  = 0;
  ctx.kappa_fun = R_NilValue;
  ctx.eval_env  = R_GlobalEnv;

  double jitter = Rf_asReal(jitterSEXP);

  int lag_max_is_null = Rf_isNull(lag_maxSEXP);
  int lag_max = lag_max_is_null ? 0 : Rf_asInteger(lag_maxSEXP);

  int lag_tol_is_null = Rf_isNull(lag_tolSEXP);
  double lag_tol = lag_tol_is_null ? 0.0 : Rf_asReal(lag_tolSEXP);

  int ahead_is_null = Rf_isNull(aheadSEXP);
  int ahead = ahead_is_null ? 1 : Rf_asInteger(aheadSEXP);

  SEXP ans = innovations_uni_core(&ctx, uSEXP, jitter,
                                  lag_max_is_null, lag_max,
                                  lag_tol_is_null, lag_tol,
                                  ahead_is_null, ahead);

  UNPROTECT(1); // KR
  return ans;
}

// kappa: R function kappa(i,j)
SEXP innovations_uni_kappa(SEXP kappaFunSEXP, SEXP uSEXP,
                           SEXP jitterSEXP, SEXP lag_maxSEXP, SEXP lag_tolSEXP, SEXP aheadSEXP)
{
  if (!Rf_isFunction(kappaFunSEXP)) error("kappa must be a function");

  kappa_ctx_t ctx;
  ctx.mode   = KAPPA_FUN;
  ctx.kappa_fun = kappaFunSEXP;
  ctx.eval_env  = R_GlobalEnv;
  ctx.gamma  = NULL; ctx.Lgamma = 0;
  ctx.K      = NULL; ctx.nrowK = 0; ctx.ncolK = 0;

  double jitter = Rf_asReal(jitterSEXP);

  int lag_max_is_null = Rf_isNull(lag_maxSEXP);
  int lag_max = lag_max_is_null ? 0 : Rf_asInteger(lag_maxSEXP);

  int lag_tol_is_null = Rf_isNull(lag_tolSEXP);
  double lag_tol = lag_tol_is_null ? 0.0 : Rf_asReal(lag_tolSEXP);

  int ahead_is_null = Rf_isNull(aheadSEXP);
  int ahead = ahead_is_null ? 1 : Rf_asInteger(aheadSEXP);

  return innovations_uni_core(&ctx, uSEXP, jitter,
                              lag_max_is_null, lag_max,
                              lag_tol_is_null, lag_tol,
                              ahead_is_null, ahead);
}

// ---- registration ----

static const R_CallMethodDef CallEntries[] = {
  {"innovations_uni_gamma", (DL_FUNC) &innovations_uni_gamma, 6},
  {"innovations_uni_K",     (DL_FUNC) &innovations_uni_K,     6},
  {"innovations_uni_kappa", (DL_FUNC) &innovations_uni_kappa, 6},
  {NULL, NULL, 0}
};

void R_init_yourpkg(DllInfo *dll)
{
  R_registerRoutines(dll, NULL, CallEntries, NULL, NULL);
  R_useDynamicSymbols(dll, FALSE);
}
