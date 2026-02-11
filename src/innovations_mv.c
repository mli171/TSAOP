#include <R.h>
#include <Rinternals.h>
#include <R_ext/BLAS.h>
#include <R_ext/Lapack.h>
#include <R_ext/RS.h>
#include <math.h>
#include <string.h>

typedef enum {
  MV_GAMMA = 1,   // Gamma array m x m x L (block Toeplitz, one-sided)
  MV_KBIG  = 2,   // full covariance matrix (T*m) x (T*m)
  MV_KFUN  = 3    // R callback kappa(i,j) -> m x m matrix
} mv_mode_t;

typedef struct {
  mv_mode_t mode;

  // Gamma backend: array dims m x m x L
  const double *Gamma;
  int m, L;

  // Kbig backend: matrix dims (Tm x Tm)
  const double *Kbig;
  int Tm;

  // function backend
  SEXP kappa_fun;
  SEXP eval_env;

  int symmetrize;
} mv_ctx_t;

static inline void mat_symmetrize(double *A, int m) {
  for (int j = 0; j < m; ++j) {
    for (int i = j+1; i < m; ++i) {
      double aij = A[i + m*j];
      double aji = A[j + m*i];
      double s = 0.5 * (aij + aji);
      A[i + m*j] = s;
      A[j + m*i] = s;
    }
  }
}

static inline double maxabs_mat(const double *A, int n) {
  double mx = 0.0;
  for (int i = 0; i < n; ++i) {
    double v = fabs(A[i]);
    if (v > mx) mx = v;
  }
  return mx;
}

// Copy block K(i,j) into out (m x m, column-major).
// For Gamma backend: if i<j return t(Gamma_{j-i}), ensuring global symmetry.
static void getK_block(const mv_ctx_t *ctx, int i, int j, int Tt, double *out)
{
  const int m = ctx->m;

  if (ctx->mode == MV_GAMMA) {
    int d = i - j;
    int h = d >= 0 ? d : -d;
    if (h >= ctx->L) error("Gamma too short: need lag %d but L=%d", h, ctx->L);

    const double *G = ctx->Gamma + (size_t)h * (size_t)m * (size_t)m;

    if (d >= 0) {
      memcpy(out, G, (size_t)m * (size_t)m * sizeof(double));
    } else {
      // transpose into out
      for (int col = 0; col < m; ++col)
        for (int row = 0; row < m; ++row)
          out[row + m*col] = G[col + m*row];
    }

    if (ctx->symmetrize && i == j) mat_symmetrize(out, m);
    return;
  }

  if (ctx->mode == MV_KBIG) {
    // Kbig is (T*m) x (T*m)
    int row0 = (i - 1) * m;
    int col0 = (j - 1) * m;
    int n = ctx->Tm; // = T*m
    for (int col = 0; col < m; ++col) {
      for (int row = 0; row < m; ++row) {
        out[row + m*col] = ctx->Kbig[(row0 + row) + n * (col0 + col)];
      }
    }
    if (ctx->symmetrize && i == j) mat_symmetrize(out, m);
    return;
  }

  // MV_KFUN
  SEXP call = PROTECT(Rf_lang3(ctx->kappa_fun, Rf_ScalarInteger(i), Rf_ScalarInteger(j)));
  SEXP ans  = PROTECT(Rf_eval(call, ctx->eval_env));
  if (!Rf_isMatrix(ans)) error("kappa(i,j) must return a matrix");
  SEXP dim = Rf_getAttrib(ans, R_DimSymbol);
  if (Rf_length(dim) != 2) error("kappa(i,j) must return a 2D matrix");
  int nr = INTEGER(dim)[0], nc = INTEGER(dim)[1];
  if (nr != m || nc != m) error("kappa(i,j) must return an m x m matrix with m=%d", m);

  SEXP A = PROTECT(Rf_coerceVector(ans, REALSXP));
  const double *Ap = REAL(A);
  memcpy(out, Ap, (size_t)m * (size_t)m * sizeof(double));
  UNPROTECT(3);

  if (ctx->symmetrize && i == j) mat_symmetrize(out, m);
}

// Make SPD-ish: symmetrize (optional), add jitter*I, then compute inverse via Cholesky.
// We try increasing jitter if chol fails.
static void make_spd_and_invert(const double *Vin, double *Vout, double *Vinv,
                                int m, int symmetrize, double jitter)
{
  double *base = (double*) R_alloc((size_t)m * (size_t)m, sizeof(double));
  memcpy(base, Vin, (size_t)m * (size_t)m * sizeof(double));

  if (symmetrize) mat_symmetrize(base, m);

  int info = 0;
  double jit = jitter > 0 ? jitter : 0.0;

  for (int attempt = 0; attempt < 6; ++attempt) {
    // Vout = base + jit I
    memcpy(Vout, base, (size_t)m * (size_t)m * sizeof(double));
    for (int d = 0; d < m; ++d) Vout[d + m*d] += jit;

    // Vinv = Vout (then invert in place)
    memcpy(Vinv, Vout, (size_t)m * (size_t)m * sizeof(double));

    char uplo = 'U';
    F77_CALL(dpotrf)(&uplo, &m, Vinv, &m, &info FCONE);
    if (info == 0) {
      F77_CALL(dpotri)(&uplo, &m, Vinv, &m, &info FCONE);
      if (info == 0) {
        // fill lower triangle
        for (int j = 0; j < m; ++j)
          for (int i = j+1; i < m; ++i)
            Vinv[i + m*j] = Vinv[j + m*i];
        return;
      }
    }

    // increase jitter
    if (jit == 0.0) jit = 1e-12;
    else jit *= 10.0;
  }

  error("Cholesky failed even after jitter inflation (last jitter=%g).", jit);
}

static SEXP innovations_mv_core(const mv_ctx_t *ctx_in,
                                SEXP uSEXP,
                                double jitter,
                                int lag_max_is_null, int lag_max,
                                int lag_tol_is_null, double lag_tol,
                                int ahead_is_null, int ahead)
{
  mv_ctx_t ctx = *ctx_in;

  int nprot = 0;

  SEXP uR = PROTECT(Rf_coerceVector(uSEXP, REALSXP)); nprot++;
  SEXP dimU = Rf_getAttrib(uSEXP, R_DimSymbol);
  if (Rf_isNull(dimU) || Rf_length(dimU) != 2) error("u must be a matrix");

  int Tt = INTEGER(dimU)[0];
  int m  = INTEGER(dimU)[1];
  if (Tt < 1 || m < 1) error("u must be T x m with T>=1, m>=1");

  // If Gamma backend, wrapper can prefill ctx.m with Gamma's m
  if (ctx.mode == MV_GAMMA && ctx.m > 0 && ctx.m != m)
    error("Gamma dims m=%d do not match u ncol=%d", ctx.m, m);

  ctx.m = m;

  const double *u = REAL(uR);
  int max_n = Tt - 1;

  // initial cap (memory/compute cap)
  int lag_cap0;
  if (lag_max_is_null) lag_cap0 = max_n;
  else {
    if (lag_max < 1) error("lag_max must be >= 1 or NULL");
    lag_cap0 = (lag_max < max_n) ? lag_max : max_n;
  }

  int do_tol_stop = !lag_tol_is_null;
  if (do_tol_stop) {
    if (!R_FINITE(lag_tol) || lag_tol < 0)
      error("lag_tol must be finite and >= 0, or NULL");
  }

  int compute_ahead = !ahead_is_null;
  if (compute_ahead) {
    if (ahead < 1) error("ahead must be >= 1 or NULL");
  }

  // outputs
  SEXP Theta = PROTECT(Rf_allocVector(VECSXP, Tt)); nprot++;
  SEXP Vlist = PROTECT(Rf_allocVector(VECSXP, Tt)); nprot++;
  SEXP VinvL = PROTECT(Rf_allocVector(VECSXP, Tt)); nprot++;

  // lag_used_path (diagnostic)
  SEXP lag_pathS = PROTECT(Rf_allocVector(INTSXP, Tt)); nprot++;
  int *lag_path = INTEGER(lag_pathS);
  for (int tt = 0; tt < Tt; ++tt) lag_path[tt] = 0;

  // Theta[[1]] = empty list
  SET_VECTOR_ELT(Theta, 0, Rf_allocVector(VECSXP, 0));

  // V[[1]]
  double *K11 = (double*) R_alloc((size_t)m*(size_t)m, sizeof(double));
  getK_block(&ctx, 1, 1, Tt, K11);

  double *V1 = (double*) R_alloc((size_t)m*(size_t)m, sizeof(double));
  double *Vinv1 = (double*) R_alloc((size_t)m*(size_t)m, sizeof(double));
  make_spd_and_invert(K11, V1, Vinv1, m, ctx.symmetrize, jitter);

  SEXP V1S = PROTECT(Rf_allocMatrix(REALSXP, m, m)); nprot++;
  memcpy(REAL(V1S), V1, (size_t)m*(size_t)m*sizeof(double));
  SET_VECTOR_ELT(Vlist, 0, V1S);

  SEXP Vinv1S = PROTECT(Rf_allocMatrix(REALSXP, m, m)); nprot++;
  memcpy(REAL(Vinv1S), Vinv1, (size_t)m*(size_t)m*sizeof(double));
  SET_VECTOR_ELT(VinvL, 0, Vinv1S);

  // adaptive lag length
  int Lcurr = lag_cap0;
  int lag_fixed = 0;
  lag_path[0] = 0;

  // recursion for t = 2..Tt
  if (max_n >= 1 && Lcurr >= 1) {
    for (int n = 1; n <= max_n; ++n) {
      int t = n + 1;               // 1-based time index
      int Lt = (n < Lcurr) ? n : Lcurr;  // how many lags we actually compute/store
      lag_path[t - 1] = Lt;

      SEXP Theta_n = PROTECT(Rf_allocVector(VECSXP, Lt)); nprot++;

      int k_start = n - Lt; // keep k in {n-Lt, ..., n-1}

      // compute Theta_n[[ell]] for ell = n-k, with k increasing => ell decreasing
      for (int k = k_start; k <= (n - 1); ++k) {
        int ell = n - k; // lag index in 1..Lt

        double *term = (double*) R_alloc((size_t)m*(size_t)m, sizeof(double));
        getK_block(&ctx, t, k + 1, Tt, term);

        if (k >= 1) {
          // Lk = number of stored lags at time (k+1)
          SEXP Theta_k1 = VECTOR_ELT(Theta, k); // time k+1
          int Lk = Rf_length(Theta_k1);

          int j0 = 0;
          int a = n - Lt;      if (a > j0) j0 = a;
          int b = k - Lk;      if (b > j0) j0 = b;

          if (j0 <= (k - 1)) {
            for (int j = j0; j <= (k - 1); ++j) {
              int ell_tj = n - j; // 1..Lt
              int ell_kj = k - j; // 1..Lk

              // A_n = Theta_n[[ell_tj]]
              SEXP A_nS = VECTOR_ELT(Theta_n, ell_tj - 1);
              const double *A_n = REAL(A_nS);

              // A_k = Theta[[k+1]][[ell_kj]]
              SEXP A_kS = VECTOR_ELT(Theta_k1, ell_kj - 1);
              const double *A_k = REAL(A_kS);

              const double *Vj = REAL(VECTOR_ELT(Vlist, j));

              double *tmp = (double*) R_alloc((size_t)m*(size_t)m, sizeof(double));
              double *sub = (double*) R_alloc((size_t)m*(size_t)m, sizeof(double));
              char N = 'N', TT = 'T';
              double one = 1.0, zero = 0.0;

              // tmp = Vj %*% t(A_k)
              F77_CALL(dgemm)(&N, &TT, &m, &m, &m, &one,
                       (double*)Vj, &m,
                       (double*)A_k, &m,
                       &zero, tmp, &m FCONE FCONE);

              // sub = A_n %*% tmp
              F77_CALL(dgemm)(&N, &N, &m, &m, &m, &one,
                       (double*)A_n, &m,
                       tmp, &m,
                       &zero, sub, &m FCONE FCONE);

              for (int q = 0; q < m*m; ++q) term[q] -= sub[q];
            }
          }
        }

        // Theta_n[[ell]] = term %*% Vinv[[k+1]]
        const double *Vinvk = REAL(VECTOR_ELT(VinvL, k)); // time k+1
        double *Aout = (double*) R_alloc((size_t)m*(size_t)m, sizeof(double));
        char N = 'N';
        double one = 1.0, zero = 0.0;
        F77_CALL(dgemm)(&N, &N, &m, &m, &m, &one,
                 term, &m,
                 (double*)Vinvk, &m,
                 &zero, Aout, &m FCONE FCONE);

        SEXP AoutS = PROTECT(Rf_allocMatrix(REALSXP, m, m)); nprot++;
        memcpy(REAL(AoutS), Aout, (size_t)m*(size_t)m*sizeof(double));
        SET_VECTOR_ELT(Theta_n, ell - 1, AoutS);
        // keep AoutS protected until end of this loop iteration
      }

      // Vn = K(t,t) - sum_{j=max(0,n-Lt)}^{n-1} A V_j t(A)
      double *Vraw = (double*) R_alloc((size_t)m*(size_t)m, sizeof(double));
      getK_block(&ctx, t, t, Tt, Vraw);

      int j0v = n - Lt;
      if (j0v < 0) j0v = 0;

      for (int j = j0v; j <= (n - 1); ++j) {
        int ell = n - j; // 1..Lt
        SEXP AS = VECTOR_ELT(Theta_n, ell - 1);
        const double *A = REAL(AS);
        const double *Vj = REAL(VECTOR_ELT(Vlist, j));

        double *tmp = (double*) R_alloc((size_t)m*(size_t)m, sizeof(double));
        double *sub = (double*) R_alloc((size_t)m*(size_t)m, sizeof(double));
        char N = 'N', TT = 'T';
        double one = 1.0, zero = 0.0;

        // tmp = Vj %*% t(A)
        F77_CALL(dgemm)(&N, &TT, &m, &m, &m, &one,
                 (double*)Vj, &m,
                 (double*)A, &m,
                 &zero, tmp, &m FCONE FCONE);

        // sub = A %*% tmp
        F77_CALL(dgemm)(&N, &N, &m, &m, &m, &one,
                 (double*)A, &m,
                 tmp, &m,
                 &zero, sub, &m FCONE FCONE);

        for (int q = 0; q < m*m; ++q) Vraw[q] -= sub[q];
      }

      double *Vn    = (double*) R_alloc((size_t)m*(size_t)m, sizeof(double));
      double *Vinvn = (double*) R_alloc((size_t)m*(size_t)m, sizeof(double));
      make_spd_and_invert(Vraw, Vn, Vinvn, m, ctx.symmetrize, jitter);

      SET_VECTOR_ELT(Theta, t - 1, Theta_n);

      SEXP VnS = PROTECT(Rf_allocMatrix(REALSXP, m, m)); nprot++;
      memcpy(REAL(VnS), Vn, (size_t)m*(size_t)m*sizeof(double));
      SET_VECTOR_ELT(Vlist, t - 1, VnS);

      SEXP VinvS = PROTECT(Rf_allocMatrix(REALSXP, m, m)); nprot++;
      memcpy(REAL(VinvS), Vinvn, (size_t)m*(size_t)m*sizeof(double));
      SET_VECTOR_ELT(VinvL, t - 1, VinvS);

      // tol rule: once the boundary lag is small, fix Lcurr forever
      if (!lag_fixed && do_tol_stop && Lt >= 1) {
        SEXP Abound = VECTOR_ELT(Theta_n, Lt - 1); // Theta_n[[Lt]]
        if (maxabs_mat(REAL(Abound), m*m) < lag_tol) {
          Lcurr = Lt;
          lag_fixed = 1;
        }
      }

      // We PROTECTed: Theta_n, plus Lt many AoutS, plus VnS, VinvS.
      // Unprotect in reverse count: (1 Theta_n) + (Lt AoutS) + 2
      UNPROTECT(1 + Lt + 2);
      nprot -= (1 + Lt + 2);
    }
  }

  // innovations and uhat
  SEXP uhatS  = PROTECT(Rf_allocMatrix(REALSXP, Tt, m)); nprot++;
  SEXP innovS = PROTECT(Rf_allocMatrix(REALSXP, Tt, m)); nprot++;
  double *uhat  = REAL(uhatS);
  double *innov = REAL(innovS);

  for (int col = 0; col < m; ++col) {
    uhat[0 + Tt*col]  = 0.0;
    innov[0 + Tt*col] = u[0 + Tt*col];
  }

  double *e_prev = (double*) R_alloc((size_t)m, sizeof(double));
  double *tmpv   = (double*) R_alloc((size_t)m, sizeof(double));

  for (int t = 2; t <= Tt; ++t) {
    for (int col = 0; col < m; ++col) tmpv[col] = 0.0;

    SEXP coefs = VECTOR_ELT(Theta, t - 1);
    int len = Rf_length(coefs);
    int jmax = (len < (t - 1)) ? len : (t - 1);

    for (int j = 1; j <= jmax; ++j) {
      int row = (t - j) - 1;
      for (int col = 0; col < m; ++col)
        e_prev[col] = innov[row + Tt*col];

      const double *A = REAL(VECTOR_ELT(coefs, j - 1));

      char trans = 'N';
      double alpha = 1.0, beta = 1.0;
      int inc1 = 1;
      F77_CALL(dgemv)(&trans, &m, &m, &alpha,
               (double*)A, &m,
               e_prev, &inc1,
               &beta,
               tmpv, &inc1 FCONE);
    }

    int rowt = t - 1;
    for (int col = 0; col < m; ++col) {
      uhat[rowt + Tt*col]  = tmpv[col];
      innov[rowt + Tt*col] = u[rowt + Tt*col] - tmpv[col];
    }
  }

  // ahead prediction
  SEXP uhat_aheadS = R_NilValue;
  if (!ahead_is_null) {
    if (ahead == 1) {
      uhat_aheadS = uhatS;
    } else {
      uhat_aheadS = PROTECT(Rf_allocMatrix(REALSXP, Tt, m)); nprot++;
      double *uha = REAL(uhat_aheadS);
      for (int i = 0; i < Tt*m; ++i) uha[i] = 0.0;

      for (int t = 1; t <= Tt; ++t) {
        if (t <= ahead) continue;

        int origin = t - ahead; // 1-based
        SEXP coefs = VECTOR_ELT(Theta, origin); // matches your R version
        int len = Rf_length(coefs);
        int jmax = (len < (t - 1)) ? len : (t - 1);

        for (int col = 0; col < m; ++col) tmpv[col] = 0.0;

        if (jmax >= ahead) {
          for (int j = ahead; j <= jmax; ++j) {
            int row = (t - j) - 1;
            for (int col = 0; col < m; ++col)
              e_prev[col] = innov[row + Tt*col];

            const double *A = REAL(VECTOR_ELT(coefs, j - 1));
            char trans = 'N';
            double alpha = 1.0, beta = 1.0;
            int inc1 = 1;
            F77_CALL(dgemv)(&trans, &m, &m, &alpha,
                     (double*)A, &m,
                     e_prev, &inc1,
                     &beta,
                     tmpv, &inc1 FCONE);
          }
        }

        int rowt = t - 1;
        for (int col = 0; col < m; ++col)
          uha[rowt + Tt*col] = tmpv[col];
      }
    }
  }

  // output list: add lag_max_used + lag_used_path (and keep mylag alias)
  int lag_max_used = Lcurr;

  SEXP out = PROTECT(Rf_allocVector(VECSXP, 8)); nprot++;
  SEXP nm  = PROTECT(Rf_allocVector(STRSXP, 8)); nprot++;

  SET_STRING_ELT(nm, 0, Rf_mkChar("Theta"));
  SET_STRING_ELT(nm, 1, Rf_mkChar("V"));
  SET_STRING_ELT(nm, 2, Rf_mkChar("uhat"));
  SET_STRING_ELT(nm, 3, Rf_mkChar("innov"));
  SET_STRING_ELT(nm, 4, Rf_mkChar("uhat_ahead"));
  SET_STRING_ELT(nm, 5, Rf_mkChar("lag_max_used"));
  SET_STRING_ELT(nm, 6, Rf_mkChar("lag_used_path"));
  SET_STRING_ELT(nm, 7, Rf_mkChar("mylag")); // alias

  Rf_setAttrib(out, R_NamesSymbol, nm);

  SET_VECTOR_ELT(out, 0, Theta);
  SET_VECTOR_ELT(out, 1, Vlist);
  SET_VECTOR_ELT(out, 2, uhatS);
  SET_VECTOR_ELT(out, 3, innovS);
  SET_VECTOR_ELT(out, 4, uhat_aheadS);
  SET_VECTOR_ELT(out, 5, Rf_ScalarInteger(lag_max_used));
  SET_VECTOR_ELT(out, 6, lag_pathS);
  SET_VECTOR_ELT(out, 7, Rf_ScalarInteger(lag_max_used));

  UNPROTECT(nprot);
  return out;
}


// ---- .Call entry points ----

SEXP innovations_mv_gamma(SEXP GammaSEXP, SEXP uSEXP,
                          SEXP jitterSEXP, SEXP symmetrizeSEXP,
                          SEXP lag_maxSEXP, SEXP lag_tolSEXP, SEXP aheadSEXP)
{
  mv_ctx_t ctx;
  ctx.mode = MV_GAMMA;
  ctx.kappa_fun = R_NilValue;
  ctx.eval_env  = R_GlobalEnv;

  ctx.symmetrize = asLogical(symmetrizeSEXP);

  // Gamma: 3D array m x m x L
  SEXP dimG = Rf_getAttrib(GammaSEXP, R_DimSymbol);
  if (Rf_isNull(dimG) || Rf_length(dimG) != 3) error("Gamma must be a 3D array (m x m x L)");
  int m1 = INTEGER(dimG)[0], m2 = INTEGER(dimG)[1], L = INTEGER(dimG)[2];
  if (m1 != m2) error("Gamma must have first two dims equal (m x m x L)");

  SEXP GR = PROTECT(Rf_coerceVector(GammaSEXP, REALSXP));
  ctx.Gamma = REAL(GR);
  ctx.m = m1;   // store Gamma's m for consistency check against u
  ctx.L = L;

  double jitter = Rf_asReal(jitterSEXP);

  int lag_max_is_null = Rf_isNull(lag_maxSEXP);
  int lag_max = lag_max_is_null ? 0 : Rf_asInteger(lag_maxSEXP);

  int lag_tol_is_null = Rf_isNull(lag_tolSEXP);
  double lag_tol = lag_tol_is_null ? 0.0 : Rf_asReal(lag_tolSEXP);

  int ahead_is_null = Rf_isNull(aheadSEXP);
  int ahead = ahead_is_null ? 1 : Rf_asInteger(aheadSEXP);

  SEXP ans = innovations_mv_core(&ctx, uSEXP, jitter,
                                 lag_max_is_null, lag_max,
                                 lag_tol_is_null, lag_tol,
                                 ahead_is_null, ahead);

  UNPROTECT(1); // GR
  return ans;
}

SEXP innovations_mv_Kbig(SEXP KbigSEXP, SEXP uSEXP,
                         SEXP jitterSEXP, SEXP symmetrizeSEXP,
                         SEXP lag_maxSEXP, SEXP lag_tolSEXP, SEXP aheadSEXP)
{
  mv_ctx_t ctx;
  ctx.mode = MV_KBIG;
  ctx.kappa_fun = R_NilValue;
  ctx.eval_env  = R_GlobalEnv;
  ctx.symmetrize = asLogical(symmetrizeSEXP);

  SEXP KR = PROTECT(Rf_coerceVector(KbigSEXP, REALSXP));
  SEXP dimK = Rf_getAttrib(KbigSEXP, R_DimSymbol);
  if (Rf_isNull(dimK) || Rf_length(dimK) != 2) error("Kbig must be a matrix");
  int nr = INTEGER(dimK)[0], nc = INTEGER(dimK)[1];
  if (nr != nc) error("Kbig must be square");
  ctx.Kbig = REAL(KR);
  ctx.Tm = nr;

  double jitter = Rf_asReal(jitterSEXP);

  int lag_max_is_null = Rf_isNull(lag_maxSEXP);
  int lag_max = lag_max_is_null ? 0 : Rf_asInteger(lag_maxSEXP);

  int lag_tol_is_null = Rf_isNull(lag_tolSEXP);
  double lag_tol = lag_tol_is_null ? 0.0 : Rf_asReal(lag_tolSEXP);

  int ahead_is_null = Rf_isNull(aheadSEXP);
  int ahead = ahead_is_null ? 1 : Rf_asInteger(aheadSEXP);

  SEXP ans = innovations_mv_core(&ctx, uSEXP, jitter,
                                 lag_max_is_null, lag_max,
                                 lag_tol_is_null, lag_tol,
                                 ahead_is_null, ahead);

  UNPROTECT(1);
  return ans;
}

SEXP innovations_mv_kappa(SEXP kappaFunSEXP, SEXP uSEXP,
                          SEXP jitterSEXP, SEXP symmetrizeSEXP,
                          SEXP lag_maxSEXP, SEXP lag_tolSEXP, SEXP aheadSEXP)
{
  if (!Rf_isFunction(kappaFunSEXP)) error("kappa must be a function");

  mv_ctx_t ctx;
  ctx.mode = MV_KFUN;
  ctx.kappa_fun = kappaFunSEXP;
  ctx.eval_env  = R_GlobalEnv;
  ctx.symmetrize = asLogical(symmetrizeSEXP);
  ctx.Gamma = NULL;
  ctx.Kbig  = NULL;
  ctx.L = 0;
  ctx.Tm = 0;

  double jitter = Rf_asReal(jitterSEXP);

  int lag_max_is_null = Rf_isNull(lag_maxSEXP);
  int lag_max = lag_max_is_null ? 0 : Rf_asInteger(lag_maxSEXP);

  int lag_tol_is_null = Rf_isNull(lag_tolSEXP);
  double lag_tol = lag_tol_is_null ? 0.0 : Rf_asReal(lag_tolSEXP);

  int ahead_is_null = Rf_isNull(aheadSEXP);
  int ahead = ahead_is_null ? 1 : Rf_asInteger(aheadSEXP);

  return innovations_mv_core(&ctx, uSEXP, jitter,
                             lag_max_is_null, lag_max,
                             lag_tol_is_null, lag_tol,
                             ahead_is_null, ahead);
}
