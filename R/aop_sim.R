#' Simulate ordinal time series under AR(p) probit-threshold model
#'
#' @param ci Numeric vector of cutpoints. Either:
#' \itemize{
#'   \item \code{length(ci) == K-2}: interior cutpoints, using the convention \eqn{c_1 = 0}.
#'   \item \code{length(ci) == K-1}: full, strictly increasing cutpoints \eqn{c_1 < \cdots < c_{K-1}}.
#' }
#' @param theta numeric, length ncol(DesignX); regression coefficients
#' @param rho Numeric. Either:
#' \itemize{
#'   \item scalar: AR(1) coefficient in (-1, 1)
#'   \item vector: AR(p) coefficients \eqn{(\phi_1,\ldots,\phi_p)}
#' }
#' @param K number of categories
#' @param Ts length of series
#' @param DesignX T x p design matrix (include intercept if desired)
#' @param seed optional integer for reproducibility
#' @param burnin integer burn-in length for AR(p) initialization (default chosen from p)
#' @param psi_trunc truncation length for MA(\eqn{\infty}) weights used to scale innovations
#' @param tol tolerance for truncation / numerical safety
#' @return list with latent Z, integer y in 1..K, and one-hot matrix Yw (T x K)
#'
#' @examples
#' ## AR(2) simulation + fit CLSEW and PL (requires compiled native code)
#' set.seed(1)
#' T  <- 400
#' K  <- 5
#' t  <- 1:T
#' X <- cbind(Intercept = 1, Trend = as.numeric(scale(t, scale = FALSE)))
#'
#' cut_true   <- c(0.5, 1.2, 2.0)     # K-2 interior cutpoints with c1=0 convention
#' theta_true <- c(-0.4, 0.002)
#' phi_true   <- c(0.6, -0.15)        # AR(2)
#'
#' sim <- aop_sim(ci = cut_true, theta = theta_true, rho = phi_true,
#'                K = K, Ts = T, DesignX = X, seed = 123)
#'
#' @export
aop_sim <- function(ci, theta, rho, K, Ts, DesignX,
                    seed = NULL,
                    burnin = NULL,
                    psi_trunc = 5000L,
                    tol = 1e-10) {
  stopifnot(is.matrix(DesignX), nrow(DesignX) == Ts, ncol(DesignX) == length(theta))
  if (!is.null(seed)) set.seed(seed)

  ## Build ordered cutpoints
  cut <- if (length(ci) == K - 1L) {
    sort(as.numeric(ci))
  } else if (length(ci) == K - 2L) {
    sort(c(0, as.numeric(ci)))  # c1 = 0 convention for simulation only
  } else {
    stop("ci must have length K-1 (full cutpoints) or K-2 (c1 = 0 convention).")
  }
  if (any(!is.finite(cut))) stop("cutpoints contain non-finite values.")
  if (any(diff(cut) <= 0))  stop("cutpoints must be strictly increasing.")

  ## Latent mean
  mst <- as.vector(DesignX %*% theta)

  ## AR(p) disturbance u_t with Var(u_t)=1
  phi <- as.numeric(rho)
  if (length(phi) < 1L) stop("rho must be a scalar (AR(1)) or a numeric vector (AR(p)).")
  p_order <- length(phi)

  ## Stationarity check via roots of 1 - phi1 z - ... - phip z^p
  roots <- polyroot(c(1, -phi))
  if (any(Mod(roots) <= 1 + 1e-8)) stop("AR(p) coefficients are not stationary (some roots within unit circle).")

  if (is.null(burnin)) burnin <- max(2000L, 200L * p_order)
  burnin <- as.integer(burnin)
  psi_trunc <- as.integer(psi_trunc)

  ## Innovation sd chosen so stationary Var(u)=1
  if (p_order == 1L) {
    if (abs(phi) >= 1) stop("For AR(1), rho must be in (-1, 1).")
    sd_eps <- sqrt(1 - phi^2)
  } else {
    ## MA(infty) weights psi_j for AR(p):
    ##    psi_0=1, psi_j = sum_{k=1..min(p,j)} phi_k psi_{j-k}
    psi <- numeric(psi_trunc + 1L)
    psi[1] <- 1.0  # psi_0
    for (j in 2:(psi_trunc + 1L)) {
      jj <- j - 1L
      kmax <- min(p_order, jj)
      psi[j] <- sum(phi[1:kmax] * psi[(j - 1L) - (1:kmax)])
    }
    var_unit <- sum(psi^2)
    if (!is.finite(var_unit) || var_unit <= tol) stop("Failed to compute AR(p) scaling (non-finite variance).")
    sd_eps <- sqrt(1 / var_unit)
  }

  ## Simulate with burn-in
  n_all <- Ts + burnin
  u_all <- numeric(n_all)
  eps   <- stats::rnorm(n_all, sd = sd_eps)

  for (t in 1:n_all) {
    acc <- eps[t]
    kmax <- min(p_order, t - 1L)
    if (kmax >= 1L) {
      acc <- acc + sum(phi[1:kmax] * u_all[t - (1:kmax)])
    }
    u_all[t] <- acc
  }
  u <- u_all[(burnin + 1L):(burnin + Ts)]
  Z <- mst + u

  ## Ordinalization
  seg <- c(-Inf, cut, Inf)
  y   <- findInterval(Z, seg, left.open = TRUE, rightmost.closed = TRUE)
  stopifnot(all(y >= 1L & y <= K))

  Yw <- matrix(0L, Ts, K)
  Yw[cbind(seq_len(Ts), y)] <- 1L
  list(Z = Z, X_hour = y, X_hour_wide = Yw)
}
