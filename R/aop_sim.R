#' Simulate ordinal time series under AR(1) probit-threshold model
#'
#' @param ci Numeric vector of cutpoints. Either:
#' \itemize{
#'   \item \code{length(ci) == K-2}: interior cutpoints, using the convention \eqn{c_1 = 0}.
#'   \item \code{length(ci) == K-1}: full, strictly increasing cutpoints \eqn{c_1 < \cdots < c_{K-1}}.
#' }
#' @param theta numeric, length ncol(DesignX); regression coefficients
#' @param rho AR(1) coefficient in (-1, 1)
#' @param K number of categories
#' @param Ts length of series
#' @param DesignX T x p design matrix (include intercept if desired)
#' @param seed optional integer for reproducibility
#' @return list with latent Z, integer y in 1..K, and one-hot matrix Yw (T x K)
#' @export
aop_sim <- function(ci, theta, rho, K, Ts, DesignX, seed = NULL) {
  stopifnot(is.matrix(DesignX), nrow(DesignX) == Ts, ncol(DesignX) == length(theta))
  if (!is.null(seed)) set.seed(seed)

  # Build ordered cutpoints
  cut <- if (length(ci) == K - 1L) {
    sort(as.numeric(ci))
  } else if (length(ci) == K - 2L) {
    sort(c(0, as.numeric(ci)))  # c1 = 0 convention for simulation only
  } else {
    stop("ci must have length K-1 (full cutpoints) or K-2 (c1 = 0 convention).")
  }
  if (any(!is.finite(cut))) stop("cutpoints contain non-finite values.")
  if (any(diff(cut) <= 0))  stop("cutpoints must be strictly increasing.")

  # Latent mean + AR(1) disturbance
  mst <- as.vector(DesignX %*% theta)
  e   <- numeric(Ts)
  sdI <- sqrt(1 - rho^2)
  e[1] <- rnorm(1)
  for (t in 2:Ts) e[t] <- rho * e[t - 1] + rnorm(1, sd = sdI)
  Z <- mst + e

  # Ordinalization
  seg <- c(-Inf, cut, Inf)
  y   <- findInterval(Z, seg, left.open = TRUE, rightmost.closed = TRUE)
  stopifnot(all(y >= 1L & y <= K))

  Yw <- matrix(0L, Ts, K); Yw[cbind(seq_len(Ts), y)] <- 1L
  list(Z = Z, X_hour = y, X_hour_wide = Yw)
}
