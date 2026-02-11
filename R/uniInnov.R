#' Univariate innovations algorithm (generic covariance backend)
#'
#' Compute the univariate innovations algorithm for a zero-mean Gaussian
#' process \eqn{u_1,\dots,u_T} with covariance \eqn{K(i,j)}. The routine builds
#' the (time-varying) linear prediction coefficients \eqn{\theta_t} and
#' innovation variances \eqn{v_t} such that
#' \deqn{\hat u_t = \sum_{j=1}^{L_t} \theta_{t,j}\, e_{t-j}, \qquad e_t = u_t - \hat u_t,}
#' where \eqn{e_t} are one-step innovations with variance \eqn{v_t}, and
#' \eqn{L_t \le t-1} is the number of lag coefficients retained at time \eqn{t}.
#'
#' This wrapper supports three interchangeable covariance backends:
#' \itemize{
#'   \item \strong{Toeplitz/autocovariance} via \code{gamma}: \eqn{K(i,j)=\gamma_{|i-j|}} (fastest).
#'   \item \strong{Full covariance matrix} via \code{K}: arbitrary \eqn{T\times T} covariance.
#'   \item \strong{Callback} via \code{kappa(i,j)}: R function returning scalar \eqn{K(i,j)} (most flexible, slowest).
#' }
#'
#' The heavy lifting is done in C via registered \code{.Call()} entry points:
#' \code{innovations_uni_gamma}, \code{innovations_uni_K}, and \code{innovations_uni_kappa}.
#'
#' @param u Numeric vector of length \eqn{T} (the observed series, typically mean-corrected).
#'
#' @param gamma Optional numeric vector of autocovariances \eqn{\gamma_0,\gamma_1,\dots}.
#'   If provided, uses Toeplitz structure \eqn{K(i,j)=\gamma_{|i-j|}}.
#'   Must be long enough for the maximum lag accessed: at least \code{T} if \code{lag_max} is \code{NULL},
#'   otherwise at least \code{lag_max + 1}.
#'
#' @param K Optional numeric matrix (or something coercible to a numeric matrix) giving the full covariance
#'   matrix of \code{u}. Must be square with dimension \eqn{T \times T}.
#'
#' @param kappa Optional function \code{kappa(i,j)} returning a finite numeric scalar covariance value
#'   \eqn{K(i,j)} for 1-based integer indices \code{i} and \code{j}. This is the most general interface
#'   but may be substantially slower than \code{gamma} or \code{K}.
#'
#' @param jitter Nonnegative scalar added to each innovation variance \code{v[t]} to improve numerical stability
#'   when the covariance is near-singular. Default \code{1e-10}.
#'
#' @param lag_max Optional integer. If \code{NULL} (default), the algorithm may retain up to \eqn{T-1} lags.
#'   If supplied, \code{lag_max} is a \strong{hard cap} on the number of lag coefficients retained/used:
#'   at each time \eqn{t}, \code{theta[[t]]} has length \code{L_t = min(t-1, lag_max_used)}, where
#'   \code{lag_max_used <= lag_max}. This controls memory and computation.
#'
#' @param lag_tol Optional nonnegative scalar. If not \code{NULL}, enables an \strong{adaptive lag-length rule}.
#'   Once the \strong{oldest retained} coefficient satisfies \code{abs(theta[[t]][L_t]) < lag_tol},
#'   the effective lag length is \strong{fixed} for subsequent times (returns \code{lag_max_used}).
#'   Default \code{1e-8}. Use \code{NULL} to disable tolerance-based lag fixing.
#'
#' @param ahead Integer \eqn{h \ge 1} controlling the \eqn{h}-step-ahead conditional mean output.
#'   If \code{NULL}, no \code{uhat_ahead} is computed. If \code{1L}, \code{uhat_ahead} equals \code{uhat}.
#'
#' @param PACKAGE Character scalar giving the package/DLL name used for registered symbol lookup in \code{.Call()}.
#'   Defaults to \code{"TSAOP"}. Change this if your shared library is registered under a different name.
#'
#' @return A named list with components:
#' \describe{
#'   \item{theta}{List of length \eqn{T}. Element \code{theta[[t]]} is a numeric vector of length
#'     \code{L_t = min(t-1, lag_max_used)} containing coefficients for predicting \code{u[t]} from past innovations.}
#'   \item{v}{Numeric vector of length \eqn{T} giving innovation variances \eqn{v_t}.}
#'   \item{uhat}{Numeric vector of length \eqn{T} giving one-step fitted means \eqn{\hat u_t}.}
#'   \item{innov}{Numeric vector of length \eqn{T} giving one-step innovations \eqn{e_t}.}
#'   \item{uhat_ahead}{If \code{ahead} is not \code{NULL}, numeric vector of length \eqn{T} giving the
#'     \eqn{h}-step-ahead conditional mean \eqn{E[u_t \mid u_{1:(t-h)}]}; otherwise \code{NULL}.}
#'   \item{lag_max_used}{Integer. The final effective lag length used after applying \code{lag_max} and (if enabled)
#'     the tolerance rule \code{lag_tol}.}
#'   \item{lag_used_path}{Integer vector of length \eqn{T}. Entry \code{lag_used_path[t]} equals \code{L_t}, the number
#'     of lag coefficients actually used at time \eqn{t}.}
#' }
#'
#' @details
#' \strong{Backend choice and performance.} Passing \code{gamma} is usually fastest because it uses Toeplitz
#' structure and avoids repeated covariance lookups. Passing \code{K} is next best. Passing a \code{kappa}
#' callback can be much slower because it triggers R-to-C calls inside the recursion.
#'
#' \strong{Likelihood usage.} For a zero-mean Gaussian model with innovation variances \code{v} and innovations \code{e},
#' the (negative) conditional log-likelihood (up to an additive constant) is
#' \deqn{\frac12 \sum_{t=1}^T \left(\log v_t + \frac{e_t^2}{v_t}\right).}
#'
#' @examples
#' # Example 1: AR(1) Toeplitz backend via gamma (fast)
#' set.seed(1)
#' Tt <- 200
#' phi <- 0.7
#' sigma2 <- 1.0
#'
#' x <- as.numeric(arima.sim(model = list(ar = phi), n = Tt, sd = sqrt(sigma2)))
#'
#' gamma_ar1 <- function(phi, sigma2, T) {
#'   g0 <- sigma2 / (1 - phi^2)
#'   g0 * (phi^(0:(T - 1L)))
#' }
#' gamma <- gamma_ar1(phi, sigma2, Tt)
#'
#' fit <- uniInnov(u = x, gamma = gamma, ahead = 1L)
#' nll <- 0.5 * sum(log(fit$v) + (fit$innov^2) / fit$v)
#' nll
#'
#' # Example 2: Full covariance backend via K
#' K <- toeplitz(gamma)  # base R
#' fitK <- uniInnov(u = x, K = K, ahead = 2L)
#' all.equal(fit$v, fitK$v, tolerance = 1e-8)
#'
#' # Example 3: Callback backend via kappa(i,j)
#' kappa_ar1 <- function(phi, sigma2, T) {
#'   gamma <- gamma_ar1(phi, sigma2, T)
#'   function(i, j) {
#'     h <- abs(as.integer(i) - as.integer(j))
#'     gamma[h + 1L]
#'   }
#' }
#' fitF <- uniInnov(u = x, kappa = kappa_ar1(phi, sigma2, Tt))
#' all.equal(fit$v, fitF$v, tolerance = 1e-8)
#'
#' # Example 4: Fixed-order approximation + tolerance-based lag fixing
#' fitApprox <- uniInnov(u = x, gamma = gamma, lag_max = 20L, lag_tol = 1e-6, ahead = NULL)
#' fitApprox$lag_max_used
#'
#' @export
uniInnov <- function(u, gamma = NULL, K = NULL, kappa = NULL,
                     jitter = 1e-10, lag_max = NULL, lag_tol = 1e-8, ahead = 1L,
                     PACKAGE = "TSAOP") {

  u <- as.numeric(u)
  if (length(u) < 1L) stop("u must have length >= 1.")
  Tt <- length(u)

  n_backends <- (!is.null(gamma)) + (!is.null(K)) + (!is.null(kappa))
  if (n_backends != 1L) stop("Provide exactly one of: gamma, K, or kappa.")

  jitter <- as.numeric(jitter)
  if (!is.finite(jitter) || jitter < 0) stop("jitter must be finite and >= 0.")

  if (!is.null(lag_max)) {
    lag_max <- as.integer(lag_max)
    if (!is.finite(lag_max) || lag_max < 1L) stop("lag_max must be >= 1 or NULL.")
  }
  if (!is.null(lag_tol)) {
    lag_tol <- as.numeric(lag_tol)
    if (!is.finite(lag_tol) || lag_tol < 0) stop("lag_tol must be finite and >= 0, or NULL.")
  }
  if (!is.null(ahead)) {
    ahead <- as.integer(ahead)
    if (!is.finite(ahead) || ahead < 1L) stop("ahead must be >= 1 or NULL.")
  }

  if (!is.null(gamma)) {
    gamma <- as.numeric(gamma)
    if (length(gamma) < 1L) stop("gamma must have length >= 1.")
    # Optional helpful check: enough lags for requested truncation
    needL <- if (is.null(lag_max)) Tt else (min(lag_max, Tt - 1L) + 1L)
    if (length(gamma) < needL) {
      stop("gamma is too short: need at least ", needL, " entries for this T/lag_max.")
    }

    .Call("innovations_uni_gamma", gamma, u, jitter, lag_max, lag_tol, ahead, PACKAGE = PACKAGE)

  } else if (!is.null(K)) {
    if (!is.matrix(K)) K <- as.matrix(K)
    storage.mode(K) <- "double"
    if (nrow(K) != Tt || ncol(K) != Tt) stop("K must be a T x T matrix with T=length(u).")

    .Call("innovations_uni_K", K, u, jitter, lag_max, lag_tol, ahead, PACKAGE = PACKAGE)

  } else {
    if (!is.function(kappa)) stop("kappa must be a function kappa(i,j) returning a scalar.")
    .Call("innovations_uni_kappa", kappa, u, jitter, lag_max, lag_tol, ahead, PACKAGE = PACKAGE)
  }
}
