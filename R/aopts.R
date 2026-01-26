#' Autoregressive Ordinal Probit Model for Categorical Time Series
#'
#' Main entry point for fitting autoregressive ordinal probit models for
#' categorical time series.
#'
#' Currently supports Conditional Least Squares Estimation (method = "clse")
#' and Maximum Pairwise (Composite) Log-Likelihood (method = "pl"), implemented
#' via native routines registered in the package shared library.
#'
#' @details
#' The package compiles the C/C++ source in \code{src/} into a single shared
#' library that R loads automatically. Native calls are made with
#' \code{PACKAGE="TSAOP"} and require that the corresponding routines are
#' registered (see \code{useDynLib(TSAOP, .registration=TRUE)} in the
#' \code{NAMESPACE} file). Users should not call \code{dyn.load()} or pass a
#' \code{.so} path directly.
#'
#' \strong{Implemented / planned methods.}
#' \itemize{
#' \item \code{"clse"} (Conditional Least Squares; implemented).
#' \itemize{
#' \item \emph{Parameter estimation:} Parameters are estimated by minimizing
#' the CLS objective via \code{\link[stats]{constrOptim}} subject to
#' monotonicity constraints on cutpoints and bounds on \code{rho}.
#' \item \emph{Standard error approximation:} Robust (sandwich) variance
#' estimator:
#' \deqn{V = H^{-1} W H^{-1},}
#' where \eqn{H} is the Hessian of the CLS objective and \eqn{W} is the
#' variability matrix returned by the native routine
#' \code{score_by_t_CLSEW_aop}. (Despite its name, this routine returns the
#' aggregated \eqn{W} matrix rather than a per-time score matrix.)
#' }
#' \item \code{"pl"} (Maximum pairwise log-likelihood; implemented).
#' \itemize{
#' \item \emph{Parameter estimation:} Parameters are estimated by maximizing
#' the (average) pairwise/composite log-likelihood using
#' \code{\link[stats]{optim}} on an unconstrained transformed parameterization
#' (cutpoint differences and AR parameters transformed to enforce constraints).
#' \item \emph{Standard error approximation:} When available, a robust
#' Godambe/sandwich variance estimator of the form
#' \deqn{V = H^{-1} J H^{-1}}
#' is returned in \code{vcov_sand}, where \eqn{H} is the Hessian (in the
#' original parameterization) and \eqn{J} is a variability matrix constructed
#' from composite-score contributions. If unavailable, standard errors are
#' returned as \code{NA}.
#' }
#' \item \code{"lse"} (Least squares estimator; \emph{TBA}).
#' }
#'
#' @param y Integer or factor vector of length \eqn{T}. If a factor, levels must
#' be ordered. Internally, values are mapped to \code{1, ..., K}.
#' @param X Design matrix of dimension \eqn{T \times p}. Include an intercept
#' column if desired.
#' @param method Estimation method. Supported: \code{"clse"} and \code{"pl"}.
#' @param control List of control arguments passed to the optimizer.
#' For \code{"clse"}, this is passed to \code{\link[stats]{constrOptim}}.
#' For \code{"pl"}, this is passed to \code{\link[stats]{optim}} (with
#' method-specific entries such as \code{order_pair_lik} and \code{ar_order}
#' handled internally and not forwarded to \code{optim}).
#' @param optim_method Optimization method passed to the optimizer
#' (e.g., \code{"BFGS"}, \code{"CG"}).
#'
#' @return An object of class \code{"aopts_fit"} with components:
#' \itemize{
#' \item \code{method}: Estimation method used.
#' \item \code{par}: Estimated parameter vector.
#' \item \code{se}: Standard errors. For \code{"clse"}, these are based on the
#' sandwich variance \eqn{V = H^{-1} W H^{-1}} when available. For \code{"pl"},
#' these are based on the Godambe/sandwich variance when available; otherwise
#' \code{NA}.
#' \item \code{vcov_sand}: Robust sandwich (Godambe) variance-covariance matrix
#' when available; otherwise \code{NULL} or an \code{NA} matrix.
#' \item \code{K}, \code{T}, \code{p}: Model dimensions.
#' \item \code{par_map}: Decoded parameter blocks: \code{cut}, \code{beta},
#' \code{rho}.
#' \item \code{value}, \code{convergence}, \code{message}: Optimizer outputs.
#' }
#' @examples
#' ## Example 1: Simulated ordinal time series with AR(1) latent correlation
#'
#' T  <- 600
#' K  <- 5
#' t  <- 1:T
#'
#' # Design matrix: intercept + mild trend + ~30-day seasonality
#' X <- cbind(
#'   Intercept = 1,
#'   Trend     = as.numeric(scale(t, scale = FALSE)),
#'   c30       = cos(2*pi*t/30),
#'   s30       = sin(2*pi*t/30)
#' )
#'
#' cut_true   <- c(0.5, 1.2, 2.0)
#' theta_true <- c(-0.5, 0.001, 0.30, -0.10)
#' rho_true   <- 0.6
#'
#' sim <- aop_sim(
#'   ci = cut_true, theta = theta_true, rho = rho_true,
#'   K = K, Ts = T, DesignX = X, seed = 123
#' )
#'
#' fit_clse <- aopts(y = sim$X_hour, X = X, method = "clse")
#' fit_pl   <- aopts(y = sim$X_hour, X = X, method = "pl")
#'
#' print(summary(fit_clse))
#' print(summary(fit_pl))
#'
#'
#'
#' ## Example 2: Simulated ordinal time series with AR(2) latent correlation
#'
#' T <- 600
#' K <- 5
#' t <- 1:T
#'
#' X <- cbind(
#'   Intercept = 1,
#'   Trend = as.numeric(scale(t, scale = FALSE)),
#'   c30 = cos(2*pi*t/30),
#'   s30 = sin(2*pi*t/30)
#' )
#'
#' cut_true   <- c(0.6, 1.4, 2.4)
#' theta_true <- c(-0.5, 0.003, 0.20, -0.10)
#' rho_true   <- c(0.6, 0.3)
#'
#' sim <- aop_sim(
#'   ci = cut_true, theta = theta_true, rho = rho_true,
#'   K = K, Ts = T, DesignX = X, seed = 123)
#'
#' fit_pl <- aopts(
#'   y = sim$X_hour, X = X, method = "pl",
#'   control = list(order_pair_lik = 5L, ar_order = 2L))
#' print(summary(fit_pl))
#'
#' @references
#' Varin, C. and Vidoni, P. (2006).
#' \emph{Pairwise likelihood inference for ordinal categorical time series}.
#' \emph{Computational Statistics \& Data Analysis} \bold{51}(4), 2365--2373.
#'
#' @export
aopts <- function(y, X,
                  method = c("clse", "pl"),
                  control = list(reltol = 1e-7, maxit = 1000),
                  optim_method = "BFGS") {

  method <- match.arg(method, c("clse", "pl"))

  ## --- y: integer 1..K; drop NAs but keep alignment with X ---
  if (is.factor(y)) y <- as.integer(y)
  stopifnot(is.numeric(y))
  mask <- is.finite(y)
  if (!all(mask)) {
    y <- y[mask]
    X <- X[mask, , drop = FALSE]
  }

  ## Map to 1..K
  levs <- sort(unique(y))
  y_mapped <- match(y, levs)
  K <- length(levs)
  Tt <- length(y_mapped)
  stopifnot(is.matrix(X), nrow(X) == Tt)

  ## One-hot matrix (T x K)
  Xw <- matrix(0L, nrow = Tt, ncol = K)
  Xw[cbind(seq_len(Tt), y_mapped)] <- 1L

  ## Fit + SE logic by method (so future methods can plug in cleanly)
  if (method == "clse") {

    fit <- fit_AopCLSW(y_mapped, Xw, X,
                       control = control,
                       method  = optim_method)

    par <- fit$par

    ## SE for CLSE
    se <- tryCatch({
      sqrt(pmax(diag(fit$vcov_sand), 0))
    }, error = function(e) rep(NA_real_, length(fit$par)))

  } else if (method == "pl") {

    ## Method-specific controls without changing your function signature:
    ## allow user to pass these via `control`, otherwise defaults.
    order_pair_lik <- control$order_pair_lik %||% 5L
    ar_order <- control$ar_order %||% 1L

    fit <- fit_AopPL(y_mapped, X,
                     order_pair_lik = order_pair_lik,
                     ar_order = ar_order,
                     control = control,
                     method = optim_method)


    par <- fit$par

    ## SE for PL: prefer sandwich (Godambe) if available;
    ##            otherwise naive Hessian fallback;
    se <- rep(NA_real_, length(par))
    if (!is.null(fit$vcov_sand) && is.matrix(fit$vcov_sand) &&
        all(dim(fit$vcov_sand) == length(par)) && !all(is.na(fit$vcov_sand))) {
      se <- sqrt(pmax(diag(fit$vcov_sand), 0))
    } else if (!is.null(fit$vcov_naive) && is.matrix(fit$vcov_naive) &&
               all(dim(fit$vcov_naive) == length(par)) && !all(is.na(fit$vcov_naive))) {
      se <- sqrt(pmax(diag(fit$vcov_naive), 0))
    }

  } else {
    stop("Unsupported method.")
  }

  ## Decode parameter vector to named blocks
  p <- ncol(X)
  nbase <- (K - 2L) + p
  q <- length(par) - nbase
  stopifnot(q >= 1L)

  cut_diff <- if (K >= 3) par[seq_len(K - 2L)] else numeric(0)
  cut  <- c(0, cut_diff)
  beta <- par[(K - 1L):(K - 1L + p - 1L)]
  rho  <- par[(nbase + 1L):length(par)]  ## length 1 for AR(1), length q for AR(p)

  out <- list(
    method      = method,
    par         = par,
    se          = se,
    vcov_sand   = fit$vcov_sand,
    vcov_naive  = fit$vcov_naive,
    par_map     = list(cut = cut, beta = beta, rho = rho),
    value       = fit$value,
    convergence = fit$convergence,
    message     = fit$message,
    counts      = fit$counts,
    K           = K,
    T           = Tt,
    p           = p,
    levels      = levs,
    X           = X,
    y           = y_mapped
  )
  class(out) <- "aopts_fit"
  out
}


## --------------------- Native oracle & optimizer wrappers ---------------------

make_CLSEW_oracle <- function(Xl, Xw, DesignX) {

  # use the current package name dynamically
  pkg <- utils::packageName() %||% "TSAOP"

  Xl <- as.integer(Xl)
  Xw <- matrix(as.integer(Xw), nrow = nrow(Xw), ncol = ncol(Xw))
  storage.mode(DesignX) <- "double"

  K  <- as.integer(ncol(Xw))
  Ts <- as.integer(length(Xl))
  p  <- as.integer(ncol(DesignX))
  npar <- (K - 2L) + p + 1L

  .C("read_dimensions",  K, p, Ts, PACKAGE = pkg, NAOK = TRUE)
  .C("allocate",                 PACKAGE = pkg, NAOK = TRUE)
  .C("read_data",        Xl,     PACKAGE = pkg, NAOK = TRUE)
  .C("read_data_matrix", as.integer(Xw), PACKAGE = pkg, NAOK = TRUE)
  .C("read_covariates",  as.double(DesignX), PACKAGE = pkg, NAOK = TRUE)

  obj_call <- function(par) {
    .C("CLSEW_aop_Rcall",
       par = as.double(par),
       out = double(1L),
       PACKAGE = pkg, NAOK = TRUE)
  }

  fn <- function(par) obj_call(par)$out

  gr <- function(par) {
    gg <- .C("gradient_CLSEW_aop",
             as.double(par),
             grad = double(npar),
             PACKAGE = pkg,
             NAOK = TRUE)$grad
    gg
  }
  W_meat <- function(par) {
    out <- .C("score_by_t_CLSEW_aop",
              as.double(par),
              Wout = double(npar * npar),
              PACKAGE = pkg,
              NAOK = TRUE)$Wout
    matrix(out, nrow = npar, ncol = npar)
  }
  free <- function() {
    .C("deallocate", PACKAGE = pkg, NAOK = TRUE)
    invisible(NULL)
  }
  list(fn = fn, gr = gr, W_meat = W_meat,
       free = free, npar = npar, K = K, p = p, Ts = Ts)
}


fit_AopCLSW <- function(X_hour, X_hour_wide, DesignXEst,
                        control = list(reltol = 1e-7, maxit = 1000),
                        method = "BFGS") {
  stopifnot(is.matrix(X_hour_wide), nrow(X_hour_wide) == length(X_hour))
  stopifnot(is.matrix(DesignXEst),  nrow(DesignXEst)  == length(X_hour))

  K <- ncol(X_hour_wide)

  cout <- summary(as.factor(X_hour))
  ci_initial <- as.vector(qnorm(cumsum(cout / sum(cout))))[1:(K - 1)]
  phi_initial <- stats::pacf(X_hour, plot = FALSE)$acf[1]
  par_initial <- c(ci_initial[2:(K-1)] - ci_initial[1], -ci_initial[1],
                   rep(0, NCOL(DesignXEst) - 1), phi_initial)

  if (K == 3) {
    constrLSE <- rbind(c(rep(0, length(par_initial)-1),  1),
                       c(rep(0, length(par_initial)-1), -1)) # rho bounds
  } else {
    constrLSE <- matrix(0, nrow = K - 2, ncol = length(par_initial))
    constrLSE[1,1] <- 1
    for (ii in 2:(K-2)) constrLSE[ii, c(ii-1, ii)] <- c(-1, 1) # ci monotone
    constrLSE <- rbind(constrLSE,
                       c(rep(0, ncol(constrLSE)-1),  1),
                       c(rep(0, ncol(constrLSE)-1), -1)) # rho bounds
  }
  constrLSE_ci <- c(rep(0, nrow(constrLSE) - 2), -1, -1)

  oracle <- make_CLSEW_oracle(X_hour, X_hour_wide, DesignXEst)
  on.exit(oracle$free(), add = TRUE)

  res <- constrOptim(theta = par_initial,
                     f       = oracle$fn,
                     grad    = oracle$gr,
                     hessian = TRUE,
                     ui      = constrLSE,
                     ci      = constrLSE_ci,
                     control = control,
                     method  = method)

  Htilde <- (res$hessian + t(res$hessian)) / 2

  sandwich_vcov <- tryCatch({
    Hinv <- tryCatch(chol2inv(chol(Htilde)), error = function(e) solve(Htilde))
    Wtilde <- oracle$W_meat(res$par) # already P x P
    V <- Hinv %*% Wtilde %*% Hinv
    (V + t(V)) / 2
  }, error = function(e) {
    matrix(NA_real_, length(res$par), length(res$par))
  })


  list(
    par_init    = par_initial,
    par         = res$par,
    value       = res$value,
    counts      = res$counts,
    hessian     = res$hessian,
    vcov_sand   = sandwich_vcov,
    convergence = res$convergence,
    message     = res$message,
    K           = K,
    control     = control,
    method      = method
  )
}

## --------------------------- S3 helpers --------------------------------------

#' @export
coef.aopts_fit <- function(object, ...) {
  K <- object$K
  p <- object$p
  par <- object$par
  q <- length(par) - ((K - 2L) + p)

  nm_phi <- if (q == 1L) "rho" else paste0("phi", seq_len(q))

  nm <- c(
    if (K >= 3) paste0("c", 2:(K-1)) else character(0),
    colnames(object$X) %||% paste0("beta", seq_len(p)),
    nm_phi
  )
  stats::setNames(par, nm)
}

# coef.aopts_fit <- function(object, ...) {
#   K <- object$K
#   p <- object$p
#   par <- object$par
#   nm <- c(
#     if (K >= 3) paste0("c", 2:(K-1)) else character(0),
#     colnames(object$X) %||% paste0("beta", seq_len(p)),
#     "rho"
#   )
#   stats::setNames(par, nm)
# }

#' @export
vcov.aopts_fit <- function(object, ...) {
  if (!is.null(object$vcov_sand) && is.matrix(object$vcov_sand)) {
    V <- object$vcov_sand
    dimnames(V) <- list(names(coef(object)), names(coef(object)))
    return(V)
  }
  if (!is.null(object$vcov_naive) && is.matrix(object$vcov_naive)) {
    V <- object$vcov_naive
    dimnames(V) <- list(names(coef(object)), names(coef(object)))
    return(V)
  }
  if (is.null(object$se) || all(is.na(object$se))) return(NA_real_)
  s <- object$se
  V <- diag(s^2)
  dimnames(V) <- list(names(coef(object)), names(coef(object)))
  V
}

#' @export
summary.aopts_fit <- function(object, ...) {
  cf <- coef(object)
  se <- object$se
  z  <- cf / se
  tab <- cbind(Estimate = cf, `Std. Error` = se, `z value` = z)
  res <- list(
    call        = match.call(),
    method      = object$method,
    table       = tab,
    value       = object$value,
    convergence = object$convergence,
    message     = object$message
  )
  class(res) <- "summary.aopts_fit"
  res
}

#' @export
print.summary.aopts_fit <- function(x, ...) {
  cat("aopts fit (method:", x$method, ")\n")
  cat("criterion value:", format(x$value, digits = 6), "\n")
  cat("convergence:", x$convergence, "\n")
  if (!is.null(x$message) && nzchar(x$message)) cat("message:", x$message, "\n")
  cat("\nCoefficients:\n")
  printCoefmat(x$table, na.print = "")
  invisible(x)
}

# Fitted marginal category probabilities under probit-threshold model
.marginal_probs <- function(cut, beta, X) {
  Tt <- nrow(X); K <- length(cut) + 1L
  mst <- as.vector(X %*% beta)
  seg <- c(-Inf, cut, Inf)
  tmp <- t(matrix(rep(seg, Tt), nrow = length(seg), ncol = Tt)) -
    matrix(rep(mst, length(seg)), nrow = Tt)
  pnorm(tmp[, 2:(K+1)]) - pnorm(tmp[, 1:K])  # T x K
}

#' @export
fitted.aopts_fit <- function(object, ...) {
  with(object, .marginal_probs(object$par_map$cut, object$par_map$beta, object$X))
}

#' @export
predict.aopts_fit <- function(object, newdata = NULL, type = c("prob"), ...) {
  type <- match.arg(type)
  X <- if (is.null(newdata)) object$X else as.matrix(newdata)
  .marginal_probs(object$par_map$cut, object$par_map$beta, X)
}

#' @export
residuals.aopts_fit <- function(object, type = c("pearson"), ...) {
  type <- match.arg(type)
  P <- fitted(object)              # T x K
  Tt <- nrow(P); K <- ncol(P)
  Yw <- matrix(0, Tt, K)
  Yw[cbind(seq_len(Tt), object$y)] <- 1
  Yw - P
}

`%||%` <- function(a, b) if (is.null(a)) b else a
