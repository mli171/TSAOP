#' Autoregressive Ordinal Probit Model for Categorical Time Series
#'
#' Main entry point for fitting ordinal time–series models.
#' Currently supports Conditional Least Squares Estimation (method = "clse"),
#' using native routines registered in the package's shared library.
#'
#' @details
#' This package compiles code in \code{src/} into a single shared library that
#' R loads automatically. Native calls use \code{PACKAGE="TSAOP"}; you should
#' not call \code{dyn.load()} or pass a \code{.so} path.
#'
#' @param y Integer or factor vector of length T (levels must be ordered);
#'          mapped to 1..K internally.
#' @param X Design matrix (T x p). Include an intercept column if you want one.
#' @param method Estimation method. For now: \code{"clse"}.
#' @param control List passed to \code{constrOptim} (e.g., \code{list(reltol=1e-7,maxit=1000)}).
#' @param optim_method Optimizer for \code{constrOptim} ("BFGS", "CG", ...).
#' @param hessian Logical; compute Hessian (for SEs).
#'
#' @return An object of class \code{"aopts_fit"} with elements:
#' \itemize{
#'   \item \code{method} – "clse"
#'   \item \code{par} – estimated parameter vector
#'   \item \code{se} – standard errors (if \code{hessian=TRUE} and invertible)
#'   \item \code{K}, \code{T}, \code{p} – dimensions
#'   \item \code{par_map} – list with decoded pieces: \code{cut}, \code{beta}, \code{rho}
#'   \item \code{value}, \code{convergence}, \code{message} – optimizer outputs
#' }
#'
#' @examples
#' ## Example 1: Simulated ordinal time series with AR(1) latent correlation
#' set.seed(1)
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
#' fit <- aopts(y = sim$X_hour, X = X, method = "clse")
#' print(summary(fit))
#'
#' @export
aopts <- function(y, X,
                  method = c("clse"),
                  control = list(reltol = 1e-7, maxit = 1000),
                  optim_method = "BFGS",
                  hessian = TRUE) {

  method <- match.arg(method, c("clse"))

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

  if (method == "clse") {
    fit <- fit_AopCLSW(y_mapped, Xw, X,
                       control = control, method = optim_method,
                       myHessian = hessian)
  } else {
    stop("Unsupported method.")
  }

  ## Decode parameter vector to named blocks
  # par layout: (K-2) cutpoint diffs (c1=0 convention), p betas, 1 rho
  p <- ncol(X)
  par <- fit$par
  stopifnot(length(par) == (K - 2L) + p + 1L)

  cut_diff <- if (K >= 3) par[seq_len(K - 2L)] else numeric(0)
  cut <- c(0, cut_diff)  # c1 = 0 convention
  beta <- par[(K - 1L):(K - 1L + p - 1L)]
  rho  <- par[length(par)]

  se <- rep(NA_real_, length(par))
  if (!is.null(fit$hessian) && is.matrix(fit$hessian) && hessian) {
    H <- (fit$hessian + t(fit$hessian)) / 2
    e <- try(chol2inv(chol(H)), silent = TRUE)
    if (inherits(e, "try-error")) {
      e <- try(solve(H + 1e-8 * diag(nrow(H))), silent = TRUE)
    }
    if (!inherits(e, "try-error")) se <- sqrt(pmax(diag(e), 0))
  }

  out <- list(
    method      = method,
    par         = par,
    se          = se,
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

  fn <- function(par) {
    .C("CLSEW_aop_Rcall", as.double(par), out = double(1),
       PACKAGE = pkg, NAOK = TRUE)$out
  }
  gr <- function(par) {
    .C("gradient_CLSEW_aop", as.double(par), grad = double(npar),
       PACKAGE = pkg, NAOK = TRUE)$grad
  }
  free <- function() {
    .C("deallocate", PACKAGE = pkg, NAOK = TRUE)
    invisible(NULL)
  }
  list(fn = fn, gr = gr, free = free, npar = npar, K = K, p = p, Ts = Ts)
}


fit_AopCLSW <- function(X_hour, X_hour_wide, DesignXEst,
                        control = list(reltol = 1e-7, maxit = 1000),
                        method = "BFGS",
                        myHessian = FALSE) {
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
                     hessian = myHessian,
                     ui      = constrLSE,
                     ci      = constrLSE_ci,
                     control = control,
                     method  = method)

  list(
    par_init    = par_initial,
    par         = res$par,
    value       = res$value,
    counts      = res$counts,
    hessian     = res$hessian,
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
  nm <- c(
    if (K >= 3) paste0("c", 2:(K-1)) else character(0),
    colnames(object$X) %||% paste0("beta", seq_len(p)),
    "rho"
  )
  stats::setNames(par, nm)
}

#' @export
vcov.aopts_fit <- function(object, ...) {
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
