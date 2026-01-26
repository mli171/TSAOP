## --------------------- PL (pairwise likelihood) wrappers ---------------------

.transf.par.aop_pl <- function(theta, num.cat, num.regr, num.ar = 1L, eps = 1e-4) {
  theta <- as.numeric(theta)
  num.ar <- as.integer(num.ar)
  transf.theta <- rep(NA_real_, length(theta))

  ## cut points: theta[1:(K-2)]
  if (num.cat > 2) {
    if (theta[1] <= 0) stop("Not admissible thresholds: first free cutpoint must be > 0.")
    transf.theta[1] <- log(theta[1] - eps)
    if (num.cat > 3) {
      for (i in 2:(num.cat - 2)) {
        transf.theta[i] <- log(theta[i] - theta[i - 1] - eps)
      }
    }
  }

  ## regressors
  if (num.regr != 0) {
    idx_beta <- (num.cat - 1):(num.cat + num.regr - 2)
    transf.theta[idx_beta] <- theta[idx_beta]
  }

  ## AR params (componentwise logit to (-1+eps, 1-eps))
  pos_phi_start <- (num.cat - 2) + num.regr + 1
  if (num.ar >= 1) {
    for (j in 0:(num.ar - 1)) {
      phi <- theta[pos_phi_start + j]
      transf.theta[pos_phi_start + j] <- log((1 - eps - phi) / (phi + 1 - eps))
    }
  }

  transf.theta
}

.back.transf.par.aop_pl <- function(transf.theta, num.cat, num.regr, num.ar = 1L, eps = 1e-4) {
  transf.theta <- as.numeric(transf.theta)
  num.ar <- as.integer(num.ar)
  LARGE.VALUE <- 20
  theta <- rep(NA_real_, length(transf.theta))

  ## cut points
  if (num.cat > 2) {
    theta[1] <- if (transf.theta[1] < LARGE.VALUE) exp(transf.theta[1]) + eps else exp(LARGE.VALUE)
    if (num.cat > 3) {
      for (i in 2:(num.cat - 2)) {
        theta[i] <- if (transf.theta[i] < LARGE.VALUE) theta[i - 1] + exp(transf.theta[i]) + eps else theta[i - 1] + exp(LARGE.VALUE)
      }
    }
  }

  ## regressors
  if (num.regr != 0) {
    idx_beta <- (num.cat - 1):(num.cat + num.regr - 2)
    theta[idx_beta] <- transf.theta[idx_beta]
  }

  ## AR params
  pos_phi_start <- (num.cat - 2) + num.regr + 1
  if (num.ar >= 1) {
    for (j in 0:(num.ar - 1)) {
      tmp <- exp(transf.theta[pos_phi_start + j])
      theta[pos_phi_start + j] <- if (tmp < LARGE.VALUE) (1 - eps) * (1 - tmp) / (1 + tmp) else -(1 - eps)
    }
  }

  theta
}

.compute.Jacobian_pl <- function(theta, num.categ, num.regr, num.ar = 1L, eps = 1e-4) {
  ## Jacobian of u = transf(theta) wrt theta: du/dtheta
  num.par <- length(theta)
  J <- diag(num.par)

  ## cutpoints
  if (num.categ > 2) {
    J[1, 1] <- 1.0 / (theta[1] - eps)
    if (num.categ > 3) {
      for (i in 2:(num.categ - 2)) {
        tmp <- 1.0 / (theta[i] - theta[i - 1] - eps)
        J[i, i - 1] <- -tmp
        J[i, i] <- tmp
      }
    }
  }

  ## AR params
  pos_phi_start <- (num.categ - 2) + num.regr + 1
  if (num.ar >= 1) {
    for (j in 0:(num.ar - 1)) {
      pos <- pos_phi_start + j
      tmp <- (1 - eps)
      J[pos, pos] <- -2 * tmp / ((tmp - theta[pos]) * (tmp + theta[pos]))
    }
  }

  J
}

make_PL_oracle <- function(y_int, DesignX, order_pair_lik = 5L, ar_order = 1L) {

  pkg <- utils::packageName() %||% "TSAOP"

  y_int <- as.integer(y_int)
  storage.mode(DesignX) <- "double"

  K  <- as.integer(max(y_int, na.rm = TRUE))
  Ts <- as.integer(length(y_int))
  p  <- as.integer(ncol(DesignX))
  q  <- as.integer(ar_order)

  npar <- (K - 2L) + p + q

  .C("read_dimensions_pl_arp",        K, p, Ts,                 PACKAGE = pkg, NAOK = TRUE)
  .C("read_pl_ar_order_pl_arp",       as.integer(q),            PACKAGE = pkg, NAOK = TRUE)
  .C("read_order_pair_lik_pl_arp",    as.integer(order_pair_lik), PACKAGE = pkg, NAOK = TRUE)
  .C("allocate_pl_arp",                                      PACKAGE = pkg, NAOK = TRUE)
  .C("read_data_pl_arp",              y_int,                   PACKAGE = pkg, NAOK = TRUE)
  .C("read_covariates_pl_arp",        as.double(DesignX),       PACKAGE = pkg, NAOK = TRUE)

  fn <- function(u) {
    theta <- .back.transf.par.aop_pl(u, K, p, q)
    .C("logPair_aop_ARp_Rcall_pl_arp",
       par     = as.double(theta),
       out     = double(1L),
       PACKAGE = pkg,
       NAOK    = TRUE)$out
  }

  gr <- function(u) {
    theta <- .back.transf.par.aop_pl(u, K, p, q)

    g_theta <- .C("gradient_logPair_aop_ARp_pl_arp",
                  as.double(theta),
                  grad = double(length(theta)),
                  PACKAGE = pkg,
                  NAOK = TRUE)$grad

    J <- .compute.Jacobian_pl(theta, K, p, q)  ## du/dtheta
    as.numeric(g_theta %*% solve(J))           ## g_u = g_theta * dtheta/du
  }

  score_by_t <- function(theta) {
    out <- .C("score_by_t_logPair_aop_ARp_pl_arp",
              as.double(theta),
              scores = double(Ts * npar),
              PACKAGE = pkg,
              NAOK = TRUE)$scores
    matrix(out, nrow = Ts, ncol = npar)
  }

  free <- function() {
    .C("deallocate_pl_arp", PACKAGE = pkg, NAOK = TRUE)
    invisible(NULL)
  }

  list(fn = fn, gr = gr, score_by_t = score_by_t,
       free = free, npar = npar, K = K, p = p, Ts = Ts, q = q,
       order_pair_lik = as.integer(order_pair_lik))
}

fit_AopPL <- function(y_int, DesignX,
                      order_pair_lik = 5L,
                      ar_order = 1L,
                      control = list(reltol = 1e-7, maxit = 1000),
                      method = "BFGS") {

  oracle <- make_PL_oracle(y_int, DesignX, order_pair_lik = order_pair_lik, ar_order = ar_order)
  on.exit(oracle$free(), add = TRUE)

  K <- oracle$K; p <- oracle$p; q <- oracle$q; Ts <- oracle$Ts
  npar <- oracle$npar; L <- oracle$order_pair_lik

  ## start values
  cout <- summary(as.factor(y_int))
  ci_initial <- as.vector(qnorm(cumsum(cout / sum(cout))))[1:(K - 1)]
  phi_initial <- rep(0, q)
  phi_initial[1] <- stats::pacf(y_int, plot = FALSE)$acf[1]

  par_initial <- c(
    ci_initial[2:(K - 1)] - ci_initial[1],
    -ci_initial[1],
    rep(0, NCOL(DesignX) - 1),
    phi_initial
  )
  stopifnot(length(par_initial) == npar)

  u0 <- .transf.par.aop_pl(par_initial, K, p, q)

  order_pair_lik <- control$order_pair_lik %||% 5L
  ar_order <- control$ar_order %||% 1L

  # optim_control <- control
  #
  # if (!is.null(names(optim_control))) {
  #   drop <- names(optim_control) %in% c("order_pair_lik", "ar_order")
  #   optim_control <- optim_control[!drop]
  # }
  # allowed <- c("trace","fnscale","parscale","ndeps","maxit","abstol","reltol",
  #              "alpha","beta","gamma","REPORT")
  # optim_control <- optim_control[names(optim_control) %in% allowed]
  #
  # ## maximize => fnscale=-1
  # optim_control <- utils::modifyList(optim_control, list(fnscale = -1))

  order_pair_lik <- control$order_pair_lik %||% 5L
  ar_order <- control$ar_order %||% 1L


  opt_control <- control
  opt_control$order_pair_lik <- NULL
  opt_control$ar_order <- NULL
  opt_control <- utils::modifyList(opt_control, list(fnscale = -1))

  res <- stats::optim(u0,
                      fn = oracle$fn,
                      gr = oracle$gr,
                      method = method,
                      control = opt_control)

  theta_hat <- .back.transf.par.aop_pl(res$par, K, p, q)

  ## ---------------- SEs (Godambe) ----------------
  pairs_end_t <- integer(Ts)  # local, avoids R CMD check NOTE
  hessian_u   <- NULL

  if (Ts >= 2L) {
    for (tt in 2:Ts) pairs_end_t[tt] <- min(L, tt - 1L)
  }
  total_pairs <- sum(pairs_end_t)

  vcov_sand  <- matrix(NA_real_, npar, npar)
  vcov_naive <- matrix(NA_real_, npar, npar)

  J_u_theta <- .compute.Jacobian_pl(theta_hat, K, p, q)          # du/dtheta
  D_theta_u <- tryCatch(solve(J_u_theta), error = function(e) NULL) # dtheta/du

  if (!is.null(D_theta_u) && total_pairs > 0L) {

    score_theta_t <- oracle$score_by_t(theta_hat)          # averages over pairs ending at t
    score_theta_sum_t <- score_theta_t * pairs_end_t       # sums over pairs ending at t
    score_theta_bar_t <- score_theta_sum_t / total_pairs   # contributions for mean-over-pairs objective

    score_u_bar_t <- score_theta_bar_t %*% D_theta_u
    J_u <- crossprod(score_u_bar_t)

    fn_neg <- function(u) -oracle$fn(u)
    gr_neg <- function(u) -oracle$gr(u)
    hessian_u <- tryCatch(stats::optimHess(res$par, fn = fn_neg, gr = gr_neg), error = function(e) NULL)

    if (!is.null(hessian_u) && is.matrix(hessian_u) && nrow(hessian_u) == npar) {
      H_u <- (hessian_u + t(hessian_u)) / 2
      Hinv_u <- tryCatch(chol2inv(chol(H_u)),
                         error = function(e) tryCatch(solve(H_u), error = function(e2) NULL))

      if (!is.null(Hinv_u)) {
        ## sandwich in u
        V_u_sand <- Hinv_u %*% J_u %*% Hinv_u
        V_u_sand <- (V_u_sand + t(V_u_sand)) / 2
        V_theta_sand <- D_theta_u %*% V_u_sand %*% t(D_theta_u)
        vcov_sand <- (V_theta_sand + t(V_theta_sand)) / 2

        ## naive inverse-Hessian, scaled by #pairs (objective is an average)
        V_u_naive <- Hinv_u / total_pairs
        V_theta_naive <- D_theta_u %*% V_u_naive %*% t(D_theta_u)
        vcov_naive <- (V_theta_naive + t(V_theta_naive)) / 2
      }
    }
  }

  list(
    par_init    = par_initial,
    par         = theta_hat,
    value       = res$value,
    convergence = res$convergence,
    message     = res$message %||% "",
    counts      = res$counts,
    K           = K, p = p, q = q, Ts = Ts,
    order_pair_lik = as.integer(order_pair_lik),
    vcov_sand   = vcov_sand,
    vcov_naive  = vcov_naive,
    hessian_u   = hessian_u,
    pairs_end_t = pairs_end_t,
    total_pairs = total_pairs,
    control     = control,
    method      = method
  )
}

`%||%` <- function(a, b) if (is.null(a)) b else a
