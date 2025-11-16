#' TSAOP: Time-Series Autoregressive Ordinal Probit
#'
#' Core infrastructure and native registration for the TSAOP package.
#'
#' @docType package
#' @name TSAOP-package
#' @aliases TSAOP
#'
#' @useDynLib TSAOP, .registration = TRUE
#'
#' @importFrom stats coef fitted pnorm printCoefmat rnorm
#' @importFrom stats constrOptim pacf qnorm
"_PACKAGE"
