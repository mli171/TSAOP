# TSAOP: Autoregressive Ordinal Probit Model for Ordinal Categorical Time series

## Overview

Autoregressive ordinal probit model fitting for ordinal time series data. 

Supported methods:

- Conditional least square with lag 1 estimator for AR(1) latent process (Watts et al., 2025);
- Maximum pairwise log-likelihood estimator (Varin & Vidoni, 2006); 

Additional useful features:

- TSAOP includes a (uni- and multi-variate) **innovations algorithm** backend for Gaussian covariance-driven prediction and likelihood evaluation, with efficient C implementations and interchangeable covariance inputs (Toeplitz/autocovariance, full covariance matrix, or callback). See Brockwell & Davis (1991) for the standard innovations framework.

To be added soon:

- The sequential least square estimator will be added soon (Li and Lu, 2022);

## Installation
You can install the development version from Github:

```{r}
# install.packages("remotes")
remotes::install_github("mli171/TSAOP", build_vignettes = TRUE, force = TRUE)
```

## Model fitting example via simulated data

### Example 1: Simulated ordinal time series with AR(1) latent correlation

```{r}
T  <- 600
K  <- 5
t  <- 1:T

# Design matrix: intercept + mild trend + ~30-day seasonality
X <- cbind(
  Intercept = 1,
  Trend     = as.numeric(scale(t, scale = FALSE)),
  c30       = cos(2*pi*t/30),
  s30       = sin(2*pi*t/30)
)

cut_true   <- c(0.5, 1.2, 2.0)
theta_true <- c(-0.5, 0.001, 0.30, -0.10)
rho_true   <- 0.6

sim <- aop_sim(
  ci = cut_true, theta = theta_true, rho = rho_true,
  K = K, Ts = T, DesignX = X, seed = 123
)

fit_clse <- aopts(y = sim$X_hour, X = X, method = "clse")
fit_pl   <- aopts(y = sim$X_hour, X = X, method = "pl")

print(summary(fit_clse))
print(summary(fit_pl))
```

### Example 2: Simulated ordinal time series with AR(2) latent correlation

Maximize pairwise log-likelihood ('pl') method is supported.

```{r}
T <- 600
K <- 5
t <- 1:T

X <- cbind(
  Intercept = 1,
  Trend = as.numeric(scale(t, scale = FALSE)),
  c30 = cos(2*pi*t/30),
  s30 = sin(2*pi*t/30)
)

cut_true   <- c(0.6, 1.4, 2.4)
theta_true <- c(-0.5, 0.003, 0.20, -0.10)
rho_true   <- c(0.6, 0.3)

sim <- aop_sim(
  ci = cut_true, theta = theta_true, rho = rho_true,
  K = K, Ts = T, DesignX = X, seed = 123
)

fit_pl <- aopts(
  y = sim$X_hour, X = X, method = "pl",
  control = list(order_pair_lik = 5L, ar_order = 2L)
)

print(summary(fit_pl))
```

## References

[1] Watts, M., Li, M., Su, Y., & Lu, Q. (2026). *Conditional least squares estimation for autoregressive ordered probit models*. Manuscript in preparation.

[2] Li, M., & Lu, Q. (2022). *Changepoint detection in autocorrelated ordinal categorical time series*. *Environmetrics, 33*(7), e2752.

[3] Varin, C., & Vidoni, P. (2006). Pairwise likelihood inference for ordinal categorical time series. *Computational Statistics & Data Analysis, 51*(4), 2365–2373.

[4] Brockwell, P. J., & Davis, R. A. (1991). Time Series: Theory and Methods (2nd ed.). *Springer*.
