# TSAOP: Autoregressive Ordinal Probit Model for Ordinal Categorical Time series

## Overview

Autoregressive ordinal probit model fitting for ordinal time series data. Currently, 
the only available method is the conditional least square with lag 1 estimator from 
Watts et al. (2025) for AR(1) latent process. The sequential least square 
estimator (Li and Lu, 2022) and maximum pairwise likelihood estimator 
(Varin & Vidoni, 2006) will be added soon.

## Installation
You can install the development version from Github:

```{r}
# install.packages("remotes")
remotes::install_github("mli171/TSAOP", build_vignettes = FALSE, force = TRUE)
```

## Model fitting example via simulated data
```{r}
## Example 1: Simulated ordinal time series with AR(1) latent correlation
set.seed(1)
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

# using clse method
fit <- aopts(y = sim$X_hour, X = X, method = "clse")
print(summary(fit))
```

## References

[1] Watts, M., Li, M., Su, Y., & Lu, Q. (2026). *Conditional least squares estimation for autoregressive ordered probit models*. Manuscript in preparation.

[2] Li, M., & Lu, Q. (2022). *Changepoint detection in autocorrelated ordinal categorical time series*. *Environmetrics, 33*(7), e2752.

[3] Varin, C., & Vidoni, P. (2006). Pairwise likelihood inference for ordinal categorical time series. *Computational Statistics & Data Analysis, 51*(4), 2365–2373.
