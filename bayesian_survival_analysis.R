# Bayesian Survival Analysis Tutorial
# This script demonstrates how to fit a Bayesian proportional hazards model
# using the rstanarm package. It uses the built-in 'lung' dataset from the
# survival package. The example is simplified for teaching purposes and is
# suitable for non-experts who are new to Bayesian survival analysis.

# Load required packages
library(survival)   # For the lung dataset and survival objects
library(rstanarm)   # For Bayesian survival modeling
library(ggplot2)    # For plotting priors and survival curves
library(tidyr)      # For reshaping data for ggplot

# ---------------------------------------------------------------------------
# Example prior specifications
# ---------------------------------------------------------------------------
# Priors help encode your beliefs about how large the regression coefficients
# might be. Below are several common choices with comments on when they might be
# useful. Adjust the "scale" parameter to reflect how strongly you want to
# regularize the coefficients.

# Normal(0, 2.5) prior -- weakly informative
prior_normal <- normal(location = 0, scale = 2.5, autoscale = TRUE)

# Student-t prior with 5 degrees of freedom -- heavier tails than normal
prior_student <- student_t(df = 5, location = 0, scale = 2.5, autoscale = TRUE)

# Laplace (double exponential) prior -- encourages sparsity
prior_laplace <- laplace(location = 0, scale = 1, autoscale = TRUE)

# Cauchy prior (very heavy tails) -- similar to Student-t with df=1
prior_cauchy <- cauchy(location = 0, scale = 2.5, autoscale = TRUE)

# Visualize the shapes of these priors ---------------------------------------
dlaplace <- function(x, mu = 0, b = 1) 1/(2 * b) * exp(-abs(x - mu)/b)

x_vals <- seq(-6, 6, length.out = 400)
prior_df <- data.frame(
  x = x_vals,
  Normal = dnorm(x_vals, 0, 2.5),
  Student_t = dt(x_vals / 2.5, df = 5) / 2.5,
  Laplace = dlaplace(x_vals, 0, 1),
  Cauchy = dcauchy(x_vals, 0, 2.5)
)

prior_long <- pivot_longer(prior_df, -x, names_to = "Prior", values_to = "Density")

ggplot(prior_long, aes(x = x, y = Density, color = Prior)) +
  geom_line(linewidth = 1) +
  labs(title = "Example prior distributions", y = "Density") +
  theme_minimal()

# Set rstan options (speed up by using fewer cores)
options(mc.cores = parallel::detectCores())

# Load the dataset
# The 'lung' dataset contains survival times of patients with advanced lung cancer
# 'time' is the survival time in days and 'status' is the censoring indicator
# (1=censored, 2=dead). We'll convert status to 0/1 format (0=censored, 1=dead).
data(lung)
lung$event <- ifelse(lung$status == 2, 1, 0)

# Fit a Bayesian Cox proportional hazards model using rstanarm
# We'll model survival time with covariates age and sex
# (1=male, 2=female). We'll center and scale age for easier interpretation.
# The prior is set to a Student-t distribution, but you could switch
# to `prior_normal` or `prior_laplace` defined above.
lung$age_z <- scale(lung$age)

fit <- stan_surv(
  formula = Surv(time, status == 2) ~ age_z + sex,
  data = lung,
  basehaz = "bs",  # B-spline baseline hazard
  prior = normal(0, 2.5),
  chains = 4,
  iter = 2000,
  seed = 1234
)

# Summarize the posterior draws
print(fit)

# Plot the posterior distributions for model coefficients
plot(fit)

# Predict survival probabilities for the first 10 patients
# We'll compute the median survival curves based on the posterior draws
pred <- posterior_survfit(fit, newdata = lung[1:10,])

# Plot survival curves for the first 3 patients as an example
plot(pred, row = 1:3, ci = FALSE)

