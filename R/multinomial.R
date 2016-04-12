
################################################################################
# Bayesian multitask regression: Multinomial sources (e.g., clustered
# coefficients)
#
# Method from "Flexible Latent Variable Models for Multi-Task Learning"
################################################################################

#' @title Likelihood contribution to VB expected posterior
#' @description Computes the term
#'  \sum_{r = 1}^{R} \sum_{i = 1}^{n_{r}} 1 / 2 * sigma ^ 2 ((y_{i}^{(r)} - x_{i}^{(r) T}m^{(r)})^2 + x_{i}^{(r) T} V^{(r)} x_{i}^{(r)}
#' @export
gsn_loglik <- function(x_list, y_list, m, v_list, sigma) {
  R <- length(x_list)
  task_sums <- vector(length = R)
  for (r in seq_len(R)) {
    bias2 <- sum( (y_list[[r]] - x_list[[r]] %*% m[, r]) ^ 2 )
    variance <- sum( diag(v_list[[r]] %*% crossprod(x_list[[r]])) )
    task_sums[r] <- bias2 + variance
  }
  (1 / (2 * sigma ^ 2)) * sum(task_sums)
}

#' @title Compute deviation between variational means and latent sources
#' @description Computes the term
#' \sum_{r = 1}^{R}\sum_{k = 1}^{K} n_{r}\pi_{k}^{(r)} *
#' (m^{(r)} - s_{\cdot k})^{T}\Psi^{-1}(m^{(r)} - s_{\cdot k})
mean_source_multinom <- function(n, log_pi, m, S, Psi_inv) {
  R <- length(r)
  K <- ncol(S)
  vals <- matrix(0, R, K)
  for (r in seq_len(R)) {
    for (k in seq_len(K)) {
      pi_rk <- exp(log_pi[k, r])
      vals[r, k]  <- n[r] *  pi_rk *
        t(m[, r] - S[, k]) %*% Psi_inv %*% (m[, r] - S[, k])
    }
  }
  sum(vals)
}

#' @title Calculate expected multinomial likelihood term in posterior
#' @description Computes
#' sum_{r = 1}^{R} \sum_{k = 1}^{K} n_{r}pi_{k}^{(r)} \log \varphi_{k}
mulitnom_loglik <- function(n, log_pi, phi) {
  task_sums <- vector(length = length(n))
  for (r in seq_along(n)) {
    task_sums[r] <- n[r] * sum( exp(log_pi[, r]) * log(phi) )
  }
  sum(task_sums)
}

#' @title Compute sum of log determinants
#' @description \sum log|V^{(r)}|
#' Maybe it would be smarter to do a cholesky factorization sort of things.
sum_log_det <- function(v_list) {
  task_sums <- vector(length = length(v_list))
  for (r in seq_along(v_list)) {
    task_sums[r] <- log(det(v_list[[r]]))
  }
  sum(task_sums)
}

#' @title Cross-Tasks multinomial entropy
#' @description Say r indexes tasks and k indexes latent source clusters.
#' Then, this term is 
#' sum_{r = 1}^{R} \sum_{k = 1}^{K} pi_{k}^{(r)} \log(\pi_{k}^{(r)})
#'
#' I really should be using the log-sum-exp trick, but hopefully pi
#' never gets too small in these preliminary experiments.
#' @export
multinom_entropy <- function(log_pi) {
  R <- ncol(log_pi)
  task_sums <- vector(length = R)
  for (r in seq_len(R)) {
    task_sums[r] <- sum(log_pi[, r] * exp(log_pi[, r]))
  }
  sum(task_sums)
}

#' @title Multinomial Variational Lower Bound
#' @description Compute the variational lower bound, equation (17) in the
#' reference, for the multinomial case.
#' @references Flexible Latent Variable Models for Multi-Task Learning
#' @importFrom magrittr %>%
#' @export
vb_bound_multinom <- function(data_list, var_list, param_list) {

  # retrieve data and parameters
  x_list <- data_list$x_list
  y_list <- data_list$y_list
  n <- sapply(x_list, nrow)
  N <- sum(n)

  # retrieve variational and fixed parameters
  m <- var_list$m
  v_list <- var_list$v_list
  log_pi <- var_list$log_pi
  S <- param_list$S
  Psi <- param_list$Psi
  phi <- param_list$phi
  sigma <- param_list$sigma
  Psi_inv <- solve(Psi)

  v_weighted <- lapply(seq_along(n), function(i) n[i] * v_list[[i]] ) %>%
    abind(along = 3) %>%
    apply(c(1, 2), sum)

  (1 / N) * (- N / 2 * log(sigma ^ 2) -
    gsn_loglik(x_list, y_list, m, v_list, sigma) -
    N / 2 * log(det(Psi)) - 1 / 2 * sum(diag(Psi_inv * v_weighted)) -
      1 / 2 * mean_source_multinom(n, log_pi, m, S, Psi_inv) +
      mulitnom_loglik(n, log_pi, phi) +
      1 / 2 * sum_log_det(v_list) - multinom_entropy(log_pi))
}

# E-step -----------------------------------------------------------------------

#' @title Update a single V^{(r)}
#' @description 1 / n_{r}  * (\Psi^{-1} + \frac{1}{n_{r}\sigma^{2}}\sum_{i = 1}^{n_{r}}x_{i}^{(r)}x_{i}^{(r) T})^{-1}
update_v <- function(x, Psi, sigma) {
  n <- nrow(x)
  1 / n * solve(solve(Psi) + 1 / (n * (sigma ^ 2)) * crossprod(x))
}

#' @title Update variational mean parameters
#' @description
#' \hat{m}^{(r)} &= \left(\frac{1}{\sigma^{2}}X^{(r) T}X^{(r)} + n_{r}\Psi^{-1}\right)^{-1}\left(\frac{1}{\sigma^{2}}X^{(r) T}y^{(r)} + n_{r}\Psi^{-1} S \pi^{(r)}\right)
#' @export
update_m <- function(x, y, Psi_inv, S, log_pi, sigma) {
  n <- nrow(x)
  A <- (1 / sigma ^ 2) * crossprod(x) + n * Psi_inv
  B <- 1 / (sigma ^ 2) * t(x) %*% y + n * Psi_inv %*% S %*% exp(log_pi)
  solve(A, B)
}

#' @title Update the variational clustering parameters
#' @description \hat{\pi}_{k}^{(r)} &\propto \varphi_{k}\exp{\left(m^{(r)} - s_{\cdot k}\right)^{T}\Psi^{-1}\left(m^{(r)} - s_{\cdot k}\right)}
update_log_pi <- function(m, S, Psi_inv, phi) {
  K <- ncol(S)
  log_pi_unnorm <- vector(length = K)
  for (k in seq_len(K)) {
    log_pi_unnorm[k] <- log(phi[k]) + t(m - S[, k]) %*% Psi_inv %*% (m - S[, k])
  }
  log_pi_unnorm - log(sum(exp(log_pi_unnorm)))
}

# M-step -----------------------------------------------------------------------

#' @title Update the multinomial clustering probabilities
#' @description
#' \varphi_{k} &\propto \sum_{r = 1}^{p_{1}} n_{r}\pi_{k}^{(r)}
#' @export
update_phi <- function(n, log_pi) {
  R <- length(n)
  K <- nrow(log_pi)
  phi_unnorm <- matrix(0, R, K)
  for (r in seq_len(R)) {
    phi_unnorm[r, ] <- n[r] * exp(log_pi[, r])
  }
  phi_unnorm <- colSums(phi_unnorm)
  phi_unnorm / sum(phi_unnorm)
}

#' @title Update unshared covariance matrix Psi
#' @description
#' \hat{Psi} &= \frac{1}{N}\sum_{r = 1}^{p_{1}}n_{r}\left(V^{(r)} + \sum_{k = 1}^{K}  \pi_{k}^{(r)}\left(m^{(r)} - s_{\cdot k}\right)\left(m^{(r)} - s_{\cdot k}\right)^{T}\right)
#' @export
update_Psi <- function(n, v_list, m, S, log_pi) {
  R <- length(n)
  K <- ncol(S)
  p <- nrow(v_list[[1]])
  task_sums <- array(0, c(p, p, R))
  for (r in seq_len(R)) {
    m_offset <- array(0, c(p, p, K))
    for(k in seq_len(K)) {
      pi_rk <- exp(log_pi[k, r])
      m_offset[,, k] <- pi_rk * (m[, r] - S[, k]) %*% t(m[, r] - S[, k])
    }
    task_sums[,, r] <- n[r] * (v_list[[r]] + apply(m_offset, c(1, 2), sum))
  }
  apply(task_sums, c(1, 2), sum) / sum(n)
}

#' @title Update latent sources S
#' @description The k^th column is updated as
#' \hat{s}_{\cdot k} &= \frac{\sum_{r = 1}^{p_{1}} n_{r}\pi_{k}^{(r)} m^{(r)}}{\sum_{r = 1}^{p_{1}} n_{r}\pi_{k}^{(r)}}
#' @export
update_S <- function(n, log_pi, m) {
  K <- nrow(log_pi)
  p <- nrow(m)
  R <- length(n)
  S <- matrix(0, p, K)
  for(k in seq_len(K)) {
    num <- matrix(0, nrow = p, ncol = R)
    denom <- vector(length = R)
    for(r in seq_len(R)) {
      denom[r] <- n[r] * exp(log_pi[k, r])
      num[, r] <- denom[r] * m[, r]
    }
    S[, k] <- (1 / sum(denom)) * rowSums(num)
  }
  S
}

# EM-update --------------------------------------------------------------------

#' @title Variational Iteration for Multinomially Clustered Regressions
#' @export
vb_multinom <- function(data_list, param_list, n_iter = 100) {
  R <- length(data_list$x_list)
  p <- ncol(data_list$x_list[[1]])
  K <- ncol(param_list$S)
  elbo <- vector(length = n_iter)

  n <- sapply(data_list$x_list, nrow)
  log_pi_init <- matrix(rgamma(K * R, 1, 1), K, R)
  log_pi_init <- log(log_pi_init %*% diag(1 / colSums(log_pi_init)))
  var_list <- list(m = matrix(0, p, R),
                   v_list = list(),
                   log_pi_list = log_pi_init)

  for (iter in seq_len(n_iter)) {

    # E-step
    for (r in seq_len(R)) {
      var_list$v_list[[r]] <- update_v(
        data_list$x_list[[r]],
        Psi,
        param_list$sigma
      )
      var_list$m[, r] <- update_m(
        data_list$x_list[[r]],
        data_list$y_list[[r]],
        solve(param_list$Psi),
        param_list$S,
        var_list$log_pi[, r],
        param_list$sigma
      )
      var_list$log_pi[, r] <- update_log_pi(
        var_list$m[, r],
        param_list$S,
        solve(param_list$Psi),
        param_list$phi
      )
    }

    # M-step
    param_list$phi <- update_phi(
      n,
      var_list$log_pi
    )

    param_list$Psi <- update_Psi(
      n,
      var_list$v_list,
      var_list$m,
      param_list$S,
      var_list$log_pi
    )

    param_list$S <- update_S(
      n,
      var_list$log_pi,
      var_list$m
    )

    # calculate evidence lower bound
    elbo[iter] <- vb_bound_multinom(data_list, var_list, param_list)
  }
  list(param_list = param_list, var_list = var_list, elbo = elbo)
}
