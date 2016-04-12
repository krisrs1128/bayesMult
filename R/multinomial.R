
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
gsn_loglik <- function(x_list, y_list, m_list, v_list, sigma) {
  R <- length(x_list)
  task_sums <- vector(length = R)
  for (r in seq_len(R)) {
    bias2 <- sum( (y_list[[r]] - x_list[[r]] %*% m_list[[r]]) ^ 2 )
    variance <- sum( diag(v_list[[r]] %*% crossprod(x_list[[r]])) )
    task_sums[r] <- bias2 + variance
  }
  (1 / (2 * sigma ^ 2)) * sum(task_sums)
}

#' @title Compute deviation between variational means and latent sources
#' @description Computes the term
#' \sum_{r = 1}^{R}\sum_{k = 1}^{K} n_{r}\pi_{k}^{(r)} *
#' (m^{(r)} - s_{\cdot k})^{T}\Psi^{-1}(m^{(r)} - s_{\cdot k})
mean_source_multinom <- function(n, log_pi_list, m_list, S, Psi_inv) {
  R <- length(r)
  K <- ncol(S)
  vals <- matrix(0, R, K)
  for (r in seq_len(R)) {
    for (k in seq_len(K)) {
      pi_rk <- exp(log_pi_list[[r]][k])
      vals[r, k]  <- n[r] *  pi_rk *
        t(m_list[[r]] - S[, k]) %*% Psi_inv %*% (m_list[[r]] - S[, k])
    }
  }
  sum(vals)
}

#' @title Calculate expected multinomial likelihood term in posterior
#' @description Computes
#' sum_{r = 1}^{R} \sum_{k = 1}^{K} n_{r}pi_{k}^{(r)} \log \varphi_{k}
mulitnom_loglik <- function(n, log_pi_list, phi) {
  task_sums <- vector(length = length(n))
  for (r in seq_along(n)) {
    task_sums[r] <- n[r] * sum( exp(log_pi_list[[r]]) * log(phi) )
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
multinom_entropy <- function(log_pi_list) {
  R <- length(log_pi_list)
  task_sums <- vector(length = R)
  for (r in seq_len(R)) {
    task_sums[r] <- sum(log_pi_list[[r]] * exp(log_pi_list[[r]]))
  }
  sum(task_sums)
}

#' @title Multinomial Variational Lower Bound
#' @description Compute the variational lower bound, equation (17) in the
#' reference, for the multinomial case.
#' @references Flexible Latent Variable Models for Multi-Task Learning
#' @importFrom magrittr %>%
#' @export
vb_bound_multin <- function(data_list, var_list, param_list) {

  # retrieve data and parameters
  x_list <- data_list$x_list
  y_list <- data_list$y_list
  n <- sapply(x_list, nrow)
  N <- sum(n)

  # retrieve variational and fixed parameters
  m_list <- var_list$m_list
  v_list <- var_list$v_list
  log_pi_list <- var_list$log_pi_list
  S <- param_list$S
  Psi <- param_list$Psi
  phi <- param_list$phi
  sigma <- param_list$sigma
  Psi_inv <- solve(Psi)

  v_weighted <- lapply(seq_along(n), function(i) n[i] * v_list[[i]] ) %>%
    abind(along = 3) %>%
    apply(c(1, 2), sum)

  (1 / N) * (- N / 2 * log(sigma ^ 2) -
    gsn_loglik(x_list, y_list, m_list, v_list, sigma) -
    N / 2 * log(det(Psi)) - 1 / 2 * sum(diag(Psi_inv * v_weighted)) -
      1 / 2 * mean_source_multinom(n, log_pi_list, m_list, S, Psi_inv) +
      mulitnom_loglik(n, log_pi_list, phi) +
      1 / 2 * sum_log_det(v_list) - multinom_entropy(log_pi_list))
}
