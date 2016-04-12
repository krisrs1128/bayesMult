
################################################################################
# Bayesian multitask regression: Multinomial sources (e.g., clustered
# coefficients)
#
# Method from "Flexible Latent Variable Models for Multi-Task Learning"
################################################################################

multinom_entropy <- function(log_pi_list) {
  R <- length(log_pi_list)
  task_sums <- vector(length = R)
  for(r in seq_len(R)) {
    task_sums[r] <- sum(log_pi_list[[r]] * exp(log_pi_list[[r]]))
  }
  sum(task_sums)
}

#' @title Multinomial Variational Lower Bound
#' @description Compute the variational lower bound, equation (17) in the
#' reference, for the multinomial case.
#' @references Flexible Latent Variable Models for Multi-Task Learning
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

  - N / 2 * log(sigma ^ 2) - lik_multin(x_list, y_list, m_list, V_list) -
    N / 2 * log(det(Psi)) - 1 / 2 * tr(Psi_inv * v_weighted) -
      1 / 2 * mean_source_multinom(n, log_pi_list, m_list, S, Psi_inv) +
      mulitnom_loglik(n, pi_list, phi) +
      1 / 2 * sum_log_det(v_list) - multinom_entropy(log_pi_list)

}
