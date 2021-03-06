% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/multinomial.R
\name{update_S}
\alias{update_S}
\title{Update latent sources S}
\usage{
update_S(pi, m)
}
\description{
The k^th column is updated as
\hat{s}_{\cdot k} &= \frac{\sum_{r = 1}^{p_{1}} n_{r}\pi_{k}^{(r)} m^{(r)}}{\sum_{r = 1}^{p_{1}} n_{r}\pi_{k}^{(r)}}
}

