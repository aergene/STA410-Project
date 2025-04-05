# BayesianImputer

A lightweight Python module for **Bayesian Missing Data Imputation** using probabilistic modeling and MCMC sampling.

This tool estimates missing values in numeric tabular datasets by modeling each variable as a distribution and inferring the posterior over missing entries using PyMC.

---

## Features

- Bayesian imputation using normal priors and MCMC sampling.
- Handles missing values in numeric (continuous) columns.
- Provides posterior summaries and visual comparison plots.
- Designed for easy extension and experimentation.

---

## Installation

Clone this repo and install dependencies:

```bash
pip install -r requirements.txt
