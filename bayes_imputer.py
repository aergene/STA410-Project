import pandas as pd
import pymc as pm
import numpy as np
import matplotlib.pyplot as plt

class BayesianImputer:
    def __init__(self, data):
        """
        Initialize with a pandas DataFrame containing missing values.
        """
        self.original_data = data.copy()
        self.data = data.copy()
        self.model = None
        self.trace = None
        self.imputed_data = None

    def fit(self, draws=1000, tune=500, target_accept=0.9):
        """
        Fit a Bayesian model to estimate missing values for each numeric column.
        """
        self.model = pm.Model()
        self.trace = {}

        with self.model:
            self.obs_vars = {}
            self.mu_params = {}
            self.sigma_params = {}

            for col in self.data.columns:
                col_data = self.data[col]
                if col_data.dtype.kind not in 'iufc':
                    continue  # skip non-numeric columns for now

                observed_mask = ~col_data.isna()
                observed_data = col_data[observed_mask].values

                mu = pm.Normal(f"{col}_mu", mu=np.mean(observed_data), sigma=10)
                sigma = pm.HalfNormal(f"{col}_sigma", sigma=10)

                # Observed values
                pm.Normal(f"{col}_obs", mu=mu, sigma=sigma, observed=observed_data)

                self.mu_params[col] = mu
                self.sigma_params[col] = sigma

            self.trace = pm.sample(draws=draws, tune=tune, target_accept=target_accept, return_inferencedata=True)

    def impute(self):
        """
        Use posterior samples to impute missing values.
        """
        imputed_df = self.data.copy()
        posterior = self.trace

        for col in self.data.columns:
            col_data = self.data[col]
            if col_data.dtype.kind not in 'iufc':
                continue  # skip non-numeric

            missing_mask = col_data.isna()
            n_missing = missing_mask.sum()

            if n_missing == 0:
                continue

            mu_samples = posterior.posterior[f"{col}_mu"].values.flatten()
            sigma_samples = posterior.posterior[f"{col}_sigma"].values.flatten()
            n_samples = len(mu_samples)

            imputations = []
            for _ in range(n_missing):
                idx = np.random.randint(0, n_samples)
                sample = np.random.normal(mu_samples[idx], sigma_samples[idx])
                imputations.append(sample)

            imputed_df.loc[missing_mask, col] = imputations

        self.imputed_data = imputed_df
        return imputed_df

    def summary(self):
        """
        Print posterior summary.
        """
        if self.trace is not None:
            print(pm.summary(self.trace))
        else:
            print("Model not yet fitted.")

    def plot(self, column):
        """
        Plot original vs. imputed distribution for a given column.
        """
        if self.imputed_data is None:
            print("You need to call impute() first.")
            return

        plt.figure(figsize=(10, 5))
        plt.hist(self.original_data[column].dropna(), bins=30, alpha=0.5, label='Original (observed)')
        plt.hist(self.imputed_data[column], bins=30, alpha=0.5, label='Imputed (full)')
        plt.legend()
        plt.title(f'Distribution of {column} (before and after imputation)')
        plt.show()
