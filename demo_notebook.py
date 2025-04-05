# This script demonstrates the use of the BayesianImputer module
# on synthetic data with missing values.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from bayes_imputer import BayesianImputer

# Set random seed for reproducibility
np.random.seed(42)

# ----------------------------
# Step 1: Simulate Data
# ----------------------------
n = 200
age = np.random.normal(30, 5, size=n)
income = age * 1000 + np.random.normal(0, 5000, size=n)

# Introduce missing values at random
age_missing = age.copy()
income_missing = income.copy()

missing_indices_age = np.random.choice(n, size=20, replace=False)
missing_indices_income = np.random.choice(n, size=25, replace=False)

age_missing[missing_indices_age] = np.nan
income_missing[missing_indices_income] = np.nan

# Create DataFrame with missing values
df_missing = pd.DataFrame({
    'age': age_missing,
    'income': income_missing
})

print("Missing values in each column:")
print(df_missing.isna().sum())
print("\nPreview of the dataset:")
print(df_missing.head())

# ----------------------------
# Step 2: Visualize Missingness
# ----------------------------
sns.heatmap(df_missing.isnull(), cbar=False, yticklabels=False)
plt.title("Missing Data Heatmap")
plt.show()

# ----------------------------
# Step 3: Fit the Imputer
# ----------------------------
print("\nFitting the Bayesian imputer...")
imputer = BayesianImputer(df_missing)
imputer.fit(draws=1000, tune=500)

# ----------------------------
# Step 4: Perform Imputation
# ----------------------------
print("\nPerforming imputation...")
df_imputed = imputer.impute()
print(df_imputed.head())

# ----------------------------
# Step 5: Visualize Imputed vs. Observed
# ----------------------------
print("\nGenerating plots...")
imputer.plot('age')
imputer.plot('income')

# ----------------------------
# Step 6: Posterior Summary
# ----------------------------
print("\nPosterior summary:")
imputer.summary()

# ----------------------------
# Step 7: Comparison with Mean Imputation
# ----------------------------
df_mean_imputed = df_missing.fillna(df_missing.mean())

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].hist(df_mean_imputed['age'], bins=30, alpha=0.6, label='Mean Imputation', edgecolor='black')
axes[0].hist(df_imputed['age'], bins=30, alpha=0.6, label='Bayesian', edgecolor='black')
axes[0].legend()
axes[0].set_title("Age: Mean vs Bayesian Imputation")

axes[1].hist(df_mean_imputed['income'], bins=30, alpha=0.6, label='Mean Imputation', edgecolor='black')
axes[1].hist(df_imputed['income'], bins=30, alpha=0.6, label='Bayesian', edgecolor='black')
axes[1].legend()
axes[1].set_title("Income: Mean vs Bayesian Imputation")

plt.tight_layout()
plt.show()
