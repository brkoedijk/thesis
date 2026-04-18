import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm

df = pd.read_csv(
    "master_capacity_factors_2000_2023.csv",
    parse_dates=["valid_time"],
    index_col="valid_time",
)
df = df.resample("D").mean()
print(df.head())


def deseasonalize_site(series):
    t = series.index.dayofyear.values

    df_reg = pd.DataFrame(index=series.index)
    df_reg["const"] = 1.0
    df_reg["sin1"] = np.sin(2 * np.pi * t / 365.25)
    df_reg["cos1"] = np.cos(2 * np.pi * t / 365.25)
    df_reg["sin2"] = np.sin(4 * np.pi * t / 365.25)
    df_reg["cos2"] = np.cos(4 * np.pi * t / 365.25)

    model = sm.OLS(series, df_reg).fit()
    mu_seas = model.predict(df_reg)

    x_tilde = series - mu_seas
    return x_tilde, mu_seas


residuals_df = pd.DataFrame(index=df.index)
seasonal_means_df = pd.DataFrame(index=df.index)

for col in df.columns:
    resid, mu_seas = deseasonalize_site(df[col])

    residuals_df[col] = resid
    seasonal_means_df[col] = mu_seas


def plot_deseasonalization(df_raw, df_resid, df_mu, site_name="Site_01"):
    # Create a 2-row plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

    # Zoom in on a specific period to see the sine waves clearly
    start_date = "2020-01-01"
    end_date = "2021-12-31"

    # Plot 1: Raw Capacity Factor vs Seasonal Trend
    ax1.plot(
        df_raw.loc[start_date:end_date, site_name],
        alpha=0.3,
        label="Raw Capacity Factor (X_raw)",
        color="gray",
    )
    ax1.plot(
        df_mu.loc[start_date:end_date, site_name],
        label="Seasonal Mean (mu_seas)",
        color="red",
        linewidth=2,
    )
    ax1.set_title(f"Seasonality Extraction: {site_name}")
    ax1.set_ylabel("Capacity Factor")
    ax1.legend()

    # Plot 2: Deseasonalized Residuals
    ax2.plot(
        df_resid.loc[start_date:end_date, site_name],
        label="Residuals (X_tilde)",
        color="blue",
        alpha=0.7,
    )
    ax2.axhline(0, color="black", linestyle="--")  # Mean should be zero
    ax2.set_title("Deseasonalized Residuals (The input for MOU)")
    ax2.set_ylabel("Deviation from Mean")
    ax2.legend()

    plt.tight_layout()
    plt.savefig("plots/deseasonalization_plot.jpg", dpi=300, bbox_inches="tight")
    plt.show()


plot_deseasonalization(df, residuals_df, seasonal_means_df, "Site_06")

# print(residuals_df['Site_01'].mean()) #

A_diag = np.zeros(15)
simple_residuals = pd.DataFrame(
    index=residuals_df.index[1:], columns=residuals_df.columns
)
for i, col in enumerate(residuals_df.columns):
    y = residuals_df[col].iloc[1:].values
    x = residuals_df[col].iloc[:-1].values

    model = sm.OLS(y, x).fit()
    A_diag[i] = model.params[0]
    simple_residuals[col] = model.resid
print(f"Diagonal A-elements (self-reversion): {A_diag}")

cross_corr = simple_residuals.corrwith(residuals_df.shift(1).iloc[1:].squeeze())

resid_corr = simple_residuals.corr()


# plotting residual cross-autocorrelation
plt.figure(figsize=(8, 6))
sns.heatmap(resid_corr, annot=True, cmap="RdBu_r", fmt=".2f")
plt.title("Residual Cross-Autocorrelation")
plt.savefig("plots/cross-correlation.jpg", dpi=300, bbox_inches="tight")
plt.show()

from sklearn.linear_model import LassoCV

Y = residuals_df.iloc[1:].values.astype(float)
X = residuals_df.iloc[:-1].values.astype(float)
A_sparse = np.zeros((15, 15))
alphas = []

# for i in range(15):
#     lasso = LassoCV(cv=5, fit_intercept=False).fit(X, Y[:, i])
#     A_sparse[i,:] = lasso.coef_
#     alphas.append(lasso.alpha_)

for i, col in enumerate(residuals_df.columns):
    y = residuals_df[col].iloc[1:].values
    x_self = residuals_df[col].iloc[:-1].values

    X_neighbors = np.delete(X, i, axis=1)

    memory_effect = A_diag[i] * x_self
    y_residual = y - memory_effect

    lasso = LassoCV(cv=5, fit_intercept=False).fit(X_neighbors, y_residual)
    row = np.insert(lasso.coef_, i, A_diag[i])
    A_sparse[i, :] = row
    alphas.append(lasso.alpha_)

A_sparse_df = pd.DataFrame(
    A_sparse, index=residuals_df.columns, columns=residuals_df.columns
)

plt.figure(figsize=(8, 6))
sns.heatmap(A_sparse_df, annot=True, cmap="RdBu_r", center=0, fmt=".2f")
plt.title("Sparse transition matrix A (Lasso penalized)")
plt.xlabel("Lagged variables (t-1)")
plt.ylabel("Current variables (t)")
plt.savefig("plots/sparse_a_mat.jpg", dpi=300, bbox_inches="tight")
plt.show()

print(f"Average alpha (penalty strength): {np.mean(alphas):.6f}")

from scipy.linalg import logm

theta = -logm(A_sparse)
# plt.figure(figsize=(8,6))
# sns.heatmap(theta, cmap="RdBu_r", center=0, fmt='.2f')
# plt.title("Theta)")
# plt.xlabel("Lagged variables (t-1)")
# plt.ylabel("Current variables (t)")
# plt.show()

gamma_inf = residuals_df.cov().values
Q = theta @ gamma_inf + gamma_inf @ theta.T
Q_discrete = gamma_inf - (A_sparse @ gamma_inf @ A_sparse.T)
sigma = np.linalg.cholesky(Q)
sigma_discrete = np.linalg.cholesky(Q_discrete + np.eye(15) * 1e-8)

v = np.sqrt(np.diag(Q))
outer_v = np.outer(v, v)
shock_corr = Q / outer_v
# plt.figure(figsize=(8,6))
# sns.heatmap(shock_corr, annot=True, cmap='RdBu_r', fmt='.2f')
# plt.title("instantaneous shock correlation")
# plt.show()


def simulate_mou(theta, A_sparse, sigma, initial_state, steps, dt=1):
    """
    Simulate MOU
    """
    n_sites = len(initial_state)
    path = np.zeros((steps, n_sites))
    path[0] = initial_state

    for t in range(1, steps):
        # pull = - (theta @ path[t-1]) * dt

        # push = sigma @ np.random.normal(size=n_sites) * np.sqrt(dt)
        # path[t] = path[t-1] + pull + push
        path[t] = A_sparse @ path[t - 1] + sigma @ np.random.normal(size=n_sites)

    return path


n_steps = 1000

start_val = residuals_df.iloc[0].values

synthetic_wind = simulate_mou(theta, A_sparse, sigma_discrete, start_val, n_steps)

real_slice = residuals_df.iloc[:1000, [4, 9]]
fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True, sharey=True)

axes[0].plot(real_slice.values[:, 0], label="real site 4", color="blue", alpha=0.7)
axes[0].plot(real_slice.values[:, 1], label="real site 9", color="orange", alpha=0.7)
axes[0].set_title("Historical residuals")
axes[0].legend()

axes[1].plot(synthetic_wind[:, 4], label="synthetic site 4", ls="--", color="blue")
axes[1].plot(synthetic_wind[:, 9], label="synthetic site 9", ls="--", color="orange")
axes[1].set_title("MOU generated path")
axes[1].legend()
plt.xlabel("days")
plt.tight_layout()
plt.savefig("plots/real_vs_synthetic.jpg", dpi=300, bbox_inches="tight")
plt.show()

# Calculate standard deviation for each site
real_std = real_slice.std().values
synthetic_std = np.std(synthetic_wind, axis=0)

# Print comparison for Site 4 and Site 9
print(f"Site 4 - Real Std: {real_std[0]:.4f} | Synthetic Std: {synthetic_std[4]:.4f}")
print(f"Site 9 - Real Std: {real_std[1]:.4f} | Synthetic Std: {synthetic_std[9]:.4f}")


from statsmodels.graphics.tsaplots import plot_acf

fig, axes = plt.subplots(1, 2, figsize=(10, 5))
plot_acf(residuals_df.iloc[:, 4], ax=axes[0], lags=24, title="real site 4 memory")
plot_acf(synthetic_wind[:, 4], ax=axes[1], lags=24, title="synthethic site 4 memory")
plt.savefig("plots/acf_plot.jpg", dpi=300, bbox_inches="tight")
plt.show()

w = np.ones(15) / 15
A_agg = w.T @ A_sparse @ w

Q_discrete_full = sigma_discrete @ sigma_discrete.T
var_agg = w.T @ Q_discrete_full @ w
sigma_agg = np.sqrt(var_agg)


np.savez(
    "nwe_wind_params.npz",
    A=A_sparse,
    A_agg=A_agg,
    sigma=sigma_discrete,
    sigma_agg=sigma_agg,
    w=w,
)
