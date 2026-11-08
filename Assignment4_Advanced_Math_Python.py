#!/usr/bin/env python
# coding: utf-8

# # Assignment 4: Advanced Applications in Math and Python
# 
# **Name:** G Anirudh
# **Register No.:** RA2411003012249
# 
# | Part | Task | Topics |
# |---|---|---|
# | 1.1 | Covariance and correlation matrices | Linear algebra, statistics |
# | 1.2 | Markov chain weather model | Probability, eigenvectors |
# | 2.1 | Time-series windowing | Pandas rolling and expanding windows, NaN handling |
# | 2.2 | Subplots with conditional colour and shading | Matplotlib, empirical rule |

# In[1]:


import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

np.random.seed(7)
pd.set_option("display.precision", 4)
plt.rcParams["figure.dpi"] = 110
RESULTS = {}   # collected for the dashboard


# ## Part 1.1: Covariance and Correlation Matrices
# 
# Three simulated sensor channels (temperature in °C, relative humidity in %, pressure in hPa),
# 100 readings each, drawn from a multivariate normal distribution. A 3×100 matrix `X` holds
# one variable per row.
# 
# The sample covariance matrix comes straight from matrix multiplication:
# 
# $$C = \frac{1}{n-1}\,(X - \bar{X})(X - \bar{X})^{T}$$
# 
# where $\bar{X}$ is each row's mean, broadcast across its 100 columns. The result is 3×3.

# In[2]:


names = ["Temperature", "Humidity", "Pressure"]
true_mean = np.array([30.0, 65.0, 1008.0])
true_cov = np.array([[ 4.0, -6.0, -1.5],
                     [-6.0, 25.0,  2.0],
                     [-1.5,  2.0,  3.0]])

X = np.random.multivariate_normal(true_mean, true_cov, size=100).T   # shape (3, 100)
print("X shape:", X.shape)
print("First 5 readings of each sensor:")
print(np.round(X[:, :5], 2))


# In[3]:


n = X.shape[1]
row_means = X.mean(axis=1, keepdims=True)    # (3, 1)
Xc = X - row_means                            # centre each variable
C_manual = (Xc @ Xc.T) / (n - 1)              # (3,100) @ (100,3) -> (3,3)

print("Covariance matrix (manual, matrix multiplication):")
print(np.round(C_manual, 4))


# Verification with the built-in function:

# In[4]:


C_numpy = np.cov(X)          # rows are variables by default
diff = np.abs(C_manual - C_numpy).max()
print("np.cov(X):")
print(np.round(C_numpy, 4))
print(f"\nLargest absolute difference: {diff:.2e}")
print("Match:", np.allclose(C_manual, C_numpy))


# The correlation matrix rescales covariance by each variable's standard deviation:
# $R = D^{-1} C D^{-1}$, where $D$ is the diagonal matrix of standard deviations.

# In[5]:


D_inv = np.diag(1 / np.sqrt(np.diag(C_manual)))
R_manual = D_inv @ C_manual @ D_inv
R_numpy = np.corrcoef(X)
print("Correlation matrix (manual):")
print(np.round(R_manual, 4))
print("Match with np.corrcoef:", np.allclose(R_manual, R_numpy))


# In[6]:


fig, axes = plt.subplots(1, 2, figsize=(10, 4))
for ax, M, title, cmap in [(axes[0], C_manual, "Covariance", "viridis"),
                           (axes[1], R_manual, "Correlation", "RdBu_r")]:
    kw = dict(vmin=-1, vmax=1) if title == "Correlation" else {}
    im = ax.imshow(M, cmap=cmap, **kw)
    ax.set_xticks(range(3), names); ax.set_yticks(range(3), names)
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center",
                    color="white" if title == "Covariance" and M[i, j] < M.max() * 0.6 else "black")
    ax.set_title(f"{title} matrix")
    fig.colorbar(im, ax=ax, shrink=0.8)
plt.tight_layout()
plt.savefig("part1_cov_corr.png")
plt.show()

RESULTS["cov"] = {
    "names": names,
    "means": X.mean(axis=1).round(3).tolist(),
    "manual": C_manual.round(4).tolist(),
    "numpy": C_numpy.round(4).tolist(),
    "max_diff": float(diff),
    "corr": R_manual.round(4).tolist(),
    "sample": X[:, :8].round(2).tolist(),
}


# ## Part 1.2: Markov Chain (Sunny, Cloudy, Rainy)
# 
# Row *i* of the transition matrix gives the probabilities of tomorrow's weather when today is
# state *i*. Each row sums to 1.

# In[7]:


states = ["Sunny", "Cloudy", "Rainy"]
P = np.array([[0.70, 0.20, 0.10],
              [0.30, 0.40, 0.30],
              [0.20, 0.30, 0.50]])
assert np.allclose(P.sum(axis=1), 1)
pd.DataFrame(P, index=[f"from {s}" for s in states], columns=[f"to {s}" for s in states])


# ### Simulating 50 steps from Sunny

# In[8]:


rng = np.random.default_rng(42)
steps = 50
current = 0                     # start Sunny
path = [current]
for _ in range(steps):
    current = rng.choice(3, p=P[current])   # next state drawn from the current row
    path.append(current)

path = np.array(path)
print("Day 0 (start):", states[path[0]])
print("Sequence:", " ".join(states[s][0] for s in path))
counts = np.bincount(path[1:], minlength=3)
for s, c in zip(states, counts):
    print(f"{s:<7}: {c:2d} of {steps} steps ({c/steps:.2%})")


# ### Steady state from eigenvectors
# 
# The steady-state row vector $\pi$ satisfies $\pi P = \pi$. Transposing gives
# $P^{T}\pi^{T} = \pi^{T}$, so $\pi^{T}$ is the eigenvector of $P^{T}$ with eigenvalue 1,
# scaled so its entries sum to 1.

# In[9]:


eigvals, eigvecs = np.linalg.eig(P.T)
print("Eigenvalues of P^T:", np.round(eigvals.real, 4))

k = np.argmin(np.abs(eigvals - 1))          # eigenvalue closest to 1
v = eigvecs[:, k].real
pi = v / v.sum()                             # normalise to a probability vector

for s, p in zip(states, pi):
    print(f"P({s}) = {p:.4f}")
print("Check pi @ P == pi:", np.allclose(pi @ P, pi))


# Cross-check: raising $P$ to a high power makes every row converge to $\pi$, and a long
# simulation's state frequencies approach it as well.

# In[10]:


print("P^50 (every row approaches pi):")
print(np.round(np.linalg.matrix_power(P, 50), 4))

long = [0]
for _ in range(100_000):
    long.append(rng.choice(3, p=P[long[-1]]))
long_freq = np.bincount(long, minlength=3) / len(long)
print("\nFrequencies over 100,000 steps:", np.round(long_freq, 4))

# probability of each state over the first 15 days, starting Sunny
dist = [np.array([1.0, 0.0, 0.0])]
for _ in range(15):
    dist.append(dist[-1] @ P)
dist = np.array(dist)

RESULTS["markov"] = {
    "states": states, "P": P.tolist(), "path": path.tolist(),
    "counts": counts.tolist(),
    "eigvals": np.round(eigvals.real, 4).tolist(),
    "eigvec_raw": np.round(v, 4).tolist(),
    "pi": np.round(pi, 4).tolist(),
    "long_freq": np.round(long_freq, 4).tolist(),
    "dist": np.round(dist, 4).tolist(),
}


# ## Part 2.1: Time-Series Windowing
# 
# 365 days of simulated daily temperature for calendar year 2025: a seasonal cycle plus noise,
# with a handful of injected extreme days so the outlier check in Part 2.2 has something to find.

# In[11]:


idx = pd.date_range("2025-01-01", "2025-12-31", freq="D")
assert len(idx) == 365
t = np.arange(365)
season = 29 + 4.5 * np.sin(2 * np.pi * (t - 80) / 365)
noise = np.random.normal(0, 1.4, 365)
temp = season + noise
spike_days = np.random.choice(365, 6, replace=False)
temp[spike_days] += np.random.choice([-1, 1], 6) * np.random.uniform(4, 6, 6)

ts = pd.DataFrame({"value": temp.round(2)}, index=idx)
ts.index.name = "date"
print(type(ts.index).__name__, ts.index.min().date(), "to", ts.index.max().date(), f"({len(ts)} rows)")
ts.head()


# In[12]:


ts["roll_mean_7"] = ts["value"].rolling(window=7).mean()
ts["exp_std_30"] = ts["value"].expanding(min_periods=30).std()

nan_before = ts.isna().sum()
print("NaNs created by the window calculations:")
print(nan_before)
ts.iloc[[0, 5, 6, 28, 29, 30]]


# ### Handling the NaNs
# 
# The 7-day rolling mean has no value for the first 6 days (fewer than 7 points), and the
# expanding standard deviation has none for the first 29 days (`min_periods=30`). Dropping those
# rows would lose a month of data, so each column gets a rule:
# 
# 1. **Rolling mean:** fill the first 6 days with the mean of the days available so far
#    (`expanding().mean()`), which is the same calculation on a shorter window.
# 2. **Expanding std:** back-fill the first 29 days with the first valid value (day 30), since
#    the spread estimate is unstable on fewer points.
# 3. A boolean column `is_filled` records which rows were filled, so nothing is hidden.

# In[13]:


ts["is_filled"] = ts[["roll_mean_7", "exp_std_30"]].isna().any(axis=1)
ts["roll_mean_7"] = ts["roll_mean_7"].fillna(ts["value"].expanding().mean())
ts["exp_std_30"] = ts["exp_std_30"].bfill()

nan_after = ts.isna().sum()
print("NaNs after handling:")
print(nan_after)
print("Rows flagged as filled:", int(ts["is_filled"].sum()))
ts.iloc[[0, 5, 6, 28, 29, 30]].round(3)


# ## Part 2.2: Subplots with Conditional Colour and Shading
# 
# **Empirical rule:** for normally distributed data about 95.4% of points fall within two
# standard deviations of the mean, so roughly 4.6% (about 17 of 365 days) would be expected
# outside. Points beyond $\mu \pm 2\sigma$ are coloured red.

# In[14]:


mu, sigma = ts["value"].mean(), ts["value"].std()
lo, hi = mu - 2 * sigma, mu + 2 * sigma
ts["outlier"] = (ts["value"] < lo) | (ts["value"] > hi)
n_out = int(ts["outlier"].sum())
print(f"mean = {mu:.3f}, std = {sigma:.3f}")
print(f"2-sigma band: [{lo:.3f}, {hi:.3f}]")
print(f"Points outside: {n_out} of 365 ({n_out/365:.2%}); expected about {0.0455*365:.1f} (4.55%)")


# Only 6 days fall outside the band, fewer than the 4.55% the empirical rule predicts. The rule
# assumes a single bell curve, but this series has a seasonal cycle: the summer-winter swing
# inflates σ, so the band is wide and only the injected extreme days cross it. The same effect
# widens the shaded band in the second subplot, because the expanding standard deviation keeps
# absorbing more of the seasonal swing as the year goes on.

# In[15]:


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5.2))

colors = np.where(ts["outlier"], "#C8325A", "#2F6F8F")
ax1.scatter(ts.index, ts["value"], c=colors, s=np.where(ts["outlier"], 34, 14),
            alpha=0.85, edgecolors="none")
ax1.axhline(mu, color="#333", lw=1, label=f"Mean ({mu:.2f})")
ax1.axhline(hi, color="#C8325A", lw=1, ls="--", label=f"Mean ± 2σ ({lo:.2f}, {hi:.2f})")
ax1.axhline(lo, color="#C8325A", lw=1, ls="--")
ax1.scatter([], [], c="#2F6F8F", s=14, label="Within 2σ")
ax1.scatter([], [], c="#C8325A", s=34, label=f"Outside 2σ ({n_out} days)")
ax1.set_title("Daily readings with 2σ outliers highlighted")
ax1.set_xlabel("Date"); ax1.set_ylabel("Temperature (°C)")
ax1.legend(fontsize=8, loc="lower center", ncol=2)

ax2.plot(ts.index, ts["value"], color="#9AA7B0", lw=0.8, label="Raw daily data")
ax2.plot(ts.index, ts["roll_mean_7"], color="#2F6F8F", lw=2, label="7-day rolling mean")
ax2.fill_between(ts.index, ts["roll_mean_7"] - ts["exp_std_30"],
                 ts["roll_mean_7"] + ts["exp_std_30"], color="#2F6F8F", alpha=0.18,
                 label="Rolling mean ± expanding std")
ax2.set_title("Raw data, 7-day rolling mean and ±1 std band")
ax2.set_xlabel("Date"); ax2.set_ylabel("Temperature (°C)")
ax2.legend(fontsize=8, loc="lower center")

for ax in (ax1, ax2):
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    ax.grid(alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)

plt.tight_layout()
plt.savefig("part2_subplots.png", dpi=150)
plt.show()


# In[16]:


RESULTS["ts"] = {
    "dates": ts.index.strftime("%Y-%m-%d").tolist(),
    "value": ts["value"].round(2).tolist(),
    "roll": ts["roll_mean_7"].round(3).tolist(),
    "std": ts["exp_std_30"].round(3).tolist(),
    "outlier": ts["outlier"].tolist(),
    "nan_before": {k: int(v) for k, v in nan_before.items()},
    "nan_after": {k: int(nan_after[k]) for k in nan_before.index},
    "filled_rows": int(ts["is_filled"].sum()),
    "mu": round(mu, 3), "sigma": round(sigma, 3), "lo": round(lo, 3), "hi": round(hi, 3),
    "n_out": n_out,
}
with open("results.json", "w") as f:
    json.dump(RESULTS, f)
ts.to_csv("timeseries_2025.csv")
print("Saved results.json and timeseries_2025.csv")

