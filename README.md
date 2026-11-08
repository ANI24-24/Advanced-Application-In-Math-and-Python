# Advanced Applications in Math and Python

Assignment 4 by G Anirudh. It covers covariance and correlation, a Markov chain weather model, time-series windowing, and conditional plotting, all with NumPy, pandas and Matplotlib. A single-page dashboard presents the results.

## What's inside

| Part | Task | Topics |
|---|---|---|
| 1.1 | Covariance and correlation matrices | Linear algebra, statistics |
| 1.2 | Markov chain weather model | Probability, eigenvectors |
| 2.1 | Time-series windowing | Pandas rolling and expanding windows, NaN handling |
| 2.2 | Subplots with conditional colour and shading | Matplotlib, empirical rule |

### Results at a glance

- **1.1:** The covariance matrix of three simulated sensor channels (temperature, humidity, pressure; 100 readings each) is computed by matrix multiplication and matches `np.cov` to about 1e-16. Temperature and humidity are negatively correlated (r = -0.549).
- **1.2:** The steady state of the Sunny/Cloudy/Rainy chain, taken from the eigenvector of the transposed transition matrix, is π = (0.4565, 0.2826, 0.2609). A 100,000-step simulation agrees.
- **2.1:** A 7-day rolling mean and a 30-day-minimum expanding standard deviation leave 6 and 29 NaNs. They are filled column by column, and an `is_filled` flag marks the 29 affected rows.
- **2.2:** 6 of 365 days fall outside the mean ± 2σ band. That is fewer than the 4.55% the empirical rule predicts, because the seasonal cycle inflates σ.

## Files

| File | Purpose |
|---|---|
| `Assignment4_Advanced_Math_Python.ipynb` | The notebook with code, output and explanations |
| `Assignment4_Advanced_Math_Python.py` | The same code exported as a script; also writes `results.json` and `timeseries_2025.csv` |
| `dashboard.html` | Self-contained dashboard; open it in a browser |
| `results.json` | Values the dashboard displays, written by the script |
| `timeseries_2025.csv` | Simulated daily temperature for 2025 with rolling mean, expanding std, fill flag and outlier flag |
| `part1_cov_corr.png`, `part2_subplots.png` | Figures saved by the script |

## Run it

Requires Python 3 with `numpy`, `pandas` and `matplotlib`.

```bash
pip install numpy pandas matplotlib
python Assignment4_Advanced_Math_Python.py
```

The script uses fixed random seeds, so it reproduces the same numbers, figures and CSV each time. It opens the Matplotlib windows and rewrites `results.json`, the CSV and both PNGs.

## Dashboard

Open `dashboard.html` in a browser. It needs no server and no build step. The charts are drawn in plain SVG, and only the font loads from the internet. The data is embedded in the file, so after you change and re-run the script the dashboard has to be rebuilt to show the new numbers.

It has:

- a year-long temperature chart with toggleable layers, hover and arrow-key readout, and the outlier days marked
- a correlation and covariance heat map you can switch between
- the Markov transition matrix, a button that re-runs the 50-day simulation in the browser, and the convergence to the steady state
- a before/after view of how the missing values are filled
