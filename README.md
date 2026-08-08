# Appliance Energy Forecasting

This repository contains a reproducible time-series forecasting pipeline for modelling and forecasting household appliance energy use.

The project uses the **Appliances Energy Prediction** dataset, which contains appliance energy consumption, indoor temperature and humidity sensor measurements, outdoor weather variables, and timestamp information. The aim is to compare simple benchmark models, a SARIMAX model, a feature-based machine-learning model, and a time-series foundation model on a common 24-hour-ahead forecasting task.

## Project aim

The aim of this assignment is to forecast short-term household appliance energy use and evaluate whether increasingly complex models improve on simple benchmark methods.

The main questions are:

1. How well do simple benchmark models forecast appliance energy use?
2. Does a SARIMAX model improve on the benchmark forecasts?
3. Do sensor, weather, and time-based covariates improve forecast accuracy?
4. Does a feature-based machine-learning model such as XGBoost improve performance?
5. Does a time-series foundation model such as Chronos, TimesFM, or TimeGPT provide any additional benefit?
6. Which model would be most suitable for a practical smart-home energy forecasting system?

## Dataset

The dataset used in this project is the **Appliances Energy Prediction** dataset (Candanedo, Feldheim and Deramaix, 2017), measured in a low-energy house in Stambruges, Belgium.

The target variable is:

```text
Appliances
```

This represents household appliance energy use in Wh for each time interval.

The original dataset is sampled every 10 minutes and contains variables including:

```text
date
Appliances
lights
T1, RH_1
T2, RH_2
T3, RH_3
T4, RH_4
T5, RH_5
T6, RH_6
T7, RH_7
T8, RH_8
T9, RH_9
T_out
Press_mm_hg
RH_out
Windspeed
Visibility
Tdewpoint
```

The `T` variables are indoor temperature measurements from different rooms or sensor locations. The `RH` variables are indoor relative humidity measurements. The outdoor weather variables include outdoor temperature, pressure, outdoor humidity, wind speed, visibility, and dew point. The two random variables `rv1` and `rv2` are dropped: they were included by the original authors purely as noise controls.

### Data acquisition

The primary source is the UCI Machine Learning Repository:

```text
https://archive.ics.uci.edu/ml/machine-learning-databases/00374/energydata_complete.csv
```

`src/appliance_energy/data.py` falls back automatically to the authors' own GitHub mirror if the UCI host is unreachable. Both files are byte-identical.

### Observed data quality

The raw file is complete, which is unusual and worth stating explicitly:

```text
19,735 rows, 11 Jan 2016 17:00 to 27 May 2016 18:00
0 missing cells
0 missing timestamps on the regular 10-minute grid
0 duplicated timestamps
```

No imputation is therefore required.

## Forecasting task

The main forecasting task is:

```text
Forecast appliance energy use over the next 24 hours.
```

The data are resampled to hourly means, so the forecast horizon is:

```python
horizon = 24
```

Aggregation to hourly means removes a large amount of high-frequency switching noise that no model can predict, and it makes SARIMA estimation with a seasonal period of 24 tractable. Averaging rather than summing is a linear rescaling by a factor of six and does not affect any scale-free comparison between models.

The test period is the final 14 complete days:

```python
test_steps = 14 * 24
```

The split is **snapped to whole calendar days**, so every 24-hour forecast block runs from 00:00 to 23:00 and is a genuine day-ahead forecast issued at midnight. This matters for the benchmark comparison: with unaligned blocks every forecast origin lands on the 18:00 demand peak, which makes the naive forecast look far worse than it is.

```text
train    2,935 hours   11 Jan 2016 17:00 - 12 May 2016 23:00
test       336 hours   13 May 2016 00:00 - 26 May 2016 23:00
holdout     18 hours   27 May 2016 00:00 - 27 May 2016 17:00
```

The 18-hour holdout is never used for fitting or evaluation and is reserved for post-sample commentary in the report.

Accuracy is measured over 14 consecutive, non-overlapping 24-hour blocks rather than from a single origin. A single 24-hour forecast is one draw from a noisy distribution; 14 rolling origins give a far more stable estimate at no extra cost. Model parameters are estimated once on the training sample and the state is updated with observed history at each new origin, mimicking an operational system that re-estimates infrequently but always conditions on the latest data.

## Models

The project compares the following model classes.

### 1. Benchmark models

```text
Mean forecast
Naive forecast
Daily seasonal naive forecast   (same hour yesterday,  lag 24)
Weekly seasonal naive forecast  (same hour last week,  lag 168)
Drift forecast
```

### 2. SARIMA / SARIMAX

The series is modelled on the natural-log scale: the level is strongly right-skewed (skew 2.39, falling to 1.00 in logs) and its variability scales with its level, so the log transform stabilises the variance and keeps back-transformed forecasts positive.

Model orders are selected by an exhaustive AIC loop over `p = [0, 6]`, `d = [0, 2]`, `q = [0, 6]`, run in two stages because a full seasonal search over all 147 combinations is computationally prohibitive:

```text
Stage 1  147 fits of ARIMA(p, d, q) on the seasonally differenced series
Stage 2  the best three (p, d, q) refitted as SARIMA(p, d, q)(P, 1, Q, 24)
```

All fits use the state-space representation, so the log-likelihood is evaluated on the same number of observations for every value of `d` and AIC values stay comparable across the grid.

The selected model is:

```python
order = (5, 0, 6)
seasonal_order = (0, 1, 1, 24)
```

Both a target-only SARIMA and a SARIMAX with exogenous regressors are fitted. Exogenous variables:

```text
T_out
RH_out
Windspeed
Tdewpoint
Press_mm_hg
hour_sin
hour_cos
dow_sin
dow_cos
is_weekend
```

### 3. Feature-based model

An XGBoost regressor is fitted in a **direct multi-step** formulation. Every target-derived feature is lagged by at least 24 hours, so a single global model maps the information set available at the origin to any lead time h = 1, ..., 24 without recursive substitution of its own predictions and without ever seeing an observation from inside the forecast window.

The feature table includes:

```text
Lagged appliance energy use     (lags 24, 25, 26, 48, 168)
Rolling means and standard deviations
Rolling max/min and same-hour 7-day mean
Hour-of-day features
Day-of-week features
Weekend indicator
Indoor temperature variables
Indoor humidity variables
Outdoor weather variables
```

Two variants are fitted, which is central to the covariate-availability question:

```text
operational  only variables genuinely known at the forecast origin
             (calendar terms, lagged appliance features, sensor/weather lagged 24 h)
conditional  additionally uses realised contemporaneous sensor and weather
             values from the test period
```

The target is modelled on the log scale and back-transformed with Duan's smearing estimator. Hyper-parameters are chosen on a validation tail (the final 14 days of the training sample); the test period is never used for tuning.

### 4. Foundation model

`src/appliance_energy/models/foundation.py` provides two back-ends:

```text
chronos  Amazon Chronos-Bolt, used zero-shot on the target series only
local    an offline fallback with the same window-to-horizon architecture,
         trained on the training split
```

The Chronos back-end is selected automatically when `chronos-forecasting` and `torch` are installed and the model hub is reachable. **The results in this repository were produced with the local fallback**, because the environment used to run the pipeline had no access to the HuggingFace model hub. The fallback is not a zero-shot foundation model and is reported as a proxy: it isolates the contribution of the architecture from the contribution of large-scale pretraining. To reproduce a true zero-shot result:

```bash
pip install chronos-forecasting torch
python scripts/run_pipeline.py --stages foundation final
```

The back-end actually used is recorded in `outputs/metrics/summary.json`.

## Feature sources

### Original measured variables

```text
Appliances
lights
indoor temperature variables
indoor humidity variables
outdoor weather variables
```

### Time-based features

```python
data["hour"] = data.index.hour
data["dayofweek"] = data.index.dayofweek
data["is_weekend"] = (data["dayofweek"] >= 5).astype(int)

data["hour_sin"] = np.sin(2 * np.pi * data["hour"] / 24)
data["hour_cos"] = np.cos(2 * np.pi * data["hour"] / 24)

data["dow_sin"] = np.sin(2 * np.pi * data["dayofweek"] / 7)
data["dow_cos"] = np.cos(2 * np.pi * data["dayofweek"] / 7)
```

### Lag and rolling features

```python
data["lag_24"] = data["Appliances"].shift(24)
data["lag_168"] = data["Appliances"].shift(168)

base = data["Appliances"].shift(24)          # last value known at the origin
data["roll_mean_24"] = base.rolling(24).mean()
data["roll_std_24"] = base.rolling(24).std()
```

The shift is applied **before** the rolling window and uses the forecast horizon, not `shift(1)`. A one-step shift is only safe for one-step-ahead forecasting; for a 24-hour horizon it silently leaks up to 23 hours of future information.

## Repository structure

```text
appliance-energy-forecasting/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── data/
│   ├── raw/                     energydata_complete.csv (not committed)
│   ├── interim/
│   └── processed/               energy_hourly.csv, features_hourly.csv
│
├── notebooks/
│   └── appliance_energy_forecasting_colab.ipynb   <- run this in Google Colab
│
├── src/
│   └── appliance_energy/
│       ├── __init__.py
│       ├── config.py            paths, split sizes, grids, feature lists
│       ├── data.py              download, clean, resample, split
│       ├── features.py          time, lag, rolling and sensor features
│       ├── eda.py               STL, ADF/KPSS, ACF/PACF, Ljung-Box
│       ├── evaluation.py        MAE, RMSE, MASE, Bias, sMAPE, Diebold-Mariano
│       ├── plotting.py          all report figures
│       ├── pipeline.py          staged, cached end-to-end pipeline
│       └── models/
│           ├── __init__.py
│           ├── benchmarks.py
│           ├── sarimax.py
│           ├── feature_models.py
│           └── foundation.py
│
├── scripts/
│   ├── appliance_energy_forecasting.py           <- single-file version of the notebook
│   ├── download_data.py
│   ├── make_features.py
│   ├── run_pipeline.py
│   ├── search_sarima_orders.py
│   └── evaluate_models.py
│
├── outputs/
│   ├── figures/                 fig01 ... fig12 (png)
│   ├── forecasts/               all_forecasts.csv and per-model forecasts
│   ├── metrics/                 model_comparison.csv, grids, diagnostics
│   └── model_objects/           cached SARIMAX fits
│
├── reports/
│   └── appliance_energy_forecasting_report.pdf / .docx
│
└── tests/
    ├── test_features.py
    ├── test_evaluation.py
    └── test_benchmarks.py
```

## Quick start in Google Colab

The fastest way to reproduce the whole analysis is the self-contained notebook:

```text
notebooks/appliance_energy_forecasting_colab.ipynb
```

Open it in Colab and choose *Runtime > Run all*. It downloads the data, runs
every model, prints the interpretation, and saves all figures and metrics to
`outputs/`. No GPU is required; a CPU runtime is sufficient.

```python
QUICK_MODE = False   # full 147-combination SARIMA search, ~25-40 min
QUICK_MODE = True    # reduced grid for a first pass, ~5 min
```

The same code is available as a plain script:

```bash
python scripts/appliance_energy_forecasting.py
```

## Installation

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Running the pipeline

```bash
python scripts/run_pipeline.py
```

The pipeline is staged and each stage caches its output, so it can be resumed or re-run in parts:

```bash
python scripts/run_pipeline.py --stages data bench
python scripts/run_pipeline.py --stages sarima
python scripts/run_pipeline.py --stages final
```

Stages, in order:

```text
data        load, clean, resample, EDA statistics and figures 1-5
bench       five benchmark forecasts over 14 rolling origins
sarima      target-only SARIMA, residual diagnostics, figure 7
sarimax     SARIMAX with exogenous regressors
xgb         operational and conditional XGBoost, ablation, figure 9
foundation  foundation model (Chronos if available, local fallback otherwise)
final       metrics, Diebold-Mariano tests, figures 6, 8, 10, 11, 12
```

The SARIMA order search is a separate, long-running step and is not part of the default run. Its results are committed in `outputs/metrics/`. To reproduce it:

```bash
python scripts/search_sarima_orders.py
```

Runtime note: SARIMA estimation with a seasonal period of 24 dominates the cost. On a single CPU core the full order search takes roughly 20 minutes and each SARIMAX fit 1-2 minutes; fitted results are cached in `outputs/model_objects/`.

## Outputs

Forecasts are written to:

```text
outputs/forecasts/all_forecasts.csv
```

containing the actual values and every model forecast:

```text
actual
mean
naive
seasonal_naive_daily
seasonal_naive_weekly
drift
sarima
sarimax
feature_model
feature_model_conditional
foundation_model
```

Model comparison metrics are written to:

```text
outputs/metrics/model_comparison.csv
```

with columns:

```text
model
MAE
RMSE
MASE
Bias
sMAPE
MAPE
```

Other metric files:

```text
stationarity_tests.csv      ADF and KPSS on five transformations
seasonal_strength.csv       STL-based daily and weekly seasonal strength
sarima_grid_stage1.csv      all 147 (p, d, q) combinations with AIC and BIC
sarima_grid_stage2.csv      seasonal order selection
sarimax_coefficients.csv    exogenous coefficients and p-values
ljung_box.csv               residual autocorrelation tests
feature_ablation.csv        incremental feature-group study
feature_importance.csv      XGBoost gain importance
block_rmse.csv              per-day RMSE for every model
diebold_mariano.csv         significance tests against the best benchmark
summary.json               all headline numbers quoted in the report
```

Figures are written to `outputs/figures/`:

```text
fig01_series_overview.png     fig07_sarima_diagnostics.png
fig02_profiles.png            fig08_sarima_forecast.png
fig03_stl.png                 fig09_feature_model.png
fig04_acf_pacf.png            fig10_foundation.png
fig05_correlations.png        fig11_comparison.png
fig06_benchmarks.png          fig12_error_diagnostics.png
```

## Evaluation metrics

All models are evaluated on the same 336-hour test period, over the same 14 rolling origins.

```text
MAE     scale-dependent, robust to the spiky peaks
RMSE    penalises the large peak-hour errors that matter for sizing
MASE    scaled by the in-sample seasonal naive MAE (53.00 Wh); MASE < 1 beats it
Bias    mean error, forecast minus actual; detects systematic under-forecasting
sMAPE   percentage error, symmetric and defined for this strictly positive series
```

Every model is compared against the **strongest benchmark**, not only against the other advanced models, and the comparison is tested for significance with a Diebold-Mariano test using the Harvey small-sample correction.

## Headline results

| Model | MAE | RMSE | MASE | Bias |
|---|---|---|---|---|
| SARIMA (5,0,6)(0,1,1,24) | 38.74 | **69.11** | 0.731 | -8.17 |
| SARIMAX (+ exogenous) | **38.06** | 69.18 | **0.718** | -9.39 |
| XGBoost (operational) | 39.96 | 70.13 | 0.754 | -9.51 |
| XGBoost (conditional) | 41.15 | 71.20 | 0.776 | -10.39 |
| Foundation-style model (local proxy) | 41.62 | 76.55 | 0.785 | -19.46 |
| Mean | 52.25 | 78.10 | 0.986 | -5.43 |
| Seasonal naive (weekly) | 44.94 | 82.80 | 0.848 | -13.06 |
| Seasonal naive (daily) | 51.20 | 89.29 | 0.966 | -5.34 |
| Naive | 50.97 | 90.55 | 0.962 | -38.26 |
| Drift | 50.99 | 90.56 | 0.962 | -38.22 |

Every model has MASE below 1, but the spread between the best model and the best benchmark is modest: appliance use at hourly resolution is dominated by unpredictable occupant behaviour.

## Data leakage

The following were actively guarded against, and the guards are covered by tests:

```text
Using future values of Appliances in lag or rolling features
Creating rolling features with shift(1) when the horizon is 24 hours
Scaling or imputing across the train-test boundary
Selecting SARIMA orders on a sample that overlaps the test window
Choosing the final model on test-set performance
```

Two points deserve emphasis.

First, model identification is itself a modelling decision. The SARIMA order search was re-run after the train/test split was realigned, because the original grid had been fitted on a sample that extended 18 hours into the test window. The order selected changed as a result.

Second:

```text
Future time-of-day and day-of-week variables are known in advance. Future
indoor sensor and weather variables are not known in a real operational
forecast. Where realised future sensor or weather values are used from the
test set, the result is a conditional forecast, not a true forecast.
```

Both variants are reported here, and the difference between them turns out to be small and in the wrong direction, which is itself the answer to that question.

## Tests

```bash
pytest
```

The suite covers:

```text
forecast lengths match the requested horizon
MASE and all error metrics are zero for a perfect forecast
lag and rolling features do not change when a future target value is altered
rolling benchmarks do not change when future test values are altered
the operational design matrix contains no contemporaneous sensor columns
the processed dataset has no missing or non-positive target values
```

## Reproducibility

```text
Random seeds are set for XGBoost and the neural fallback.
The pipeline runs from a fresh clone via scripts/run_pipeline.py.
Long-running fits are cached but always regenerable.
The foundation-model back-end actually used is recorded in summary.json.
```

Results may differ in the last significant figure across platforms because of BLAS-level floating-point differences in the SARIMA optimiser.

## References

Candanedo, L.M., Feldheim, V. and Deramaix, D. (2017) 'Data driven prediction models of energy use of appliances in a low-energy house', *Energy and Buildings*, 140, pp. 81-97.

Hyndman, R.J. and Athanasopoulos, G. (2021) *Forecasting: Principles and Practice*. 3rd edn. Melbourne: OTexts.

Hyndman, R.J. and Koehler, A.B. (2006) 'Another look at measures of forecast accuracy', *International Journal of Forecasting*, 22(4), pp. 679-688.

Ansari, A.F. et al. (2024) 'Chronos: Learning the language of time series', *Transactions on Machine Learning Research*.
