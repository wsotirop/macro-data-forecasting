# Macro Data Forecasting Report

## Summary

- Generated timestamp: 2026-05-22T22:01:43+00:00
- Dataset path: data\processed\cpi_feature_matrix.csv
- Number of models: 3
- Forecast rows: 507
- Unique forecast timestamps: 169
- Target id: cpi_mom
- Forecast date range: 2012-03-16 to 2026-05-12

## Model Comparison

| model_name | n_forecasts | rmse | mae | directional_accuracy | beats_naive_rmse | beats_naive_mae | rmse_vs_naive | mae_vs_naive |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| naive_last_value | 169 | 0.304414 | 0.232139 | 0.470238 | 0 | 0 | 0 | 0 |
| ridge | 169 | 0.26643 | 0.205415 | 0.458333 | 1 | 1 | 0.0379845 | 0.0267237 |
| lightgbm | 169 | 0.301771 | 0.22735 | 0.404762 | 1 | 1 | 0.00264307 | 0.00478896 |

## Naive Benchmark Interpretation

- ridge beats the naive baseline on RMSE.
- ridge beats the naive baseline on MAE.
- lightgbm beats the naive baseline on RMSE.
- lightgbm beats the naive baseline on MAE.

## Plots

![Predictions vs Actuals](plots/predictions_vs_actuals.png)

![Forecast Errors](plots/forecast_errors.png)

![RMSE Comparison](plots/rmse_comparison.png)

![MAE Comparison](plots/mae_comparison.png)

## Forecast Output Summary

- Forecast timestamp range: 2012-03-16 to 2026-05-12

Forecast count by model:
| model_name | forecast_count |
| --- | --- |
| lightgbm | 169 |
| naive_last_value | 169 |
| ridge | 169 |

Actual summary:
| statistic | actual |
| --- | --- |
| mean | 0.223853 |
| std | 0.297512 |

Prediction summary by model:
| model_name | prediction_mean | prediction_std |
| --- | --- | --- |
| lightgbm | 0.213573 | 0.219485 |
| naive_last_value | 0.222044 | 0.296484 |
| ridge | 0.245547 | 0.255538 |

## Methodology

- Validation uses expanding-window walk-forward validation only.
- Each fold trains on rows strictly before the forecast row.
- No k-fold validation is used for macro time-series forecasting.
- Point-in-time features must satisfy `release_date <= forecast_timestamp`.
- `naive_last_value` predicts the prior known target value.
- `ridge` uses median imputation, standard scaling, and ridge regression.
- `lightgbm` uses median imputation and a fixed-default LightGBM regressor.

## Limitations

- Data quality depends on stored observations and release dates.
- CPI point-in-time work requires a complete official CPI release calendar.
- LightGBM is a fixed-default baseline and is not tuned.
- The current feature matrix uses latest available values only.
- Results may not beat the naive baseline.
