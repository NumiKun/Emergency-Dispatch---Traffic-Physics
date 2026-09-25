# model_artifacts - Knowledge Store

This directory contains all serialized outputs produced by the notebook
`road_closure_duration_prediction.ipynb`. Every file here is generated
programmatically and can be fully reproduced by re-running the notebook
from the top with the same `RANDOM_SEED = 42`.

---

## File Inventory

| File | Type | Description |
|---|---|---|
| `best_pipeline.joblib` | Binary | Full sklearn Pipeline (preprocessor + best model). Load with `joblib.load()` to run inference without re-fitting. |
| `model_metadata.json` | JSON | Complete experiment record: feature lists, train/val/test sizes, best hyperparameters, CV config, winsorization bounds, final test metrics, and full benchmark comparison table. |
| `shap_feature_importance.json` | JSON | Ranked dictionary of mean absolute SHAP values per feature. Used for downstream dashboards or lightweight feature audits. |
| `benchmark_results.csv` | CSV | Cross-model comparison table (Ridge, Random Forest, XGBoost, LightGBM) with CV RMSE, validation RMSE/MAE/MAPE/R2. |
| `eda_overview.png` | PNG | 2x3 EDA figure: target distribution (raw + log1p), median closure by severity, median closure by collision type, boxplot by area type, and numerical correlation heatmap. |
| `benchmark_comparison.png` | PNG | Bar chart comparison of RMSE, MAE, and R2 across all four baseline models on the validation set. |
| `tuning_convergence.png` | PNG | Scatter + line plot tracking best CV RMSE across all 60 Randomized Search iterations. |
| `final_evaluation.png` | PNG | Three-panel test-set evaluation: Actual vs Predicted scatter, Residuals vs Predicted, and Residual distribution histogram. |
| `error_stratification.png` | PNG | MAE and RMSE grouped by Accident_Severity to identify high-cost prediction subgroups. |
| `shap_beeswarm.png` | PNG | SHAP beeswarm plot showing global feature impact across the entire test set (top 20 features). |
| `shap_bar_importance.png` | PNG | Horizontal bar chart of top 20 features ranked by mean absolute SHAP value. |
| `shap_waterfall_worst.png` | PNG | SHAP waterfall plot for the single highest-error test instance. |

---

## How to Load the Pipeline for Inference

`python
import joblib
import numpy as np

pipeline = joblib.load('model_artifacts/best_pipeline.joblib')

# X_new must be a DataFrame with the same ALL_FEATURES column list
y_pred_log = pipeline.predict(X_new)
y_pred_min = np.expm1(y_pred_log)
`

---

## Target Variable Notes

- **Target:** `Road_Closure_Duration_min`
- **Training scope:** Only records where `Accident_Occurred == 1` (8,231 records)
- **Target transform:** `log1p` applied before training; use `np.expm1()` to back-transform.
- **Winsorization:** Target capped at the 1st and 99th percentile. Exact bounds in `model_metadata.json`.

---

## Physics Feature Reference

| Feature | Derivation |
|---|---|
| `friction_coeff` | Mapped from Road_Condition via FRICTION_MAP |
| `stopping_distance_m` | v_ms^2 / (2 * friction_coeff * 9.81) |
| `kinetic_energy_kJ` | 0.5 * vehicle_mass_kg * v_ms^2 / 1000 |
| `speed_deviation` | Recorded_Speed_kmh - Speed_Limit_kmh |
| `speed_deviation_sq` | speed_deviation^2 |
| `response_delay` | Actual_Response_Time_min - Baseline_Expected_Response_Time_min |
| `driver_inexperience` | 1 / (Driver_Experience_yrs + 1) |
| `low_visibility_flag` | 1 if Visibility_m < 500 else 0 |
| `hour_sin`, `hour_cos` | sin/cos(2*pi*Hour/24) |
| `month_sin`, `month_cos` | sin/cos(2*pi*Month/12) |
