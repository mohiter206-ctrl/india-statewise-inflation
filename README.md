# India CPI Research Intelligence

A Streamlit research dashboard for monthly state-wise CPI analysis and comparison of statistical and machine-learning forecasting methods. The bundled analytical panel contains 28 State/UTs, three CPI sectors and 149 monthly observations per sector from January 2011 to May 2023.

## Run on Windows
Open PowerShell in this folder and run:

```powershell
py -m pip install -r requirements.txt
py -m streamlit run app.py
```

If using the project virtual environment without activating it:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
```

## Dashboard modules
1. Overview
2. Explore data (State explorer and exploratory analysis combined)
3. Compare states
4. Forecasting Lab (automatic model selection plus manual model evaluation/forecasting)
5. Model evaluation & research (metrics guidance, archived holdout and rolling-origin research results)
6. Data quality

## Forecasting methods
The Forecasting Lab includes 12 candidates: Naive, Seasonal Naive, 3-month Moving Average, Drift, Holt Linear, Theta, ETS (Holt-Winters), SARIMA, Random Forest, Extra Trees, Gradient Boosting (HistGradientBoostingRegressor) and XGBoost. The interactive lab runs series-specific expanding-window one-step rolling-origin comparisons. Its results are distinct from the archived fixed-origin holdout below.

## Independent fixed-origin 12-month holdout
The bundled holdout experiment evaluates all 12 methods for all 84 State/sector series (28 State/UTs × 3 sectors):
- Training data: January 2012–May 2022 (125 monthly YoY observations).
- Forecast origin: 31 May 2022.
- Holdout: June 2022–May 2023 (12 months).
- Forecasts are fixed-origin; actual holdout values are not used to update forecasts.
- Total: 1,008 model-series evaluations and 12,096 forecast rows; no failures recorded.

Aggregate results are in `Independent_12_Month_Holdout_Aggregate_Summary.csv`. Detailed per-series metrics, individual forecasts, best model per series and the full research report are provided in the correspondingly named `Independent_12_Month_Holdout_*.csv` and `.md` files. The Model Evaluation tab also displays the aggregate results and offers downloads.

In this single holdout period, Drift achieved the lowest mean RMSE (1.890 percentage points); Naive achieved the lowest mean MAE (1.606 pp) and was best by RMSE in 18 of 84 series. Drift was best in 16 series and SARIMA in 13. There is no universal winner. These results are specific to this dataset, forecast origin, horizon and fixed model configurations. They are not repeated multi-origin validation.

## Data definition and limitations
- Target: monthly year-on-year inflation calculated from CPI General Index (base 2012=100) as `100 × (CPI[t] / CPI[t-12] − 1)`.
- The supplied CPI panel covers 28 retained State/UT columns; it is not an all-India series or a complete 36-State/UT panel.
- The final analytical panel contains 28 State/UTs and has no blank CPI cells. Some observations were imputed during earlier data preparation; see the imputation audit.
- The cleaned source already contains values imputed before this app was created. The retained-state audit file `CPI_Imputation_Audit_28_States.csv` contains 356 log records. Zero missing values does not mean every value is an original observation.
- The dataset ends in May 2023. The holdout results do not verify performance on later actual inflation values.
- The holdout is one fixed-origin period. A separate repeated-origin evaluation and later external validation would strengthen the research.
- Hyperparameters were fixed rather than tuned independently for each State/sector. Tree models used 60 estimators for Random Forest, Extra Trees and XGBoost; Gradient Boosting used 200 iterations in the holdout.
- The app's empirical forecast bands are not formally calibrated prediction intervals. Forecast error metrics alone do not establish interval coverage.
- State averages in the overview are unweighted and are not official all-India CPI.

## Upload and combine
The sidebar accepts CSV or Excel. Preferred wide layout: `Sector`, `Year`, `Name`, then one column per State/UT. A long layout is also accepted with `Date`, `Sector`, `State/UT`, `CPI`. The user must select and confirm compatible CPI base years before merging. Upload merge conflicts are logged; uploads are session-based and do not overwrite the bundled source file.

## Validation notes
The project package includes syntax validation, dataset checks and the archived holdout outputs. A successful Python syntax check does not constitute a full browser end-to-end test of every upload, download and session-state workflow on every Windows setup. Install the listed dependencies in the target environment and verify the app locally before relying on it for submission.
\n## Manual forecasting mode\nIn the Forecasting Lab, use the manual evaluation section to choose one of the 12 individual models, run its rolling-origin evaluation, and generate a recursive forecast using that same model. This runs independently of automatic model selection. The automatic evaluation and forecasting workflow remains available.\n

## Statistical audit update
The Forecasting Lab keeps automatic and manual modes. The research tab now includes fixed-holdout diagnostics for the currently selected State/sector, including horizon-block RMSE from the single archived holdout origin. Bias is standardised as forecast minus actual (positive = over-forecast). See `Maharashtra_Rural_Audit_Findings.md` for the current independent audit note. Horizon-block values from one origin are descriptive only, not repeated-origin validation.


## Active dataset workflow (updated)
- The original bundled CPI CSV is preserved and never overwritten.
- Uploaded compatible data is merged into a separate combined dataset and conflicting cells are logged.
- Optional short-gap interpolation creates a separate candidate cleaned dataset.
- The user must click **Apply cleaned dataset and set as active** before forecasts and other analytical tabs use it.
- All analytical tabs read from the same active dataset. The user can switch back to the current combined/original dataset from Data Quality.
- Forecast outputs should be interpreted with the dataset version and selected series used to generate them.

## Updating a GitHub-hosted website
Yes. You can update the repository whenever you improve the app: commit and push the changed files to GitHub. If the website is deployed from that repository (for example, Streamlit Community Cloud), the deployment can rebuild from the updated commit. Keep secrets out of GitHub, preserve dataset provenance, and use Git commits/tags so earlier research versions can be recovered.

